from typing import Optional, Callable, Literal
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce
from ..backbones import (
    Unet3D,
    DiT3D,
    DiT3DPose,
    UViT3D,
    UViT3DPose,
)
from .noise_schedule import make_beta_schedule
from ..history_guidance import (
    HistoryGuidanceManager,
    SimpleHistoryGuidanceManager,
    GeneralizedHistoryGuidanceManager,
)


def extract(a, t, x_shape):
    shape = t.shape
    out = a[t]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)


class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.x_shape = x_shape
        self.max_tokens = max_tokens
        self.external_cond_dim = external_cond_dim
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_schedule = cfg.beta_schedule
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.objective = cfg.objective
        self.loss_weighting = cfg.loss_weighting
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise

        self.backbone_cfg = backbone_cfg
        self.use_causal_mask = cfg.use_causal_mask
        self._build_model()
        self._build_buffer()

    def _build_model(self):
        match self.backbone_cfg.name:
            case "u_net3d":
                model_cls = Unet3D
            case "u_vit3d":
                model_cls = UViT3D
            case "u_vit3d_pose":
                model_cls = UViT3DPose
            case "dit3d":
                model_cls = DiT3D
            case "dit3d_pose":
                model_cls = DiT3DPose
            case _:
                raise ValueError(f"unknown model type {self.model_type}")
        self.model = model_cls(
            cfg=self.backbone_cfg,
            x_shape=self.x_shape,
            max_tokens=self.max_tokens,
            external_cond_dim=self.external_cond_dim,
            use_causal_mask=self.use_causal_mask,
        )

    def _build_buffer(self):
        betas = make_beta_schedule(
            schedule=self.beta_schedule,
            timesteps=self.timesteps,
            zero_terminal_snr=self.objective != "pred_noise",
            **self.schedule_fn_kwargs,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32), persistent=False
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        # if (
        #     self.objective == "pred_noise"
        #     or self.cfg.reconstruction_guidance is not None
        # ):
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer("snr", snr)
        if self.loss_weighting.strategy in {"min_snr", "fused_min_snr"}:
            clipped_snr = snr.clone()
            clipped_snr.clamp_(max=self.loss_weighting.snr_clip)
            register_buffer("clipped_snr", clipped_snr)
        elif self.loss_weighting.strategy == "sigmoid":
            register_buffer("logsnr", torch.log(snr))

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(self, x, k, external_cond=None, external_cond_mask=None):
        model_output = self.model(x, k, external_cond, external_cond_mask)

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def predict_noise_from_start(self, x_k, k, x0):
        # return (
        #     extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        # ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (x_k - extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0) / extract(
            self.sqrt_one_minus_alphas_cumprod, k, x_k.shape
        )

    def predict_v(self, x_start, k, noise):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )

    def q_mean_variance(self, x_start, k):
        mean = extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, k, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, k, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_k, k):
        posterior_mean = (
            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = extract(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, k, x_k.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, k, noise=None, generator=None):
        if noise is None:
            if generator is None:
                noise = torch.randn_like(x_start)
            else:
                noise = torch.randn(
                    *x_start.shape, generator=generator, device=x_start.device
                )
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    def p_mean_variance(self, x, k, external_cond=None, external_cond_mask=None):
        model_pred = self.model_predictions(
            x=x, k=k, external_cond=external_cond, external_cond_mask=external_cond_mask
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_k=x, k=k)

    def compute_loss_weights(
        self,
        k: torch.Tensor,
        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
    ) -> torch.Tensor:
        if strategy == "uniform":
            return torch.ones_like(k)
        snr = self.snr[k]
        epsilon_weighting = None
        match strategy:
            case "sigmoid":
                logsnr = self.logsnr[k]
                # sigmoid reweighting proposed by https://arxiv.org/abs/2303.00848
                # and adopted by https://arxiv.org/abs/2410.19324
                epsilon_weighting = torch.sigmoid(
                    self.cfg.loss_weighting.sigmoid_bias - logsnr
                )
            case "min_snr":
                # min-SNR reweighting proposed by https://arxiv.org/abs/2303.09556
                clipped_snr = self.clipped_snr[k]
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case "fused_min_snr":
                # fused min-SNR reweighting proposed by Diffusion Forcing v1
                # with an additional support for bi-directional Fused min-SNR for non-causal models
                snr_clip, cum_snr_decay = (
                    self.loss_weighting.snr_clip,
                    self.loss_weighting.cum_snr_decay,
                )
                clipped_snr = self.clipped_snr[k]
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip

                def compute_cum_snr(reverse: bool = False):
                    new_normalized_clipped_snr = (
                        normalized_clipped_snr.flip(1)
                        if reverse
                        else normalized_clipped_snr
                    )
                    cum_snr = torch.zeros_like(new_normalized_clipped_snr)
                    for t in range(0, k.shape[1]):
                        if t == 0:
                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                        else:
                            cum_snr[:, t] = (
                                cum_snr_decay * cum_snr[:, t - 1]
                                + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
                            )
                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum_snr.flip(1) if reverse else cum_snr

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    # bi-directional cum_snr when not using causal mask
                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
                    cum_snr *= 0.5
                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
                    1 - normalized_clipped_snr
                )
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                clipped_snr = clipped_fused_snr * snr_clip
                snr = fused_snr * snr_clip
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case _:
                raise ValueError(f"unknown loss weighting strategy {strategy}")

        match self.objective:
            case "pred_noise":
                return epsilon_weighting
            case "pred_x0":
                return epsilon_weighting * snr
            case "pred_v":
                return epsilon_weighting * snr / (snr + 1)
            case _:
                raise ValueError(f"unknown objective {self.objective}")

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
    ):
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.q_sample(x_start=x, k=k, noise=noise)
        model_pred = self.model_predictions(
            x=noised_x, k=k, external_cond=external_cond
        )

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, k, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")

        loss_weight = self.compute_loss_weights(k, self.loss_weighting.strategy)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss

    def ddim_idx_to_noise_level(
        self,
        indices: torch.Tensor,
        timestep_max: int = None,
        timestep_min: int = None,
        sampling_timesteps: int = None,
    ):
        shape = indices.shape

        if timestep_max is None:
            timestep_max = self.timesteps - 1
        if timestep_min is None:
            timestep_min = -1
        if sampling_timesteps is None:
            sampling_timesteps = self.sampling_timesteps

        # real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = torch.linspace(timestep_min, timestep_max, sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    def sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        update_where_noise_level_same: Optional[bool] = False,
    ):
        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                guidance_fn=guidance_fn,
                update_where_noise_level_same=update_where_noise_level_same,
            )

        # FIXME: temporary code for checking ddpm sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            curr_noise_level=curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
            guidance_fn=guidance_fn,
        )

    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("guidance_fn is not yet implmented for ddpm.")

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            k=clipped_curr_noise_level,
            external_cond=external_cond,
            external_cond_mask=external_cond_mask,
        )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred)

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        update_where_noise_level_same: Optional[bool] = False,
    ):

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]

        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        # sigma fixed small
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )

        c = (1 - alpha_next - sigma**2).sqrt()

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                )

                guidance_loss = guidance_fn(
                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )

                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]
                grad = torch.nan_to_num(grad, nan=0.0)

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,  # to avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(
                        x, clipped_curr_noise_level, pred_noise
                    ),
                    model_pred.pred_x_start,
                )

        else:
            model_pred = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        if not update_where_noise_level_same:
            mask = curr_noise_level == next_noise_level
            x_pred = torch.where(
                self.add_shape_channels(mask),
                x,
                x_pred,
            )

        return x_pred

    def sample_step_windows_history_guidance(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        num_windows: Optional[int] = None,
        update_where_noise_level_same: Optional[bool] = False,
        noise: Optional[torch.Tensor] = None,
        minibatch_size: Optional[int] = None,
        history_guidance_manager: Optional[
            HistoryGuidanceManager
            | SimpleHistoryGuidanceManager
            | GeneralizedHistoryGuidanceManager
        ] = None,
        max_stoch: Optional[bool] = False,
    ):
        if self.is_ddim_sampling:
            return self.ddim_sample_step_windows_history_guidance(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                external_cond=external_cond,
                external_cond_mask=external_cond_mask,
                guidance_fn=guidance_fn,
                num_windows=num_windows,
                update_where_noise_level_same=update_where_noise_level_same,
                noise=noise,
                minibatch_size=minibatch_size,
                history_guidance_manager=history_guidance_manager,
                max_stoch=max_stoch,
            )
        else:
            raise NotImplementedError(
                "ddpm sampling is not yet implemented. It can instead be realized by setting eta = 1.0, max_stoch = False, and sampling_timesteps = timesteps."
            )

    def ddim_sample_step_windows_history_guidance(
        self,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        external_cond_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        num_windows: Optional[int] = None,
        update_where_noise_level_same: Optional[bool] = False,
        noise: Optional[torch.Tensor] = None,
        minibatch_size: Optional[int] = None,
        history_guidance_manager: Optional[
            HistoryGuidanceManager
            | SimpleHistoryGuidanceManager
            | GeneralizedHistoryGuidanceManager
        ] = None,
        max_stoch: Optional[bool] = False,
    ):

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )

        if max_stoch:
            sigma_squared_base = 1 - alpha_next
        else:
            # sigma fixed small
            sigma_squared_base = (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            )

        # shape = (B * nfe * windows, self.max_tokens)
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta * sigma_squared_base.sqrt(),
            # * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = torch.clamp(
            1 - alpha_next - (self.ddim_sampling_eta**2) * sigma_squared_base, min=0
        ).sqrt()

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None:
            raise NotImplementedError(
                "guidance_fn is not yet implemented for ddim_sample_step_windows_history_guidance."
            )
        else:

            if minibatch_size is None:
                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    external_cond=external_cond,
                    external_cond_mask=external_cond_mask,
                )
            else:
                # split inputs into minibatches to avoid OOM
                # x.shape = (b, ...)
                # clipped_curr_noise_level.shape = (b, ...)
                # external_cond.shape = (b, ...)
                # external_cond_mask.shape = (b, ...)

                assert (
                    minibatch_size < x.shape[0]
                ), "minibatch_size {} should be less than the original batch size of batch_size * nfe * num_windows = {}".format(
                    minibatch_size, x.shape[0]
                )

                x_splits = torch.split(x, minibatch_size, dim=0)
                clipped_curr_noise_level_splits = torch.split(
                    clipped_curr_noise_level, minibatch_size, dim=0
                )
                external_cond_splits = torch.split(external_cond, minibatch_size, dim=0)

                if external_cond_mask is not None:
                    external_cond_mask_splits = torch.split(
                        external_cond_mask, minibatch_size, dim=0
                    )
                else:
                    external_cond_mask_splits = [None for _ in range(len(x_splits))]

                model_pred_splits = [
                    self.model_predictions(
                        x_split,
                        k=clipped_curr_noise_level_split,
                        external_cond=external_cond_split,
                        external_cond_mask=external_cond_mask_split,
                    )
                    for x_split, clipped_curr_noise_level_split, external_cond_split, external_cond_mask_split in zip(
                        x_splits,
                        clipped_curr_noise_level_splits,
                        external_cond_splits,
                        external_cond_mask_splits,
                    )
                ]
                pred_noise = torch.cat(
                    [
                        model_pred_split.pred_noise
                        for model_pred_split in model_pred_splits
                    ],
                    dim=0,
                )
                pred_x_start = torch.cat(
                    [
                        model_pred_split.pred_x_start
                        for model_pred_split in model_pred_splits
                    ],
                    dim=0,
                )
                model_out = torch.cat(
                    [
                        model_pred_split.model_out
                        for model_pred_split in model_pred_splits
                    ],
                    dim=0,
                )

                model_pred = ModelPrediction(
                    pred_noise=pred_noise,
                    pred_x_start=pred_x_start,
                    model_out=model_out,
                )

            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        if noise is None:
            noise = torch.randn_like(x)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred_without_sigma_noise = x_start * alpha_next.sqrt() + pred_noise * c

        # apply history guidance to model predictions
        if history_guidance_manager is not None:

            # shape = (b * nfe, windows, self.max_tokens, self.x_shape)
            x_pred_without_sigma_noise = rearrange(
                x_pred_without_sigma_noise,
                "(b windows) max_tokens ... -> b (windows max_tokens) ...",
                b=x_pred_without_sigma_noise.shape[0] // num_windows,
                windows=num_windows,
            )

            x_pred_without_sigma_noise = history_guidance_manager.compose(
                x_pred_without_sigma_noise
            )

            # shape = (b, windows, self.max_tokens, self.x_shape)
            x_pred_without_sigma_noise = rearrange(
                x_pred_without_sigma_noise,
                "b (windows max_tokens) ... -> (b windows) max_tokens ...",
                windows=num_windows,
                max_tokens=self.max_tokens,
            )

            x_start = rearrange(
                x_start,
                "(b windows) max_tokens ... -> b (windows max_tokens) ...",
                b=x_start.shape[0] // num_windows,
                windows=num_windows,
            )

            x_start = history_guidance_manager.compose(x_start)

            x_start = rearrange(
                x_start,
                "b (windows max_tokens) ... -> (b windows) max_tokens ...",
                windows=num_windows,
                max_tokens=self.max_tokens,
            )

            nfe = history_guidance_manager.nfe

            # add sigma noise
            sigma = rearrange(
                sigma,
                "(b nfe windows) ... -> b nfe windows ...",
                b=sigma.shape[0] // (num_windows * nfe),
                nfe=nfe,
                windows=num_windows,
            )

            # only keep the first sigma noise term corresponding to the fully conditional score
            noise = rearrange(
                noise,
                "(b nfe windows) ... -> b nfe windows ...",
                b=noise.shape[0] // (num_windows * nfe),
                nfe=nfe,
                windows=num_windows,
            )
            sigma_noise = sigma * noise
            sigma_noise = sigma_noise[
                :, -1
            ]  # sigma noise corresponding to the fully conditional score
            sigma_noise = rearrange(sigma_noise, "b windows ... -> (b windows) ...")
        else:
            sigma_noise = sigma * noise

        x_pred = x_pred_without_sigma_noise + sigma_noise

        # only update frames where the noise level decreases
        if not update_where_noise_level_same:
            mask = curr_noise_level == next_noise_level
            # assert that mask is same for all nfe's
            mask = rearrange(
                mask,
                "(b nfe windows) ... -> b nfe windows ...",
                b=mask.shape[0] // (num_windows * nfe),
                nfe=nfe,
                windows=num_windows,
            )
            mask = mask[:, -1]
            mask = rearrange(mask, "b windows ... -> (b windows) ...")

            x = rearrange(
                x,
                "(b nfe windows) ... -> b nfe windows ...",
                b=x.shape[0] // (num_windows * nfe),
                nfe=nfe,
                windows=num_windows,
            )
            x = x[:, -1]
            x = rearrange(x, "b windows ... -> (b windows) ...")

            x_pred = torch.where(
                self.add_shape_channels(mask),
                x,
                x_pred,
            )

        return x_pred, x_start

    def estimate_noise_level(self, x, mu=None):
        # x ~ ( B, T, C, ...)
        if mu is None:
            mu = torch.zeros_like(x)
        x = x - mu
        mse = reduce(x**2, "b t ... -> b t", "mean")
        ll_except_c = -self.log_one_minus_alphas_cumprod[None, None] - mse[
            ..., None
        ] * self.alphas_cumprod[None, None] / (1 - self.alphas_cumprod[None, None])
        k = torch.argmax(ll_except_c, -1)
        return k
