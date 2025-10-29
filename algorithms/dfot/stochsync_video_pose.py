from typing import Optional
from jaxtyping import Float
from omegaconf import DictConfig
import torch
from torch import Tensor
from einops import rearrange, repeat
from tqdm import tqdm
from utils.logging_utils import log_video
from .history_guidance import HistoryGuidance
from .dfot_video_pose import DFoTVideoPose


class StochSyncVideoPose(DFoTVideoPose):
    """
    A child class of DFoTVideoPose that implements Stochastic Synchronization (StochSync), a diffusion stitching method for images such as 360-degree panoramas and 3D mesh texturing (https://stochsync.github.io).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _predict_videos(
        self,
        xs: Float[Tensor, "B T C H W"],
        conditions: Optional[Float[Tensor, "B T ..."]] = None,
    ) -> Tensor:

        # 1) initialize chunks with Gaussian noise, unless they are the context frames
        # note that xs, conditions are of shape (B, T, ...), where xs contains all ground truth frames and we just want to condition on the first n_context_frames as context
        batch_size = xs.shape[0]
        target_length = xs.shape[1]
        assert (
            target_length % self.max_tokens == 0
        ), "target_length must be divisible by max_tokens"

        xs_pred = torch.randn(
            (xs.shape[0], target_length + self.max_tokens, *self.x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        # extend conditions
        conditions = torch.cat(
            [
                conditions[:, : self.max_tokens // 2],
                conditions,
                conditions[:, -(self.max_tokens - self.max_tokens // 2) :],
            ],
            dim=1,
        )

        # xs_pred is a padded version of the placeholder for our generated video of length xs.shape[1]
        # it has padding of length self.max_tokens // 2 on the left and (self.max_tokens - self.max_tokens // 2) on the right
        # e.g. if self.max_tokens = 9, then the padding is 4 on the left and 5 on the right
        # the first 4 frames of xs_pred are padded with 4 frames of noise
        # the last 5 frames of xs_pred are padded with 5 frames of noise
        # the frames in between are the placeholder for our generated video
        xs_pred[
            :, self.max_tokens // 2 : self.max_tokens // 2 + self.n_context_tokens
        ] = xs[:, : self.n_context_tokens].clone()

        # a binary mask that indicates which tokens are context (either ground-truth or fully denoised) and which ones need to be denoised
        # possible values:
        # -1: padding
        # 0: to be denoised
        # 1: ground-truth context
        # 2: fully denoised
        context_mask = torch.zeros(xs_pred.shape[:2], device=self.device)
        context_mask[
            :, self.max_tokens // 2 : self.max_tokens // 2 + self.n_context_tokens
        ] = 1

        # padding on the left-most and right-most sides of the sequence
        context_mask[:, : self.max_tokens // 2] = -1
        context_mask[:, -(self.max_tokens - self.max_tokens // 2) :] = -1

        # create sliding windows, each comprised of contiguous indices of length self.max_tokens
        # the array is of shape (B, num_windows, self.max_tokens)
        # first dimension = number of batches
        # second dimension = number of sliding windows
        # third dimension = length of the sliding window
        indices_even_iter = (
            torch.arange(target_length, device=self.device) + self.max_tokens // 2
        )
        num_windows_even_iter = target_length // self.max_tokens

        if self.cfg.tasks.prediction.loop_closing:
            indices_odd_iter = (
                torch.remainder(
                    torch.arange(target_length, device=self.device)
                    + self.max_tokens // 2,
                    target_length,
                )
                + self.max_tokens // 2
            )
            num_windows_odd_iter = target_length // self.max_tokens
        else:
            indices_odd_iter = torch.arange(
                target_length + self.max_tokens, device=self.device
            )
            num_windows_odd_iter = target_length // self.max_tokens + 1

        indices_even_iter = rearrange(
            indices_even_iter,
            "(windows max_tokens) -> windows max_tokens",
            windows=num_windows_even_iter,
            max_tokens=self.max_tokens,
        )
        indices_odd_iter = rearrange(
            indices_odd_iter,
            "(windows max_tokens) -> windows max_tokens",
            windows=num_windows_odd_iter,
            max_tokens=self.max_tokens,
        )

        indices_even_iter = repeat(
            indices_even_iter,
            "windows max_tokens -> b windows max_tokens",
            b=batch_size,
        )
        indices_odd_iter = repeat(
            indices_odd_iter, "windows max_tokens -> b windows max_tokens", b=batch_size
        )

        # merge the batch dimension with the chunk dimension to denoise all chunks in parallel
        indices_even_iter = rearrange(
            indices_even_iter, "b windows max_tokens -> b (windows max_tokens)"
        )  # shape = (B, num_windows_even_iter*self.max_tokens)
        indices_odd_iter = rearrange(
            indices_odd_iter, "b windows max_tokens -> b (windows max_tokens)"
        )  # shape = (B, num_windows_odd_iter*self.max_tokens)

        if self.cfg.tasks.prediction.two_loops:
            assert (
                self.cfg.tasks.prediction.loop_closing
            ), "two loops are only supported for loop closing"
            assert self.max_tokens % 2 == 0, "max_tokens must be even for two loops"
            assert target_length % 2 == 0, "target_length must be even for two loops"
            indices_sync_loops = torch.arange(target_length // 2, device=self.device)
            num_windows_sync_loops = target_length // self.max_tokens
            indices_sync_loops = rearrange(
                indices_sync_loops,
                "(windows max_tokens) -> windows max_tokens",
                windows=num_windows_sync_loops,
                max_tokens=self.max_tokens // 2,
            )
            indices_sync_loops = torch.cat(
                [
                    indices_sync_loops,
                    torch.flip(
                        torch.remainder(
                            indices_sync_loops + target_length // 2, target_length
                        ),
                        dims=[1],
                    ),
                ],
                dim=1,
            )
            indices_sync_loops = indices_sync_loops + self.max_tokens // 2
            indices_sync_loops = repeat(
                indices_sync_loops,
                "windows max_tokens -> b windows max_tokens",
                b=batch_size,
            )
            indices_sync_loops = rearrange(
                indices_sync_loops, "b windows max_tokens -> b (windows max_tokens)"
            )

        history_guidance = HistoryGuidance.from_config(
            config=self.cfg.tasks.prediction.history_guidance,
            timesteps=self.timesteps,
        )

        # used to define t_curr (i.e. t_src in the multistep loop)
        scheduling_matrix_multistep_start = self._generate_scheduling_matrix(
            xs_pred.shape[1],
            timestep_max=self.cfg.tasks.prediction.timestep_max,
            timestep_min=self.cfg.tasks.prediction.timestep_min,
            warmup_steps=self.cfg.tasks.prediction.warmup_steps,
        )  # shape = (num_denoising_steps, target_length)
        scheduling_matrix_multistep_start = scheduling_matrix_multistep_start.to(
            self.device
        )

        # used to define intermediate timesteps in the multistep loop when seam removal is not enabled
        scheduling_matrix_multistep_intermediate = self._generate_scheduling_matrix(
            xs_pred.shape[1], ode_steps=self.cfg.tasks.prediction.ode_steps
        )  # shape = (num_denoising_steps, target_length)
        scheduling_matrix_multistep_intermediate = (
            scheduling_matrix_multistep_intermediate.to(self.device)
        )

        # sued to define intermediate timesteps in the multistep loop when seam removal is enabled (ode_steps = 50)
        scheduling_matrix_multistep_intermediate_seam_removal = (
            self._generate_scheduling_matrix(xs_pred.shape[1], ode_steps=50)
        )  # shape = (num_denoising_steps, target_length)
        scheduling_matrix_multistep_intermediate_seam_removal = (
            scheduling_matrix_multistep_intermediate_seam_removal.to(self.device)
        )

        # let's now create a list of scheduling matrices, each of which denote the series of timesteps that start at scheduling_matrix_multistep_start[m] and then proceeds along the portion of scheduling_matrix_multistep_intermediate > scheduling_matrix_multistep_start[m]
        # this will be a list of scheduling_matrix_multistep_start.shape[0] scheduling matrices
        scheduling_matrices_multistep = []

        if self.cfg.tasks.prediction.seam_removal_steps is None:
            seam_removal_steps = 0
        else:
            seam_removal_steps = self.cfg.tasks.prediction.seam_removal_steps
        seam_removal_step_counter = 0

        for m in range(scheduling_matrix_multistep_start.shape[0] - 1, -1, -1):

            if torch.all(scheduling_matrix_multistep_start[m : m + 1] == -1):
                continue

            if seam_removal_step_counter < seam_removal_steps:
                seam_removal_step_counter += 1
                # create concatenated scheduling matrix
                # shape = (51, target_length)
                scheduling_matrix_multistep = torch.cat(
                    [
                        scheduling_matrix_multistep_start[m : m + 1],
                        scheduling_matrix_multistep_intermediate_seam_removal,
                    ],
                    dim=0,
                )
            else:
                # create concatenated scheduling matrix
                # shape = (ode_steps + 1, target_length)
                scheduling_matrix_multistep = torch.cat(
                    [
                        scheduling_matrix_multistep_start[m : m + 1],
                        scheduling_matrix_multistep_intermediate,
                    ],
                    dim=0,
                )

            assert torch.allclose(
                torch.abs(
                    scheduling_matrix_multistep.float()
                    - scheduling_matrix_multistep.float().mean(dim=-1, keepdim=True)
                ),
                torch.zeros_like(scheduling_matrix_multistep.float()),
                atol=1e-3,
            ), "all noise levels across all frames must be the same in this implementation"

            # sort along dim = 0 in descending order
            scheduling_matrix_multistep, indices = torch.sort(
                scheduling_matrix_multistep, dim=0, descending=True
            )

            # find the index where scheduling_matrix_multistep_start was sorted to
            index_start = torch.where(indices[:, 0] == 0)[0]

            assert torch.allclose(
                scheduling_matrix_multistep[index_start : index_start + 1].float(),
                scheduling_matrix_multistep_start[m : m + 1].float(),
            ), "scheduling_matrix_multistep_start[m] was not sorted to the correct position"
            scheduling_matrix_multistep = scheduling_matrix_multistep[
                index_start : scheduling_matrix_multistep.shape[0]
            ]

            # post-processing of the scheduling matrices
            # 1) repeat them along batch dimension
            scheduling_matrix_multistep = repeat(
                scheduling_matrix_multistep, "m t -> m b t", b=batch_size
            )

            # 2) if not self.is_full_sequence,  fill in the context token's noise levels as -1 in scheduling matrix and fill the padding token's noise levels as max in scheduling matrix
            if not self.is_full_sequence:
                scheduling_matrix_multistep = torch.where(
                    context_mask[None] >= 1, -1, scheduling_matrix_multistep
                )
                scheduling_matrix_multistep = torch.where(
                    context_mask[None] == -1,
                    self.timesteps - 1,
                    scheduling_matrix_multistep,
                )

            # 3) reshape
            scheduling_matrix_multistep = rearrange(
                scheduling_matrix_multistep, "m b t -> (m b) t"
            )  # shape = (num_denoising_steps*B, T=target_length + max_tokens)

            scheduling_matrices_multistep.append(scheduling_matrix_multistep)

        # reverse the list so that we can scheduling_matrix_multistep that start from the highest noise level at the front of the list
        scheduling_matrices_multistep = scheduling_matrices_multistep[::-1]

        num_outer_loops = len(scheduling_matrices_multistep)
        pbar = tqdm(
            total=num_outer_loops,
            initial=0,
            desc="StochSync",
            leave=False,
        )

        record_x0s_pred = []
        xs_pred_init = xs_pred.clone()

        for m, scheduling_matrix_multistep in enumerate(scheduling_matrices_multistep):

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()

            # (B, num_windows * self.max_tokens)
            if self.cfg.tasks.prediction.two_loops:
                if m % 3 == 0:
                    indices = indices_even_iter
                    num_windows = num_windows_even_iter
                elif m % 3 == 1:
                    indices = indices_odd_iter
                    num_windows = num_windows_odd_iter
                elif m % 3 == 2:
                    indices = indices_sync_loops
                    num_windows = num_windows_sync_loops
            else:
                indices = indices_odd_iter if m % 2 == 1 else indices_even_iter
                num_windows = (
                    num_windows_odd_iter if m % 2 == 1 else num_windows_even_iter
                )

            # modeled after get_diffusion_softmask from StochSync (https://github.com/KAIST-Visual-AI-Group/StochSync/blob/253e8e8349afab42d7c423e5946b7873c2d9e16b/stochsync/model/panorama.py#L222)
            # our version is slightly different as our prior is a video model (as opposed to an image model) that diffuses multiple frames
            # so our softmask is contains "num_windows" copies of a base softmask of shape (height, width * max_tokens),
            # where the radial falloff pattern is applied across the entire sliding window over "max_tokens" frames (as opposed to just across each frame)
            # this implementation assumes seam_removal_mode == "horizontal"
            def get_diffusion_softmask(num_windows, max_tokens, height, width, device):
                x = torch.linspace(0, 1, width * max_tokens // 2, device=device)
                x = torch.cat([x, torch.flip(x, dims=[0])])
                mask = x.repeat(height, 1)  # shape = (height, width * max_tokens)
                mask = rearrange(
                    mask, "h (t w) -> t h w", w=width, t=max_tokens
                )  # shape = (max_tokens, height, width, max_tokens)
                mask = repeat(
                    mask, "t h w -> num_windows t () h w", num_windows=num_windows
                )
                return mask  # shape = (num_windows, max_tokens, 1, height, width)

            # used for edge-preserving sampling near the end of the diffusion process
            if m >= num_outer_loops - seam_removal_steps:
                softmask = get_diffusion_softmask(
                    num_windows,
                    self.max_tokens,
                    self.x_shape[1],
                    self.x_shape[2],
                    self.device,
                )
                softmask = repeat(
                    softmask,
                    "num_windows max_tokens 1 h w -> (b num_windows) max_tokens 1 h w",
                    b=batch_size,
                )
            else:
                softmask = None

            # create context_mask_windows used to initialize the history guidance manager
            # do so by using indices to index into context_mask
            # indices.shape = (B, num_windows * self.max_tokens)
            # context_mask.shape = (B, target_length + max_tokens)
            # we want an output array "context_mask_windows" of shape (B, num_windows * self.max_tokens)
            context_mask_windows = context_mask[
                torch.arange(batch_size, device=self.device)[:, None], indices
            ]

            # indices.shape = (B, num_windows * self.max_tokens)
            # scheduling_matrix_multistep.shape = ((m * B), target_length + max_tokens)
            indices_repeated = repeat(
                indices,
                "b w_times_maxt -> (m b) w_times_maxt",
                m=scheduling_matrix_multistep.shape[0] // batch_size,
            ).contiguous()
            scheduling_matrix_multistep_windows = scheduling_matrix_multistep[
                torch.arange(scheduling_matrix_multistep.shape[0], device=self.device)[
                    :, None
                ],
                indices_repeated,
            ]
            scheduling_matrix_multistep_windows = rearrange(
                scheduling_matrix_multistep_windows,
                "(m b) w_times_maxt -> m b w_times_maxt",
                m=scheduling_matrix_multistep_windows.shape[0] // batch_size,
                b=batch_size,
            )

            # 5) coordinate the noise levels for all chunks
            #    shape = (num_multisteps, b, w_times_maxt)
            assert (
                scheduling_matrix_multistep_windows.shape[0] >= 2
            ), "at least 2 multisteps are required"
            from_noise_levels_windows_multisteps = scheduling_matrix_multistep_windows[
                0 : scheduling_matrix_multistep_windows.shape[0] - 1, ...
            ]
            to_noise_levels_windows_multisteps = scheduling_matrix_multistep_windows[
                1 : scheduling_matrix_multistep_windows.shape[0], ...
            ]

            # extract xs_pred_chunk_triplets from xs_pred
            # shape = (B, num_windows*self.max_tokens, self.x_shape)

            if self.cfg.tasks.prediction.warmup_steps is not None:
                if m < self.cfg.tasks.prediction.warmup_steps:
                    xs_pred_windows = xs_pred_init[
                        torch.arange(xs_pred_init.shape[0], device=self.device)[
                            :, None
                        ],
                        indices,
                    ]
                else:
                    # interpret xs_pred_windows as the x0 predictions (as opposed to the noisy latents)
                    # and run forward process to get noisy latent from xs_pred_windows
                    xs_pred_windows_clean = xs_pred[
                        torch.arange(xs_pred.shape[0], device=self.device)[:, None],
                        indices,
                    ]
                    xs_pred_windows = self.diffusion_model.q_sample(
                        xs_pred_windows_clean,
                        k=torch.clamp(from_noise_levels_windows_multisteps[0], min=0),
                    )
            else:
                # interpret xs_pred_windows as the x0 predictions (as opposed to the noisy latents)
                # and run forward process to get noisy latent from xs_pred_windows
                xs_pred_windows_clean = xs_pred[
                    torch.arange(xs_pred.shape[0], device=self.device)[:, None], indices
                ]
                xs_pred_windows = self.diffusion_model.q_sample(
                    xs_pred_windows_clean,
                    k=torch.clamp(from_noise_levels_windows_multisteps[0], min=0),
                )

            # 6) initialize the history guidance manager
            # inputs should have following shape
            # xs_pred_windows.shape = (B, num_windows*self.max_tokens, self.x_shape)
            # from_noise_levels_windows.shape = (B, num_windows*self.max_tokens)
            # to_noise_levels_windows.shape = (B, num_windows*self.max_tokens)
            # context_mask_windows.shape = (B, num_windows*self.max_tokens)

            conditions_mask_windows = None
            with history_guidance(context_mask_windows) as history_guidance_manager:

                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)

                # iterate over multisteps
                for i, (
                    from_noise_levels_windows_raw,
                    to_noise_levels_windows_raw,
                ) in enumerate(
                    zip(
                        from_noise_levels_windows_multisteps,
                        to_noise_levels_windows_multisteps,
                    )
                ):

                    (
                        xs_pred_windows,
                        from_noise_levels_windows,
                        to_noise_levels_windows,
                        conditions_mask_windows,
                    ) = history_guidance_manager.prepare(
                        xs_pred_windows,
                        from_noise_levels_windows_raw,
                        to_noise_levels_windows_raw,
                        replacement_fn=self.diffusion_model.q_sample,
                        replacement_only=True,  # for SimpleHistoryGuidanceManager, this flag isn't used
                    )

                    # 7) denoise all chunks in parallel by invoking self.diffusion_model.sample_step
                    # update xs_pred by DDIM or DDPM sampling
                    #
                    # xs_pred_chunk_triplets.shape = (B*nfe, num_windows*self.max_tokens, self.x_shape)
                    # from_noise_levels_chunk_triplets.shape = (B*nfe, num_windows*self.max_tokens)
                    # to_noise_levels_chunk_triplets.shape = (B*nfe, num_windows*self.max_tokens)
                    # conditions_chunk_triplets.shape = (B, num_windows*self.max_tokens, self.conditions_shape) -> (B*nfe, num_windows*self.max_tokens, self.conditions_shape)
                    # conditions_mask_chunk_triplets.shape = (B*nfe)
                    conditions_windows = (
                        conditions[
                            torch.arange(conditions.shape[0], device=self.device)[
                                :, None
                            ],
                            indices,
                        ]
                        if conditions is not None
                        else None
                    )
                    conditions_windows = repeat(
                        conditions_windows, "b ... -> (b nfe) ...", nfe=nfe
                    )

                    # reshape chunk_triplets from (B, windows*self.max_tokens, ...) to (B * windows), self.max_tokens, ...)
                    xs_pred_windows = rearrange(
                        xs_pred_windows,
                        "b (windows max_tokens) ... -> (b windows) max_tokens ...",
                        windows=num_windows,
                        max_tokens=self.max_tokens,
                    )
                    conditions_windows = rearrange(
                        conditions_windows,
                        "b (windows max_tokens) ... -> (b windows) max_tokens ...",
                        windows=num_windows,
                        max_tokens=self.max_tokens,
                    )
                    # noise_windows = rearrange(noise_windows, "b (windows max_tokens) ... -> (b windows) max_tokens ...", windows=num_windows, max_tokens=self.max_tokens)
                    from_noise_levels_windows = rearrange(
                        from_noise_levels_windows,
                        "b (windows max_tokens) ... -> (b windows) max_tokens ...",
                        windows=num_windows,
                        max_tokens=self.max_tokens,
                    )
                    to_noise_levels_windows = rearrange(
                        to_noise_levels_windows,
                        "b (windows max_tokens) ... ->(b windows) max_tokens ...",
                        windows=num_windows,
                        max_tokens=self.max_tokens,
                    )

                    reconstruction_guidance = self.cfg.diffusion.reconstruction_guidance
                    assert (
                        reconstruction_guidance == 0
                    ), "reconstruction guidance is not supported for stochsync"

                    # xs_pred_windows.shape = (B * num_windows, self.max_tokens, self.x_shape)
                    # x0s_pred_windows.shape = (B * num_windows, self.max_tokens, self.x_shape)
                    xs_pred_windows, x0s_pred_windows = (
                        self.diffusion_model.sample_step_windows_history_guidance(
                            xs_pred_windows,
                            from_noise_levels_windows,
                            to_noise_levels_windows,
                            self._process_conditions(
                                conditions_windows,
                                from_noise_levels_windows,
                            ),
                            (
                                repeat(
                                    conditions_mask_windows,
                                    "b -> (b windows)",
                                    windows=num_windows,
                                )
                                if conditions_mask_windows is not None
                                else None
                            ),
                            num_windows=num_windows,
                            update_where_noise_level_same=self.cfg.scheduling_matrix.update_where_noise_level_same,
                            noise=None,  # in Stochsync we don't need to worry about noise_windows as we don't have overlapping windows
                            minibatch_size=self.cfg.diffusion.minibatch_size,
                            history_guidance_manager=history_guidance_manager,
                            max_stoch=self.cfg.diffusion.max_stoch,
                        )
                    )

                    # for single-step denoising, we just take the x0 predictions instead of the noisy latents,
                    # and we don't apply edge-preserving sampling near the end of the diffusion process
                    # and then we break out of the inner multistep loop
                    if not self.cfg.tasks.prediction.multistep_denoising:
                        xs_pred_windows = x0s_pred_windows.clone()
                        xs_pred_windows = rearrange(
                            xs_pred_windows,
                            "(b windows) max_tokens ... -> b (windows max_tokens) ...",
                            b=batch_size,
                            windows=num_windows,
                        )
                        break

                    # edge-preserving sampling near the end of the diffusion process
                    # softmask.shape = (num_windows, max_tokens, 1, height, width)
                    if softmask is not None:
                        xs_pred_windows_clean_reshaped = rearrange(
                            xs_pred_windows_clean,
                            "b (windows max_tokens) ... -> (b windows) max_tokens ...",
                            windows=num_windows,
                            max_tokens=self.max_tokens,
                        )
                        N = len(scheduling_matrix_multistep)
                        M = (softmask >= (1 - i / N)).to(xs_pred_windows.dtype)

                        # xs_pred_windows.shape = (B * num_windows, self.max_tokens, self.x_shape)
                        # xs_pred_windows_clean.shape = (B, num_windows*self.max_tokens, self.x_shape)
                        # to_noise_levels_window_raw.shape = (B * num_windows, self.max_tokens)
                        to_noise_levels_windows_raw_reshaped = rearrange(
                            to_noise_levels_windows_raw,
                            "b (windows max_tokens) ... -> (b windows) max_tokens ...",
                            windows=num_windows,
                            max_tokens=self.max_tokens,
                        )

                        xs_pred_windows = (
                            xs_pred_windows * M
                            + self.diffusion_model.q_sample(
                                xs_pred_windows_clean_reshaped,
                                k=torch.clamp(
                                    to_noise_levels_windows_raw_reshaped, min=0
                                ),
                            )
                            * (1 - M)
                        )

                    xs_pred_windows = rearrange(
                        xs_pred_windows,
                        "(b windows) max_tokens ... -> b (windows max_tokens) ...",
                        b=batch_size,
                        windows=num_windows,
                    )

                xs_pred_windows = rearrange(
                    xs_pred_windows,
                    "b (windows max_tokens) ... -> b windows max_tokens ...",
                    windows=num_windows,
                    max_tokens=self.max_tokens,
                )
                x0s_pred_windows = rearrange(
                    x0s_pred_windows,
                    "(b windows) max_tokens ... -> b windows max_tokens ...",
                    b=batch_size,
                    windows=num_windows,
                )

                # if m is even, then we want to append one more windows to the end of xs_pred_windows and x0s_pred_windows
                # just to match the number of windows for when m is odd
                # if loop closing is enabled, the window size already match between odd m and even m
                if not self.cfg.tasks.prediction.loop_closing:
                    if m % 2 == 0:
                        xs_pred_windows = torch.cat(
                            [
                                xs_pred_windows,
                                torch.zeros_like(xs_pred_windows[:, 0:1]).clone(),
                            ],
                            dim=1,
                        )
                        x0s_pred_windows = torch.cat(
                            [
                                x0s_pred_windows,
                                torch.zeros_like(x0s_pred_windows[:, 0:1]).clone(),
                            ],
                            dim=1,
                        )

                # record_xs_pred.append(rearrange(xs_pred_windows, "b windows max_tokens ... -> b (windows max_tokens) ...").clone().detach())
                record_x0s_pred.append(
                    rearrange(
                        x0s_pred_windows,
                        "b windows max_tokens ... -> b (windows max_tokens) ...",
                    )
                    .clone()
                    .detach()
                )

                if not self.cfg.tasks.prediction.loop_closing:
                    if m % 2 == 0:
                        xs_pred_windows = rearrange(
                            xs_pred_windows[:, :-1],
                            "b windows max_tokens ... -> b (windows max_tokens) ...",
                        )
                        x0s_pred_windows = rearrange(
                            x0s_pred_windows[:, :-1],
                            "b windows max_tokens ... -> b (windows max_tokens) ...",
                        )
                    else:
                        xs_pred_windows = rearrange(
                            xs_pred_windows,
                            "b windows max_tokens ... -> b (windows max_tokens) ...",
                        )
                        x0s_pred_windows = rearrange(
                            x0s_pred_windows,
                            "b windows max_tokens ... -> b (windows max_tokens) ...",
                        )
                else:
                    xs_pred_windows = rearrange(
                        xs_pred_windows,
                        "b windows max_tokens ... -> b (windows max_tokens) ...",
                    )
                    x0s_pred_windows = rearrange(
                        x0s_pred_windows,
                        "b windows max_tokens ... -> b (windows max_tokens) ...",
                    )

                # indices.shape = (B, num_windows*self.max_tokens)
                # indices_to_denoise = indices
                indices_to_denoise = indices.clone()

                def index_reduce_mean(placeholder, indices, source):
                    dim = 0
                    return torch.index_reduce(
                        placeholder,
                        dim,
                        indices,
                        source,
                        reduce="mean",
                        include_self=False,
                    )

                # 7) update xs_pred by extracting the denoised chunks from the output of the diffusion model
                # we essentially want to do the following:
                # 1) create a placeholder tensor for xs_pred of shape (B, T, self.x_shape)
                # 2) fill in the denoised chunks in the placeholder tensor at the appropriate locations:
                #    xs_pred_placeholder[indices_to_denoise_repeated] += xs_pred_chunk_triplets[center_chunk_indices]
                #    by accumulating the denoised chunks into the placeholder tensor, if there are any chunks that are denoised multiple times, the denoised chunks will be summed up
                # 3) average chunks that are denoised multiple times by dividing the placeholder tensor by the number of times each chunk is denoised
                #
                # xs_pred_windows.shape = (B, num_windows*self.max_tokens, self.x_shape)
                # xs_pred.shape = (B, T, self.x_shape)

                xs_pred_placeholder = torch.zeros_like(
                    xs_pred
                )  # shape = (B, T, self.x_shape)

                assert torch.allclose(
                    indices_to_denoise.sort().values,
                    indices_to_denoise.unique().sort().values,
                ), "windows must be non-overlapping"
                xs_pred = torch.vmap(index_reduce_mean)(
                    xs_pred_placeholder, indices_to_denoise, xs_pred_windows
                )  # shape = (B, T, self.x_shape)

            # only replace the tokens being generated (revert context tokens and padding tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )
            pbar.update(1)

        pbar.close()

        # remove padding
        xs_pred = xs_pred[
            :, self.max_tokens // 2 : -(self.max_tokens - self.max_tokens // 2)
        ]

        return xs_pred
