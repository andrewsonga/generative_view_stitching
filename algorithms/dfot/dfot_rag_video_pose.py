from typing import Optional, Callable, Tuple, Any, Dict, Literal
from functools import partial
import torch
from tqdm import tqdm
from torch import Tensor
from omegaconf import DictConfig
import hydra
from einops import rearrange, repeat, reduce
from utils.retrieval_utils import impute_support_frames
from .dfot_video_pose import DFoTVideoPose
from .history_guidance import HistoryGuidance


class DFoTRAGVideoPose(DFoTVideoPose):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) with RAG for pose-conditioned video generation.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg_rag = cfg.rag
        self.impute_indices = cfg.rag.impute_indices
        self.retrieval_window_end_frame_rel = cfg.rag.retrieval_window_end_frame_rel
        self.n_support_frames = len(self.impute_indices)

        super().__init__(cfg)

    @property
    def n_support_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_support_frames)

    def _predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict a sequence given context tokens at the beginning, using sliding window if necessary.
        Args
        ----
        context: torch.Tensor, Shape (batch_size, init_context_len, *self.x_shape)
            Initial context tokens to condition on
        length: Optional[int]
            Desired number of tokens in sampled sequence.
            If None, fall back to to self.max_tokens, and
            If bigger than self.max_tokens, sliding window sampling will be used.
        conditions: Optional[torch.Tensor], Shape (batch_size, conditions_len, ...)
            Unprocessed external conditions for sampling, e.g. action or text, optional
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        reconstruction_guidance: float
            Scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        sliding_context_len: Optional[int]
            Max context length when using sliding window. -1 to use max_tokens - 1.
            Has no influence when length <= self.max_tokens as no sliding window is needed.
        return_all: bool
            Whether to return all steps of the sampling process.

        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Predicted sequence with both context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            Record of all steps of the sampling process
        retrieved_frames: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
        """
        if length is None:
            length = self.max_tokens
        if sliding_context_len is None:
            if self.max_tokens < length:
                raise ValueError(
                    "when length > max_tokens, sliding_context_len must be specified."
                )
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        batch_size, gt_len, *_ = context.shape

        if sliding_context_len < gt_len:
            raise ValueError(
                "sliding_context_len is expected to be >= length of initial context,"
                f"got {sliding_context_len}. If you are trying to use max context, "
                "consider specifying sliding_context_len=-1."
            )

        # chunk_size = self.chunk_size if self.use_causal_mask else self.max_tokens - self.n_support_tokens
        chunk_size = self.chunk_size if self.use_causal_mask else self.max_tokens

        # length of initial gt context
        curr_token = gt_len
        xs_pred = context
        retrieved_videos = torch.zeros_like(xs_pred)
        x_shape = self.x_shape
        record = None

        # POTENTIAL TODO:
        num_main_loop_iterations = 1 + (length - self.max_tokens) // (
            self.max_tokens - sliding_context_len - self.n_support_tokens
        )

        pbar = tqdm(
            total=self.sampling_timesteps * num_main_loop_iterations,
            initial=0,
            desc="Predicting with DFoT",
            leave=False,
        )

        main_loop_iter = 0

        # assumptions we make for following while loop to work correctly:
        # essentially, we want (length - (self.max_tokens - self.n_support_tokens)) to be divisible by h, which is the standard number of generated frames per loop iteration
        assert (length - self.max_tokens) % (
            min(
                self.max_tokens - self.n_support_tokens - sliding_context_len,
                chunk_size,
            )
        ) == 0, "length - (self.max_tokens - self.n_support_tokens) must be divisible by min(chunk_size, self.max_tokens - self.n_support_tokens - sliding_context_len) for the correct number of frames to be fed into self.sample_sequence"

        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")
            # actual context depends on whether it's during sliding window or not
            # corner case at the beginning
            # c=sliding_context_len (e.g. 4), except for the first window where c=gt_len (e.g. 2)
            c = min(sliding_context_len, curr_token)

            assert (
                conditions.shape[1] >= length
            ), "Not enough conditions to sample sequence of length {}".format(length)

            if curr_token <= sliding_context_len:
                # no retrieval
                n_support_tokens = 0
                impute_indices = []
            else:
                n_support_tokens = self.n_support_tokens
                impute_indices = self.impute_indices

            # try biggest prediction chunk size
            # 1) either in the final window. fill up to the target sequence length
            # 2) OR fill in the missing tokens AFTER the context and BEFORE the support frames
            h = min(length - curr_token, self.max_tokens - c - n_support_tokens)
            # chunk_size caps how many future tokens are diffused at once to save compute for causal model
            h = (
                min(h, chunk_size) if chunk_size > 0 else h
            )  # chunk_size is 8 -> h stays the same
            l = c + h + n_support_tokens
            pad = torch.zeros((batch_size, h, *x_shape)).to(context.device)
            # context is last c tokens out of the sequence of generated/gt tokens
            # pad to length that's required by _sample_sequence
            context = torch.cat([xs_pred[:, -c:], pad], 1)
            # calculate number of model generated tokens (not GT context tokens)
            generated_len = curr_token - max(curr_token - c, gt_len)
            # make context mask

            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask = torch.ones((batch_size, c), dtype=torch.long).to(
                context.device
            )
            if generated_len > 0:
                context_mask[:, -generated_len:] = 2
            pad_h = torch.zeros((batch_size, h), dtype=torch.long).to(context.device)
            context_mask = torch.cat([context_mask, pad_h.long()], 1)

            cond_len = l if self.use_causal_mask else self.max_tokens - n_support_tokens

            cond_slice = None
            if conditions is not None:
                # 1) when there are no generated tokens, cond_slice = conditions[:, 0:self.max_tokens - n_support_tokens]
                # 2) when there are generated tokens, cond_slice = conditions[:, curr_token - c : curr_token - c + (self.max_tokens - n_support_tokens)]
                cond_slice = conditions[:, curr_token - c : curr_token - c + cond_len]

            if n_support_tokens > 0:
                # Perform retrieval using:
                # queries: the conditions for the future token we're abt to diffuse
                # keys: the conditions for all previous tokens generated at least self.retrieval_window_end_frame_rel before curr_token
                # values: all previous tokens generated at least self.retrieval_window_end_frame_rel before curr_token
                query = cond_slice[:, c:cond_len]
                key = conditions[:, : curr_token - sliding_context_len + 1]
                value = xs_pred[:, : curr_token - sliding_context_len + 1]

                # perform retrieval: returns retrieved_cond (key) and retrieved_videos (value)
                # 1) compute the distance between the query set and every history frame (key)
                fov_distance = self.fov_distance(query, key)[
                    1
                ]  # shape = (B, key_length), (B, key_length)

                # 2) and then pick as many key frames whose visual overlap with the query set exceeds a threshold, where the max number of key frames = n_support frames
                # sort fov_distance in descending order
                fov_distance_sorted, fov_distance_sorted_indices = fov_distance.sort(
                    dim=-1, descending=True
                )  # shape = (B, key_length), (B, key_length)
                # fov_distance_sorted, fov_distance_sorted_indices = fov_distance.sort(dim=-1, descending=False)             # shape = (B, key_length), (B, key_length)

                xs_support = []
                conds_support = []
                context_mask_support = []
                for b in range(batch_size):
                    # 1 - self.cfg.fov_distance.overlap_threshold because fov_distance is defined as 1 - overlap
                    fov_distance_sorted_viable_indices_b = fov_distance_sorted_indices[
                        b,
                        fov_distance_sorted[b]
                        < 1.0 - self.cfg.fov_distance.overlap_threshold,
                    ]

                    num_viable_support_tokens = min(
                        n_support_tokens, len(fov_distance_sorted_viable_indices_b)
                    )
                    support_indices = fov_distance_sorted_viable_indices_b[
                        :num_viable_support_tokens
                    ]

                    xs_support_b = value[b, support_indices]
                    conds_support_b = key[b, support_indices]
                    context_mask_support_b = 2 * torch.ones(
                        (num_viable_support_tokens), dtype=torch.long
                    ).to(context.device)

                    # if there’s still space left in the sliding window, adding padding
                    if num_viable_support_tokens < n_support_tokens:
                        # extend xs_support_b and conds_support_b with padding to match n_support_tokens
                        xs_support_b = torch.cat(
                            [
                                xs_support_b,
                                torch.zeros(
                                    (
                                        n_support_tokens - num_viable_support_tokens,
                                        *x_shape,
                                    ),
                                    device=conditions.device,
                                ),
                            ]
                        )

                        # TODO: need to check if it's ok to pad conds with zeros <- IT'S NOT OK
                        # let's use the first frame of cond_slice to pad for support-frame conditions given that it's likely that we're gonna be normalizing the poses w.r.t. first frame of cond_slice anyways <- this is worse than just using the last n_support_tokens - num_viable_support_tokens frames of cond_slice
                        # so let's just use the last n_support_tokens - num_viable_support_tokens frames of cond_slice
                        assert (
                            cond_slice.shape[1]
                            >= n_support_tokens - num_viable_support_tokens
                        ), "Not enough conditions in cond_slice to pad for support-frame conditions"
                        conds_support_b = torch.cat(
                            [
                                conds_support_b,
                                cond_slice[
                                    b,
                                    cond_slice.shape[1]
                                    - (
                                        n_support_tokens - num_viable_support_tokens
                                    ) : cond_slice.shape[1],
                                ],
                            ]
                        )

                        # adding padding to context_mask_support_b
                        # NOTE: -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
                        context_mask_support_b = torch.cat(
                            [
                                context_mask_support_b,
                                -torch.ones(
                                    (n_support_tokens - num_viable_support_tokens),
                                    device=conditions.device,
                                ),
                            ]
                        )

                    xs_support.append(xs_support_b)
                    conds_support.append(conds_support_b)
                    context_mask_support.append(context_mask_support_b)

                xs_support = torch.stack(xs_support, dim=0)
                conds_support = torch.stack(conds_support, dim=0)
                context_mask_support = torch.stack(context_mask_support, dim=0)

                # 3) impute the selected key frames and conds into context and cond_slice
                # TODO: this assumes that the support frames are added to the end of the sliding window
                conds_and_contextmask = torch.cat(
                    [cond_slice, context_mask[:, :, None]], dim=-1
                )
                conds_and_contextmask_support = torch.cat(
                    [conds_support, context_mask_support[:, :, None]], dim=-1
                )

                context, conds_and_contextmask, _, _, _ = impute_support_frames(
                    video=context,
                    cond=conds_and_contextmask,
                    support_video=xs_support,
                    support_cond=conds_and_contextmask_support,
                    impute_indices=self.impute_indices,
                )
                cond_slice = conds_and_contextmask[..., :-1]
                context_mask = conds_and_contextmask[..., -1]

            new_pred, record = self._sample_sequence(
                batch_size,
                length=l,
                context=context,
                context_mask=context_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                history_guidance=history_guidance,
                return_all=return_all,
                pbar=pbar,
            )

            # update xs_pred with non-imputed, generated tokens
            impute_indices = sorted(
                [i if i >= 0 else new_pred.shape[1] + i for i in impute_indices]
            )
            valid_indices = [i for i in range(l) if i not in impute_indices]
            valid_frames = new_pred[:, valid_indices]
            xs_pred = torch.cat([xs_pred, valid_frames[:, -h:]], 1)

            # update curr_token
            curr_token = xs_pred.shape[1]

            main_loop_iter += 1

        assert (
            main_loop_iter == num_main_loop_iterations
        ), f"Expected {num_main_loop_iterations} main loop iterations, but got {main_loop_iter}."

        pbar.close()
        return xs_pred, record

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.max_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """
        x_shape = self.x_shape

        if length is None:
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
            raise ValueError(
                f"length is expected to <={self.max_tokens}, got {length}."
            )

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(
                    f"context shape not compatible with x_stacked_shape {x_shape}."
                )

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        if conditions is not None:
            if self.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
                )
            elif not self.use_causal_mask and conditions.shape[1] != self.max_tokens:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {conditions.shape[1]}."
                )

        horizon = length if self.use_causal_mask else self.max_tokens
        padding = horizon - length
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(
                (batch_size, horizon), dtype=torch.long, device=self.device
            )
        elif padding > 0:
            # pad context and context mask to reach horizon
            context_pad = torch.zeros(
                (batch_size, padding, *x_shape), device=self.device
            )
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)

        if history_guidance is None:
            # by default, use conditional sampling
            history_guidance = HistoryGuidance.conditional(
                timesteps=self.timesteps,
            )

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            horizon - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)

        # fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:
            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )
        # fill padding tokens' noise levels to be maximum in scheduling matrix
        # this covers the case where the input to _sample_sequence is already padded via external padding logic
        # e.g. padding logic in _predict_sequence of DFoTVideoPoseRAG
        scheduling_matrix = torch.where(
            context_mask[None] == -1, self.timesteps - 1, scheduling_matrix
        )

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT-RAG",
                leave=False,
            )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            with history_guidance(context_mask) as history_guidance_manager:
                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)
                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
                    history_guidance_manager.prepare(
                        xs_pred,
                        from_noise_levels,
                        to_noise_levels,
                        replacement_fn=self.diffusion_model.q_sample,
                        replacement_only=self.is_full_sequence,
                    )
                )
                if reconstruction_guidance > 0:

                    def base_reconstruction_guidance_fn(
                        target: Float[Tensor, "B T C H W"],
                        pred_x0: Float[Tensor, "B T C H W"],
                        alpha_cumprod: Float[Tensor, "B T 1 1 1"],
                        context_mask: Float[Tensor, "B T"],
                        reconstruction_guidance: float,
                        x_shape: Int[Tensor, "3"],
                    ) -> Float[Tensor, ""]:
                        target = target.clone().detach()
                        loss = (
                            F.mse_loss(pred_x0, target, reduction="none")
                            * alpha_cumprod.sqrt()
                        )
                        _context_mask = rearrange(
                            context_mask.bool(),
                            "b t -> b t" + " 1" * len(x_shape),
                        )
                        # scale inversely proportional to the number of context frames
                        loss = torch.sum(
                            loss
                            * _context_mask
                            / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                        )
                        likelihood = -reconstruction_guidance * 0.5 * loss
                        return likelihood

                    composed_guidance_fn = partial(
                        base_reconstruction_guidance_fn,
                        target=xs_pred,  # target = [x^a, z^b]
                        context_mask=context_mask,  # context_mask = [1] * len(x^a) + [0] * len(z^b)
                        reconstruction_guidance=reconstruction_guidance,
                        x_shape=x_shape,
                    )
                else:
                    composed_guidance_fn = guidance_fn

                # for the context frames, noise them to the current noise level using q_sample
                # basically, we are taking x = [x^a, z^b] -> [z^a_t, z^b_t]
                if self.is_full_sequence:
                    xs_pred = torch.where(
                        self._extend_x_dim(context_mask) >= 1,
                        self.diffusion_model.q_sample(xs_pred, from_noise_levels),
                        xs_pred,
                    )

                # update xs_pred by DDIM or DDPM sampling
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(
                        (
                            repeat(
                                conditions,
                                "b ... -> (b nfe) ...",
                                nfe=nfe,
                            ).clone()
                            if conditions is not None
                            else None
                        ),
                        from_noise_levels,
                    ),
                    conditions_mask,
                    composed_guidance_fn,
                )

                xs_pred = history_guidance_manager.compose(xs_pred)

            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record
