from typing import Optional
import torch
from torch import Tensor
from torchmetrics import Metric
from .shared_registry import SharedVideoMetricModelRegistry
from utils.retrieval_utils import FOVDistance


class LoopConsistency(Metric):
    """
    A metric that measures the consistency between frames generated with the same external conditions (poses).
    For each pair of frames with the same external condition, we compute their similarity and average over all pairs.

    Args:
        registry: The shared video metric model registry.
        similarity_metric: The metric to use for comparing frames. Can be 'lpips', 'mse', or 'ssim'.
        normalize: Whether to normalize the input frames to [0, 1] range.
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        similarity_metric: str = "lpips",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.similarity_metric = similarity_metric
        self.normalize = normalize
        self.registry = registry

        # Initialize the similarity metric
        if similarity_metric == "lpips":
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            self.metric = LearnedPerceptualImagePatchSimilarity(
                registry=registry, normalize=normalize
            )
        elif similarity_metric == "ssim":
            from torchmetrics.image import StructuralSimilarityIndexMeasure

            self.metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        elif similarity_metric == "psnr":
            from torchmetrics.image import PeakSignalNoiseRatio

            self.metric = PeakSignalNoiseRatio(data_range=1.0)
        elif similarity_metric == "mse":
            from torchmetrics.regression import MeanSquaredError

            self.metric = MeanSquaredError()
        elif similarity_metric == "met3r_cosine":
            from .met3r import MET3R

            self.metric = MET3R(
                img_size=kwargs["img_size"], distance="cosine", normalize=normalize
            )
        elif similarity_metric == "met3r_lpips":
            from .met3r import MET3R

            self.metric = MET3R(
                img_size=kwargs["img_size"], distance="lpips", normalize=normalize
            )
        elif similarity_metric == "met3r_ssim":
            from .met3r import MET3R

            self.metric = MET3R(
                img_size=kwargs["img_size"], distance="ssim", normalize=normalize
            )
        elif similarity_metric == "met3r_mse":
            from .met3r import MET3R

            self.metric = MET3R(
                img_size=kwargs["img_size"], distance="mse", normalize=normalize
            )
        elif similarity_metric == "met3r_psnr":
            from .met3r import MET3R

            self.metric = MET3R(
                img_size=kwargs["img_size"], distance="psnr", normalize=normalize
            )
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        self.overlap_threshold = kwargs["overlap_threshold"]
        self.max_tokens = kwargs["max_tokens"]

        # input params provided as part of kwargs
        self.fov_distance = FOVDistance(
            frustum_length=kwargs["frustum_length"],
            num_samples=kwargs["n_samples"],
            fix_intrinsics=kwargs["fix_intrinsics"],
        )

        # Add state for accumulating results
        self.add_state(
            "similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_pairs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, conditions: Tensor) -> None:
        self._update(preds, conditions)

    def _update(
        self,
        preds: Tensor,
        conditions: Tensor,
    ) -> None:
        """
        Update metric states with new predictions and conditions.

        Args:
            preds: Predicted frames of shape (B, T, C, H, W)
            conditions: External conditions of shape (B, T, D)
        """
        B, T = preds.shape[:2]

        # compute fov overlap between all frame pairs
        fov_overlap = self.fov_distance(conditions, conditions)[0]  # shape = (B, T, T)

        # average fov overlap with its transpose to make overlap symmetric
        fov_overlap = torch.mean(
            torch.stack([fov_overlap, fov_overlap.transpose(-1, -2)], dim=0), dim=0
        )

        # assert that fov_overlap is symmetric
        assert torch.allclose(fov_overlap, fov_overlap.transpose(-1, -2))

        # compute metric for frame pairs (i, j) such that fov_overlap[i, j] > overlap_threshold
        # for now this assumes batch_size = 1
        for b in range(B):
            for i in range(T):
                for j in range(i + 1, T):
                    if fov_overlap[b, i, j] > self.overlap_threshold:
                        # we also want to only visualize frame pairs that are temporally far away i.e. not within the same sliding window
                        if abs(i - j) > self.max_tokens:
                            # Compute similarity between the frames
                            frame1 = preds[b, i].unsqueeze(0)  # Add batch dimension
                            frame2 = preds[b, j].unsqueeze(0)

                            # calling forward() of metric will
                            # 1) automatically calls ``update()`` and
                            # 2) also returns the metric value at the current step
                            similarity = self.metric(frame1, frame2)

                            # Update state
                            self.similarity_sum += similarity
                            self.total_pairs += 1

    def compute(self) -> Tensor:
        """Compute the final metric value."""
        if self.total_pairs == 0:
            return torch.tensor(0.0, device=self.similarity_sum.device)

        return self.similarity_sum / self.total_pairs

