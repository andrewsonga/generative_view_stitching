from typing import Optional
import torch
from torch import Tensor
from torchmetrics import Metric
from .shared_registry import SharedVideoMetricModelRegistry

class TemporalConsistency(Metric):
    """
    A metric that measures the consistency between neighboring frames.
    
    Args:
        registry: The shared video metric model registry.
        similarity_metric: The metric to use for comparing frames. Can be lpips, ssim, psnr, mse, met3r_cosine, met3r_lpips, met3r_ssim, met3r_mse, met3r_psnr.
        normalize: Whether to normalize the input frames to [0, 1] range.
    """
    
    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        similarity_metric: str = 'lpips',
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.similarity_metric = similarity_metric
        self.normalize = normalize
        self.registry = registry
        
        # Initialize the similarity metric
        if similarity_metric == 'lpips':
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.metric = LearnedPerceptualImagePatchSimilarity(
                registry=registry,
                normalize=normalize
            )
        elif similarity_metric == "ssim":
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            self.metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        elif similarity_metric == "psnr":
            from torchmetrics.image import PeakSignalNoiseRatio
            self.metric = PeakSignalNoiseRatio(data_range=1.0)
        elif similarity_metric == 'mse':
            from torchmetrics.regression import MeanSquaredError
            self.metric = MeanSquaredError()
        elif similarity_metric == "met3r_cosine":
            from .met3r import MET3R
            self.metric = MET3R(img_size=kwargs["img_size"], distance="cosine", normalize=normalize)
        elif similarity_metric == 'met3r_lpips':
            from .met3r import MET3R
            self.metric = MET3R(img_size=kwargs["img_size"], distance="lpips", normalize=normalize)
        elif similarity_metric == 'met3r_ssim':
            from .met3r import MET3R
            self.metric = MET3R(img_size=kwargs["img_size"], distance="ssim", normalize=normalize)
        elif similarity_metric == 'met3r_mse':
            from .met3r import MET3R
            self.metric = MET3R(img_size=kwargs["img_size"], distance="mse", normalize=normalize)
        elif similarity_metric == 'met3r_psnr':
            from .met3r import MET3R
            self.metric = MET3R(img_size=kwargs["img_size"], distance="psnr", normalize=normalize)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        # Add state for accumulating results
        self.add_state("similarity_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
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


        # compute metric for frame pairs (i, i+1)
        # for now this assumes batch_size = 1
        for b in range(B):
            for i in range(T - 1):
                j = i + 1
                frame1 = preds[b, i].unsqueeze(0)
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
