from typing import Any
import torch
from torch import Tensor
from jaxtyping import Float
from torchmetrics import Metric
from met3r import MEt3R as MET3RModel
from einops import rearrange
from utils.retrieval_utils import FOVDistance


def valid_img(img: Tensor, normalize: bool) -> bool:
    """check that input is a valid image to the network."""
    value_check = (
        img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    )
    return img.ndim == 4 and img.shape[1] == 3 and value_check


class MET3R(Metric):
    """
    Calculates MET3R score between gt and pred frames of videos.
    """

    def __init__(
        self,
        img_size: int = 256,
        distance: str = "cosine",  # Default to feature similarity, select from ["cosine", "lpips", "rmse", "psnr", "mse", "ssim"]
        backbone: str = "mast3r",
        normalize: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.add_state("met3r_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        # placeholder value for when the overlap mask is empty
        if distance == "cosine":
            #self.fallback_value = 2
            self.fallback_value = 1
        elif distance == "lpips":
            self.fallback_value = 1.0
        elif distance == "mse" or distance == "psnr":
            self.fallback_value = 100
        elif distance == "ssim":
            self.fallback_value = 0.0
        else:
            raise ValueError(f"Invalid distance: {distance}")

        self.metric = MET3RModel(
            img_size=img_size,  # Default to 256
            use_norm=True,  # Default to True
            backbone=backbone,  # Default to MASt3R, select from ["mast3r", "dust3r"]
            feature_backbone="dino16",  # Default to DINO, select from ["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"]
            feature_backbone_weights="mhamilton723/FeatUp",  # Default
            upsampler="featup",  # Default to FeatUP upsampling, select from ["featup", "nearest", "bilinear", "bicubic"]
            distance=distance,  # Default to feature similarity, select from ["cosine", "lpips", "psnr", "mse", "ssim"]
            freeze=True,  # Default to True
        ).cuda()

        if not isinstance(normalize, bool):
            raise ValueError(
                f"Argument `normalize` should be an bool but got {normalize}"
            )
        self.normalize = normalize

    def update(
        self, preds: Float[Tensor, "B C H W"], target: Float[Tensor, "B C H W"]
    ) -> None:

        if not (valid_img(preds, self.normalize) and valid_img(target, self.normalize)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
                f" Got input with shape {preds.shape} and {target.shape} and values in range"
                f" {[preds.min(), preds.max()]} and {[target.min(), target.max()]} when all values are"
                f" expected to be in the {[0,1] if self.normalize else [-1,1]} range."
            )

        B = preds.shape[0]
        preds_target = torch.stack([preds, target], dim=1)  # shape = [B, 2, C, H, W]
        # preds_twice = torch.stack([preds, preds], dim=1)      # shape = [B, 2, C, H, W]

        if self.normalize:
            # Convert to [-1, 1] range as required by MET3R
            preds_target = preds_target * 2 - 1
            preds_target = preds_target.clip(-1, 1)

            # preds_twice = preds_twice * 2 - 1
            # preds_twice = preds_twice.clip(-1, 1)

        # score_preds.shape = [B]
        # overlap_mask_preds.shape = [B, H, W]
        # score_map_preds.shape = [B, H, W]
        # projections_preds.shape = [B, 2, C_out, H, W]
        score_preds, overlap_mask_preds, score_map_preds, projections_preds = (
            self.metric(
                images=preds_target,
                return_overlap_mask=True,
                return_score_map=True,
                return_projections=True,
            )
        )

        """
        score_preds_twice, overlap_mask_preds_twice, score_map_preds_twice, projections_preds_twice = self.metric(
            images=preds_twice,
            return_overlap_mask=True,
            return_score_map=True,
            return_projections=True
        )
        """

        # compute the mean score for each frame
        score_preds[overlap_mask_preds.sum(dim=(1, 2)) == 0] = self.fallback_value

        self.met3r_sum += score_preds.sum().float()  # sum over batches
        self.num_samples += B

    def compute(self) -> Tensor:
        """
        Compute final MET3R score.
        """
        if self.num_samples == 0:
            return self.fallback_value
        return self.met3r_sum / self.num_samples
