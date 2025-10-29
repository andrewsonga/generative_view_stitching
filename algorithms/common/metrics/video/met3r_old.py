from typing import Any
import torch
from torch import Tensor
from torchmetrics import Metric
from met3r import MEt3R as MET3RModel
from einops import rearrange
from utils.retrieval_utils import FOVDistance, BaseConditionDistance

class MET3RMetric(Metric):
    """
    Calculates MET3R score between gt and pred frames of videos.
    """
        
    def __init__(self, img_size: int = 256, **kwargs: Any):
            super().__init__(**kwargs)
            self.add_state("met3r_sum", torch.tensor(0.0).double(), dist_reduce_fx="sum")
            self.add_state("num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
            
            self.metric = MET3RModel(
                img_size=img_size,
                use_norm=True,
                feat_backbone="dino16",
                featup_weights="mhamilton723/FeatUp",
                dust3r_weights="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                use_mast3r_dust3r=True,
            ).cuda()
    def compute(self) -> Tensor:
        """
        Compute final MET3R score.
        """
        return self.met3r_sum / self.num_samples 
    
    def update(self, preds: Tensor, target: Tensor, conditions: Tensor) -> None:
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions tensor of shape (B, T, C, H, W)
            target: Target tensor of shape (B, T, C, H, W)
            conditions: Conditions tensor of shape (B, T, C)
        """

        # Calculate the FOV of the conditions
        fov_distance = FOVDistance(frustum_length=2.0, fix_intrinsics=True)

        # last 10 frames as i, first 10 frames as j
        all_allowed_pairs = []
        for i in range(conditions.shape[1] - 10, conditions.shape[1]):
            for j in range(10):
                all_allowed_pairs.append((i, j))

        preds_frames = []
        for i,j in all_allowed_pairs:
            # assert all([torch.isclose(conditions[0], condition).all() for condition in conditions]), "all trajectories within a batch should be the same"
            fov_distance_score = fov_distance(conditions[0, i].unsqueeze(0).unsqueeze(0), conditions[0, j].unsqueeze(0).unsqueeze(0))[0]
            if fov_distance_score.squeeze(0,1).item() < 0.4:
                print(f"adding pair {i} {j}")
                preds_frames.append(torch.stack([preds[:, i], preds[:, j]], dim=1))
        # Get first and last frames
        # preds_frames = torch.stack([preds[:, 0], preds[:, 15]], dim=1)  # (B, 2, C, H, W)
        preds_frames = torch.cat(preds_frames, dim=0)

        # Convert to [-1, 1] range as required by MET3R
        preds_frames = preds_frames * 2 - 1
      
        score_preds, overlap_mask_preds, score_map_preds, projections_preds = self.metric(
            images=preds_frames,
            return_overlap_mask=True,
            return_score_map=True,
            return_projections=True
        )

        score_preds[overlap_mask_preds.sum(dim=(1,2)) == 0] = 2

        # score_preds = score_preds if overlap_mask_preds.sum() > 0 else torch.ones_like(score_preds) * 2
        self.met3r_sum += score_preds.mean()
        self.num_samples += 1


class NormalizedMET3RMetric(MET3RMetric):
    """
    Calculates MET3R score between first and last frames of videos.
    """    

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions tensor of shape (B, T, C, H, W)
            target: Target tensor of shape (B, T, C, H, W)
        """
        # Get first and last frames
        preds_frames = torch.stack([preds[:, 0], preds[:, -1]], dim=1)  # (B, 2, C, H, W)
        target_frames = torch.stack([target[:, 0], target[:, -1]], dim=1)  # (B, 2, C, H, W)
        
        # Convert to [-1, 1] range as required by MET3R
        preds_frames = preds_frames * 2 - 1
        target_frames = target_frames * 2 - 1
        
      
        score_pred, overlap_mask_pred, score_map_pred, projections_pred = self.metric(
            images=preds_frames,
            return_overlap_mask=True,
            return_score_map=True,
            return_projections=True
        )
        score_target, overlap_mask_target, score_map_target, projections_target = self.metric(
            images=target_frames,
            return_overlap_mask=True,
            return_score_map=True,
            return_projections=True
        )
        score_pred = score_pred if overlap_mask_pred.sum() > 0 else torch.ones_like(score_pred) * 2
        score_target = score_target if overlap_mask_target.sum() > 0 else torch.ones_like(score_target) * 2
        # normalize scores
        self.met3r_sum += (score_pred - score_target).mean()
        self.num_samples += 1



def main():
    # Create sample video data
    batch_size = 2
    num_frames = 20
    channels = 3
    height = 256
    width = 256
    
    # Create random video frames in [0, 1] range
    preds = torch.rand(batch_size, num_frames, channels, height, width)
    target = torch.rand(batch_size, num_frames, channels, height, width)
    conditions = torch.rand(batch_size, num_frames, channels)
    
    # Test regular MET3R metric
    print("Testing MET3RMetric:")
    metric = MET3RMetric(img_size=256)
    metric.update(preds, target, conditions)
    score = metric.compute()
    print(f"MET3R score: {score.item():.4f}")
    
    # Test normalized MET3R metric
    print("\nTesting NormalizedMET3RMetric:")
    norm_metric = NormalizedMET3RMetric(img_size=256)
    norm_metric.update(preds, target)
    norm_score = norm_metric.compute()
    print(f"Normalized MET3R score: {norm_score.item():.4f}")

if __name__ == "__main__":
    main()


