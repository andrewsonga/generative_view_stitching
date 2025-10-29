import os
import sys
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from typing import List, Optional, Union, Dict, Any
from einops import rearrange

from .shared_registry import SharedVideoMetricModelRegistry

# Add Video-Depth-Anything to path 
# Get the repository root (4 levels up from this file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
VIDEO_DEPTH_ANYTHING_PATH = os.path.join(REPO_ROOT, "third_party", "Video-Depth-Anything")
if VIDEO_DEPTH_ANYTHING_PATH not in sys.path:
    sys.path.insert(0, VIDEO_DEPTH_ANYTHING_PATH)

class DepthCollisionMetric(Metric):
    """
    DFOT-compatible depth collision metric that detects when generated video trajectories get too close to objects.

    This metric:
    1. Runs depth estimation on full video frames
    2. Applies a center crop mask for collision detection
    3. Detects collisions per video at multiple thresholds (default: 1.0)

    Based on Chen et al. 2025 - Video Depth Anything: Consistent Depth Estimation for Super-Long Videos
    """

    def __init__(
        self,
        registry: SharedVideoMetricModelRegistry,
        encoder: str = "vits",
        depth_threshold: float = 1.0,
        depth_thresholds: Optional[List[float]] = None,
        center_crop_ratio: float = 0.2,
        **kwargs,
    ):
        """
        Initialize DepthCollisionMetric for DFOT system.

        Args:
            registry: Shared video metric model registry (required for DFOT compatibility)
            encoder: Video depth model size ('vits', 'vitb', 'vitl')
            depth_threshold: Single threshold for collision detection (backward compatibility)
            depth_thresholds: List of thresholds to evaluate simultaneously
            center_crop_ratio: Ratio of center crop for ROI (0.2 = 20% of frame size)
        """
        super().__init__(**kwargs)
        self.registry = registry
        self.encoder = encoder
        self.center_crop_ratio = center_crop_ratio

        # Support both single threshold and multiple thresholds
        if depth_thresholds is not None:
            self.depth_thresholds = depth_thresholds
        else:
            self.depth_thresholds = [depth_threshold]

        self.depth_threshold = depth_threshold  # Keep for backward compatibility

        # Initialize depth estimation model
        self._init_depth_model()

        # Metric state variables
        self.add_state("total_videos", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "total_collision_videos", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state("total_frames", default=torch.tensor(0), dist_reduce_fx="sum")

        # Multi-threshold tracking
        for threshold in self.depth_thresholds:
            self.add_state(
                f"collision_videos_th_{threshold}",
                default=torch.tensor(0),
                dist_reduce_fx="sum",
            )
            # Store collision frames as a list (converted to string for state compatibility)
            self.add_state(
                f"collision_frames_info_th_{threshold}", default=[], dist_reduce_fx=None
            )

        if depth_thresholds is not None:
            print(
                f"DepthCollisionMetric initialized with {encoder} encoder, thresholds={depth_thresholds}, mode=metric depth (meters)"
            )
        else:
            print(
                f"DepthCollisionMetric initialized with {encoder} encoder, threshold={depth_threshold}, mode=metric depth (meters)"
            )

    def _init_depth_model(self):
        """Initialize Video Depth Anything model."""
        try:
            from video_depth_anything.video_depth import VideoDepthAnything

            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
                "vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                },
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                },
            }

            checkpoint_path = f"{VIDEO_DEPTH_ANYTHING_PATH}/checkpoints/metric_video_depth_anything_{self.encoder}.pth"

            # Initialize model with metric=True (only metric depth supported)
            self.depth_model = VideoDepthAnything(
                **model_configs[self.encoder], metric=True
            )

            # Load checkpoint - only metric depth supported
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Metric depth checkpoint not found at {checkpoint_path}. "
                    f"Only metric depth models are supported. Please ensure the metric checkpoint is available."
                )

            self.depth_model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu"), strict=True
            )
            print(f"Loaded metric depth model from: {checkpoint_path}")

            # Move model to device
            if torch.cuda.is_available():
                self.depth_model = self.depth_model.cuda()
            else:
                self.depth_model = self.depth_model.cpu()
            self.depth_model.eval()
            print(f"Depth model loaded successfully")

        except Exception as e:
            print(f"Error initializing depth model: {e}")
            import traceback

            traceback.print_exc()
            self.depth_model = None

    def _create_center_crop_mask(self, H: int, W: int) -> np.ndarray:
        """
        Create binary mask for center crop of the full frame.

        Args:
            H: Frame height
            W: Frame width

        Returns:
            Binary mask with center crop region set to 1.0
        """
        # Center crop dimensions
        crop_h = int(H * self.center_crop_ratio)
        crop_w = int(W * self.center_crop_ratio)

        # Calculate crop boundaries for center crop
        start_h = (H - crop_h) // 2  # Vertically centered
        start_w = (W - crop_w) // 2  # Horizontally centered

        # Create binary mask - only the center crop region is 1.0
        mask = np.zeros((H, W), dtype=np.float32)
        mask[start_h : start_h + crop_h, start_w : start_w + crop_w] = 1.0

        return mask

    def _compute_masked_average_depth(
        self, depth_map: np.ndarray, mask: np.ndarray
    ) -> float:
        """Compute the average depth within the masked region."""
        masked_depth = depth_map * mask
        return np.sum(masked_depth) / np.sum(mask)

    def _read_video_tensor(self, video_tensor: Tensor) -> List[np.ndarray]:
        """
        Convert video tensor to list of numpy frames.
        Args:
            video_tensor: Tensor of shape (T, C, H, W) or (B, T, C, H, W), range [0, 1]
        Returns:
            List of RGB frames as numpy arrays (H, W, 3), range [0, 255]
        """
        if video_tensor.dim() == 5:  # (B, T, C, H, W)
            video_tensor = video_tensor[0]  # Take first video in batch

        # Convert from (T, C, H, W) to list of (H, W, C)
        frames = []
        for t in range(video_tensor.shape[0]):
            frame = video_tensor[t].permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            frame = (frame * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            frames.append(frame)

        return frames

    def update(self, preds: Tensor, target: Optional[Tensor] = None) -> None:
        """
        Update the metric with predicted videos.

        Args:
            preds: Predicted videos of shape (B, T, C, H, W), range [0, 1]
            target: Target videos (not used for this metric, can be None)
        """
        batch_size = preds.shape[0]

        for b in range(batch_size):
            video = preds[b]  # (T, C, H, W)

            try:
                # Convert to numpy frames
                frames = self._read_video_tensor(video)
                original_height, original_width = frames[0].shape[:2]

                # Convert to numpy array for depth model
                if isinstance(frames, list):
                    frames = np.array(frames)

                # Run depth estimation on full frames (not cropped)
                depths, _ = self.depth_model.infer_video_depth(
                    frames,
                    target_fps=-1,
                    input_size=256,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    fp32=False,
                )

                # Create center crop mask for the full frame
                H, W = depths.shape[-2:]  # Full frame dimensions
                mask = self._create_center_crop_mask(H, W)

                # Compute collision detection for all thresholds
                avg_depths_center = []
                collision_detected_dict = {}
                collision_frame_dict = {}

                # Initialize collision tracking for all thresholds
                for threshold in self.depth_thresholds:
                    collision_detected_dict[threshold] = False
                    collision_frame_dict[threshold] = None

                for t, depth_map in enumerate(depths):
                    # Center crop average (for collision detection)
                    avg_depth_center = self._compute_masked_average_depth(
                        depth_map, mask
                    )
                    avg_depths_center.append(avg_depth_center)

                    # Check for collisions at all thresholds
                    for threshold in self.depth_thresholds:
                        if (
                            avg_depth_center < threshold
                            and not collision_detected_dict[threshold]
                        ):
                            collision_detected_dict[threshold] = True
                            collision_frame_dict[threshold] = (
                                t  # Store the collision frame index
                            )

                # Update metrics for primary threshold (backward compatibility)
                primary_threshold = self.depth_thresholds[0]
                collision_detected = collision_detected_dict[primary_threshold]

                # Log collision detection for video
                collision_detected_any = any(collision_detected_dict.values())
                if collision_detected_any:
                    collision_thresholds = [
                        str(th)
                        for th in self.depth_thresholds
                        if collision_detected_dict[th]
                    ]
                    print(f"Video {b}: Collision detected at thresholds: {', '.join(collision_thresholds)}")

                # Update state variables
                self.total_videos += 1
                self.total_frames += len(frames)

                if collision_detected:
                    self.total_collision_videos += 1

                # Update multi-threshold metrics
                for threshold in self.depth_thresholds:
                    if collision_detected_dict[threshold]:
                        # Can't use getattr with augmented assignment, need to use setattr
                        collision_videos_attr = getattr(
                            self, f"collision_videos_th_{threshold}"
                        )
                        setattr(
                            self,
                            f"collision_videos_th_{threshold}",
                            collision_videos_attr + 1,
                        )

                        # Store collision frame information
                        collision_frame = collision_frame_dict[threshold]
                        collision_info = f"video_{b}_frame_{collision_frame}"
                        collision_frames_info = getattr(
                            self, f"collision_frames_info_th_{threshold}"
                        )
                        collision_frames_info.append(collision_info)

            except Exception as e:
                print(f"Error processing video {b}: {e}")
                # Still count the video to maintain consistency
                self.total_videos += 1
                self.total_frames += video.shape[0]  # T

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric value.

        Returns:
            Primary metric value (video collision rate) as a single tensor
        """
        if self.total_videos == 0:
            return torch.tensor(0.0)

        # Return the primary metric: video collision rate
        return self.total_collision_videos.float() / self.total_videos

    def compute_detailed(self) -> Dict[str, torch.Tensor]:
        """
        Compute detailed metric values for analysis.

        Returns:
            Dictionary with all metric results
        """
        if self.total_videos == 0:
            return {
                "depth_collision_video_rate": torch.tensor(0.0),
            }

        results = {
            "depth_collision_video_rate": self.total_collision_videos.float()
            / self.total_videos,
        }

        # Add multi-threshold results
        for threshold in self.depth_thresholds:
            collision_videos_th = getattr(self, f"collision_videos_th_{threshold}")
            collision_frames_info_th = getattr(
                self, f"collision_frames_info_th_{threshold}"
            )

            results[f"depth_collision_video_rate_th_{threshold}"] = (
                collision_videos_th.float() / self.total_videos
            )
            results[f"collision_frames_info_th_{threshold}"] = collision_frames_info_th

        return results
