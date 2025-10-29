from typing import Optional
from jaxtyping import Float
import torch
from torch import Tensor
from omegaconf import DictConfig
from einops import rearrange, einsum, repeat
from utils.geometry_utils import CameraPose, generate_points_in_sphere, is_inside_fov
from .dfot_video import DFoTVideo
import math

class DFoTVideoPose(DFoTVideo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for pose-conditioned video generation.
    """

    def __init__(self, cfg: DictConfig):
        self.camera_pose_conditioning = cfg.camera_pose_conditioning
        self.conditioning_type = cfg.camera_pose_conditioning.type
        self._check_cfg(cfg)
        self._update_backbone_cfg(cfg)
        super().__init__(cfg)

    def _check_cfg(self, cfg: DictConfig):
        """
        Check if the config is valid
        """
        if cfg.backbone.name not in {"dit3d_pose", "u_vit3d_pose"}:
            raise ValueError(
                f"DiffusionForcingVideo3D only supports backbone 'dit3d_pose' or 'u_vit3d_pose', got {cfg.backbone.name}"
            )

        if (
            cfg.backbone.name == "dit3d_pose"
            and self.conditioning_type == "global"
            and cfg.backbone.conditioning.modeling != "film"
        ):
            raise ValueError(
                f"When using global camera pose conditioning, `algorithm.backbone.conditioning.modeling` should be 'film', got {cfg.backbone.conditioning.modeling}"
            )

        if cfg.backbone.name == "u_vit3d_pose" and self.conditioning_type == "global":
            raise ValueError(
                "Global camera pose conditioning is not supported for U-ViT3DPose"
            )

    def _update_backbone_cfg(self, cfg: DictConfig):
        """
        Update backbone config with camera pose conditioning
        """
        conditioning_dim = None
        match self.conditioning_type:
            case "global":
                conditioning_dim = 12
            case "ray" | "plucker":
                conditioning_dim = 6
            case "ray_encoding":
                conditioning_dim = 180
            case _:
                raise ValueError(
                    f"Unknown camera pose conditioning type: {self.conditioning_type}"
                )
        cfg.backbone.conditioning.dim = conditioning_dim

    @torch.no_grad()
    @torch.autocast(
        device_type="cuda", enabled=False
    )  # force 32-bit precision for camera pose processing
    def _process_conditions(
        self, conditions: Tensor, noise_levels: Optional[Tensor] = None
    ) -> Tensor:
        """
        Process conditions (raw camera poses) to desired format for the model
        Args:
            conditions (Tensor): raw camera poses (B, T, 16)
        """
        camera_poses = CameraPose.from_vectors(conditions)
        if self.cfg.tasks.prediction.history_guidance.name == "temporal":
            # NOTE: when using temporal history guidance,
            # some frames are fully masked out and thus their camera poses are not needed
            # so we replace them with interpolated camera poses from the nearest non-masked frames
            # this is important b/c we normalize camera poses by the first frame
            camera_poses.replace_with_interpolation(
                mask=noise_levels == self.timesteps - 1
            )

        match self.camera_pose_conditioning.normalize_by:
            case "first":
                camera_poses.normalize_by_first()
            case "mean":
                camera_poses.normalize_by_mean()
            case _:
                raise ValueError(
                    f"Unknown camera pose normalization method: {self.camera_pose_conditioning.normalize_by}"
                )

        if self.camera_pose_conditioning.bound is not None:
            camera_poses.scale_within_bounds(self.camera_pose_conditioning.bound)

        match self.conditioning_type:
            case "global":
                return camera_poses.extrinsics(flatten=True)
            case "ray" | "ray_encoding" | "plucker":
                rays = camera_poses.rays(resolution=self.x_shape[1])
                if self.conditioning_type == "ray_encoding":
                    rays = rays.to_pos_encoding()[0]
                else:
                    rays = rays.to_tensor(
                        use_plucker=self.conditioning_type == "plucker"
                    )
                return rearrange(rays, "b t h w c -> b t c h w")

    def _compute_fov_overlap(
        self,
        conditions_chunk_to_denoise: Float[torch.Tensor, "batch chunks_to_denoise chunk_size 16"],
        conditions_chunk_candidate_neighbor: Float[torch.Tensor, "batch chunks_candidate_neighbor chunk_size 16"],
        n_samples: int,
        sampling_radius: float,
        far_plane: float,
    ) -> Float[torch.Tensor, "batch chunks_to_denoise chunks_candidate_neighbor"]:
        """
        Compute the fraction of sampled points that is visible in the field-of-view (FOV) of both chunks, each of which are subsequences of chunk_size frames
        """
        batch_size = conditions_chunk_to_denoise.shape[0]
        chunk_size = conditions_chunk_to_denoise.shape[2]
        num_chunks_to_denoise = conditions_chunk_to_denoise.shape[1]
        num_chunks_candidate_neighbor = conditions_chunk_candidate_neighbor.shape[1]

        # before feeding it into CameraPose.from_vectors, reshape into (B, T, 16)

        conditions_chunk_to_denoise = rearrange(conditions_chunk_to_denoise, "b chunk_to_denoise chunk_size d -> (b chunk_to_denoise) chunk_size d")
        conditions_chunk_candidate_neighbor = rearrange(conditions_chunk_candidate_neighbor, "b chunk_candidate_neighbor chunk_size d -> (b chunk_candidate_neighbor) chunk_size d")

        cameras_chunk_to_denoise = CameraPose.from_vectors(conditions_chunk_to_denoise)
        cameras_chunk_candidate_neighbor = CameraPose.from_vectors(conditions_chunk_candidate_neighbor)

        cam2world_chunk_to_denoise = cameras_chunk_to_denoise.cam2world_4x4(flatten=False)
        cam2world_chunk_candidate_neighbor = cameras_chunk_candidate_neighbor.cam2world_4x4(flatten=False)

        # intrinsics assumed to be normalized (fx, fy, px, py)
        # left-top corner is (0, 0) and right-bottom corner is (1, 1)
        intrinsics_chunk_to_denoise = cameras_chunk_to_denoise.intrinsics(flatten=True)
        intrinsics_chunk_candidate_neighbor = cameras_chunk_candidate_neighbor.intrinsics(flatten=True)

        # reshape back to (b num_chunks chunk_size 4 4)
        cam2world_chunk_to_denoise = rearrange(cam2world_chunk_to_denoise, "(b num_chunks) chunk_size i j -> b num_chunks chunk_size i j", b=batch_size, num_chunks=num_chunks_to_denoise)
        cam2world_chunk_candidate_neighbor = rearrange(cam2world_chunk_candidate_neighbor, "(b num_chunks) chunk_size i j -> b num_chunks chunk_size i j", b=batch_size, num_chunks=num_chunks_candidate_neighbor)
        intrinsics_chunk_to_denoise = rearrange(intrinsics_chunk_to_denoise, "(b num_chunks) chunk_size i -> b num_chunks chunk_size i", b=batch_size, num_chunks=num_chunks_to_denoise)
        intrinsics_chunk_candidate_neighbor = rearrange(intrinsics_chunk_candidate_neighbor, "(b num_chunks) chunk_size i -> b num_chunks chunk_size i", b=batch_size, num_chunks=num_chunks_candidate_neighbor)
        
        assert self.camera_pose_conditioning.bound is None, "Scaling is not supported for FOV overlap computation"

        # 1) randomly sample in a sphere of radius sampling_radius or n_samples points
        points = generate_points_in_sphere(n_samples, sampling_radius).cuda()      # shape = (n_samples, 3)
        points = repeat(points, "n_samples xyz -> b chunks_to_denoise n_samples xyz", b=batch_size, chunks_to_denoise=num_chunks_to_denoise)
        points_x = points[:, :, :, 0]
        points_y = points[:, :, :, 1]
        points_z = points[:, :, :, 2]
        points_azimuth = torch.atan2(points_x, points_z) * (180 / math.pi)      # shape = (b, chunks_to_denoise, n_samples)
        points_elevation = torch.atan2(points_y, torch.sqrt(points_x**2 + points_z**2)) * (180 / math.pi)      # shape = (b, chunks_to_denoise, n_samples)
        fx, fy, px, py = intrinsics_chunk_to_denoise[..., chunk_size // 2, 0], intrinsics_chunk_to_denoise[..., chunk_size // 2, 1], intrinsics_chunk_to_denoise[..., chunk_size // 2, 2], intrinsics_chunk_to_denoise[..., chunk_size // 2, 3]

        # filter points that are within FOV of the camera
        fov_x_min = torch.atan2(-px, fx) * (180 / math.pi)      # shape = (b, chunks_to_denoise)
        fov_x_max = torch.atan2(1 - px, fx) * (180 / math.pi)      # shape = (b, chunks_to_denoise)
        fov_y_min = torch.atan2(-py, fy) * (180 / math.pi)      # shape = (b, chunks_to_denoise)   
        fov_y_max = torch.atan2(1 - py, fy) * (180 / math.pi)      # shape = (b, chunks_to_denoise)

        are_points_in_fov_chunk_to_denoise = (points_azimuth > fov_x_min[..., None]) & (points_azimuth < fov_x_max[..., None]) & (points_elevation > fov_y_min[..., None]) & (points_elevation < fov_y_max[..., None])          # shape = (b, chunks_to_denoise, n_samples)


        # 2) map points defined in camera coordinates of camera_chunk_to_denoise_middle_frame to world coordinates
        cam2world_chunk_to_denoise_middle_frame = cam2world_chunk_to_denoise[:, :, chunk_size // 2]       # shape = (b, chunks_to_denoise, 4, 4)
        points_world_homogeneous = einsum(cam2world_chunk_to_denoise_middle_frame, torch.cat([points, torch.ones_like(points[..., :1])], dim=-1), "b chunks_to_denoise i j, b chunks_to_denoise n_samples j -> b chunks_to_denoise n_samples i")
        points_world = points_world_homogeneous[..., :3]

        # 3) check if each point is in FOV of any frame of the chunk_to_denoise, resulting in a boolean array of shape (*, chunks_to_denoise, n_samples)
        are_points_in_fov = is_inside_fov(repeat(points_world, "b chunks_to_denoise n_samples xyz -> b chunks_to_denoise chunks_candidate_neighbor n_samples xyz", chunks_candidate_neighbor=num_chunks_candidate_neighbor), repeat(are_points_in_fov_chunk_to_denoise, "b chunks_to_denoise n_samples -> b chunks_to_denoise chunks_candidate_neighbor n_samples", chunks_candidate_neighbor=num_chunks_candidate_neighbor), repeat(cam2world_chunk_candidate_neighbor, "b chunks_candidate_neighbor chunk_size i j -> b chunks_to_denoise chunks_candidate_neighbor chunk_size i j", chunks_to_denoise=num_chunks_to_denoise), repeat(intrinsics_chunk_candidate_neighbor, "b chunks_candidate_neighbor i j -> b chunks_to_denoise chunks_candidate_neighbor i j", chunks_to_denoise=num_chunks_to_denoise), far_plane=far_plane)           # shape = (b, chunks_to_denoise, chunk_candidate_neighbor, n_samples)

        fov_overlap_symmetric = are_points_in_fov.sum(dim=-1) + rearrange(are_points_in_fov, "b c a s -> b a c s").sum(dim=-1)
        fov_overlap_symmetric_0to1 = fov_overlap_symmetric / (2 * repeat(are_points_in_fov_chunk_to_denoise.sum(dim=-1), "b c_d -> b c_d c_n", c_n=num_chunks_candidate_neighbor))

        # 6) return the fraction
        return fov_overlap_symmetric_0to1