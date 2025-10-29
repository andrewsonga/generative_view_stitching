"""
3D Geometry Utils (Camera Pose, Ray, Plücker coordinates)
All functions in this file follows the following convention:
- We assume a "batch" of multiple scenes, i.e. B T ...
- Camera Pose is composed of Rotation R (Tensor, B T 3 3), Translation T (Tensor, B T 3) - world to camera - and the intrinsics (Tensor, B T 4) - camera to image.
- Intrinsic matrix assumes that the pixel coordinates are normalized, i.e. left-top corner is (0, 0) and right-bottom corner is (1, 1). The intrinsics matrix is represented as (fx, fy, px, py).
- Rays are represented as either:
    - Original rays (Tensor, B T H W 6) - origin (Tensor, B T H W 3) and unnormalized direction (Tensor, B T H W 3) concatenated.
    - Plücker coordinates (Tensor, B T H W 6), normalized direction (Tensor, B T H W 3) and moment (Tensor, B T H W 3) concatenated.
"""

from typing import Tuple, Optional
from jaxtyping import Float
import math
import torch
import roma
from einops import rearrange, einsum, repeat
from torch import Tensor
from typing import Sequence, Literal


class Ray:
    """
    A class to represent the batched rays.
    """

    def __init__(self, origin: torch.Tensor, direction: torch.Tensor):
        """
        Args:
            origin (torch.Tensor): The origin of the rays. Shape (B, T, H, W, 3).
            direction (torch.Tensor): The direction of the rays. Shape (B, T, H, W, 3).
        """
        self._origin = origin
        self._direction = direction

    def to_tensor(self, use_plucker: bool = False) -> torch.Tensor:
        """
        Returns the rays represented as a tensor.
        Args:
            use_plucker (bool): Whether to use Plücker coordinates or not.
        Returns:
            torch.Tensor: The rays tensor. Shape (B, T, H, W, 6).
        """
        if not use_plucker:
            return torch.cat([self._origin, self._direction], dim=-1)

        # Plücker coordinates
        direction = self._direction / self._direction.norm(dim=-1, keepdim=True)
        moment = torch.cross(self._origin, direction, dim=-1)
        return torch.cat([direction, moment], dim=-1)

    @staticmethod
    def _nerf_pos_encoding(x: torch.Tensor, freq: int) -> torch.Tensor:
        scale = (
            2 ** torch.linspace(0, freq - 1, freq, device=x.device, dtype=x.dtype)
            * math.pi
        )
        encoding = rearrange(x[..., None] * scale, "b t h w i s -> b t h w (i s)")
        return torch.sin(torch.cat([encoding, encoding + 0.5 * math.pi], dim=-1))

    def to_pos_encoding(
        self,
        freq_origin: int = 15,
        freq_direction: int = 15,
        return_rays: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns the rays represented as positional encoding. Follows NeRF to map the rays into a higher-dimensional space.
        Args:
            freq_origin (int): The frequency for the origin.
            freq_direction (int): The frequency for the direction.
            return_rays (bool): Whether to return the rays tensor or not.
        Returns:
            torch.Tensor: The rays tensor. Shape (B, T, H, W, 6 * (freq_origin + freq_direction)).
        """
        encoding = torch.cat(
            [
                self._nerf_pos_encoding(self._origin, freq_origin),
                self._nerf_pos_encoding(self._direction, freq_direction),
            ],
            dim=-1,
        )
        rays_tensor = self.to_tensor(use_plucker=False) if return_rays else None
        return encoding, rays_tensor


class CameraPose:
    """
    A class to represent the batched camera poses.
    """

    # pylint: disable=invalid-name

    def __init__(self, R: torch.Tensor, T: torch.Tensor, K: torch.Tensor):
        """
        Args:
            R (torch.Tensor): The rotation matrix. Shape (B, T, 3, 3).
            T (torch.Tensor): The translation vector. Shape (B, T, 3).
            K (torch.Tensor): The intrinsics vector. Shape (B, T, 4).
        """
        self._R = R
        self._T = T
        self._K = K

    @classmethod
    def from_vectors(cls, raw_camera_poses: torch.Tensor):
        """
        Creates a CameraPose object from the raw camera poses.
        Args:
            raw_camera_poses (torch.Tensor): The raw camera poses. Shape (B, T, 4 + 12). The first 4 elements are the intrinsics, the next 12 elements are the flattened extrinsics.
        Returns:
            CameraPose: The CameraPose object.
        """
        K, RT = raw_camera_poses.split([4, 12], dim=-1)
        RT = rearrange(RT, "b t (i j) -> b t i j", i=3, j=4)
        R = RT[..., :3, :3]
        T = RT[..., :3, 3]
        return cls(R, T, K)

    def _normalize_by(self, R_ref: torch.Tensor, T_ref: torch.Tensor) -> None:
        """
        Normalizes so that the camera given by R_ref and T_ref becomes the world coordinates.
        Args:
            R_ref (torch.Tensor): The rotation matrix. Shape (B, 3, 3).
            T_ref (torch.Tensor): The translation vector. Shape (B, 3).
        """
        R_inv = rearrange(R_ref, "b i j -> b j i")
        self._R = einsum(self._R, R_inv, "b t i j1, b j1 j2 -> b t i j2")
        self._T = self._T - einsum(self._R, T_ref, "b t i j, b j -> b t i")

    def normalize_by_first(self) -> None:
        """
        Normalizes the camera poses by the first camera, i.e. computes the relative poses w.r.t. the first camera.
        After normalization, the first camera will have identity rotation and zero translation, i.e. the first camera = world coordinates.
        """
        self._normalize_by(self._R[:, 0], self._T[:, 0])
    
    def normalize_by_middle(self) -> None:
        """
        Normalizes the camera poses by the first camera, i.e. computes the relative poses w.r.t. the first camera.
        After normalization, the first camera will have identity rotation and zero translation, i.e. the first camera = world coordinates.
        """
        seq_len = self._R.shape[1]
        self._normalize_by(self._R[:, seq_len // 2], self._T[:, seq_len // 2])

    def normalize_by_mean(self) -> None:
        """
        Normalizes the camera poses by the mean of all cameras, i.e. computes the relative poses w.r.t. the mean frame.
        The mean camera becomes the world coordinates.
        """
        # convert to quaternions, average them, and convert back to rotation matrices
        q = roma.rotmat_to_unitquat(self._R)
        q_mean = q.mean(dim=1)
        R_mean = roma.unitquat_to_rotmat(q_mean)
        # average translation on world coordinates
        # R_mean^T @ T_mean = mean(sum(R_i^T @ T_i))
        T_world_mean = einsum(
            rearrange(self._R, "b t i j -> b t j i"), self._T, "b t i j, b t j -> b t i"
        ).mean(dim=1)
        T_mean = einsum(R_mean, T_world_mean, "b i j, b j -> b i")
        self._normalize_by(R_mean, T_mean)

    def scale_within_bounds(self, bounds: float = 1.0) -> None:
        """
        Scales the camera locations, so that they are within the boundary box [-bounds, bounds]^3.
        Each scene is scaled independently, while each frame within a scene is scaled by the same factor.
        Args:
            bounds (float): The boundary box. Requires bounds > 0.
        """
        # simply scale the translation vectors by the same factor
        max_vals = (
            self._T.abs()
            .max(dim=1, keepdim=True)
            .values.max(dim=2, keepdim=True)
            .values
        )
        scale = bounds / max_vals.clamp(min=1e-6)
        self._T = self._T * scale

    def replace_with_interpolation(self, mask: torch.Tensor) -> None:
        """
        For each sequence in the batch,
        replaces the invalid camera poses (mask == True) by interpolating
        between the nearest valid camera poses (mask == False).
        Args:
            mask (torch.Tensor): The mask for the camera poses to replace. Shape (B, T).
        """
        q = roma.rotmat_to_unitquat(self._R)  # (B, T, 4)
        T = self._T.clone()

        for b in range(mask.shape[0]):
            curr_mask = mask[b]
            if not curr_mask.any() or curr_mask.all():
                continue
            valid_ts = torch.where(~curr_mask)[0]

            if valid_ts[0] != 0:
                q[b, : valid_ts[0]] = q[b, valid_ts[0]]
                T[b, : valid_ts[0]] = T[b, valid_ts[0]]
            if valid_ts[-1] != mask.shape[1] - 1:
                q[b, valid_ts[-1] + 1 :] = q[b, valid_ts[-1]]
                T[b, valid_ts[-1] + 1 :] = T[b, valid_ts[-1]]

            for left_t, right_t in zip(valid_ts[:-1], valid_ts[1:]):
                if right_t - left_t == 1:
                    continue
                left_q, right_q = q[b, [left_t, right_t]]
                q[b, left_t : right_t + 1] = roma.unitquat_slerp(
                    left_q,
                    right_q,
                    torch.linspace(0, 1, right_t - left_t + 1, device=q.device),
                )
                T[b, left_t : right_t + 1] = torch.lerp(
                    T[b, left_t],
                    T[b, right_t],
                    torch.linspace(
                        0, 1, right_t - left_t + 1, device=T.device
                    ).unsqueeze(-1),
                )

        self._R = roma.unitquat_to_rotmat(q)
        self._T = T

    def extrinsics(self, flatten: bool = False) -> torch.Tensor:
        """
        Returns the extrinsics matrix [R | T] for the camera poses.
        Args:
            flatten (bool): Whether to flatten the extrinsics matrix.
        Returns:
            torch.Tensor: The extrinsics matrix. Shape (B, T, 12) if flatten is True, else (B, T, 3, 4).
        """
        extrinsics = torch.cat(
            [self._R, rearrange(self._T, "b t i -> b t i 1")], dim=-1
        )
        return rearrange(extrinsics, "b t i j -> b t (i j)") if flatten else extrinsics

    def cam2world_4x4(self, flatten: bool = False) -> torch.Tensor:
        """
        Returns the 4x4 camera-to-world matrix for the camera poses.
        """

        world2cam_3x4 = self.extrinsics()  # shape = (B, T, 3, 4)
        world2cam_4x4 = torch.cat(
            [world2cam_3x4, torch.zeros_like(world2cam_3x4[..., 0:1, :])], dim=-2
        )
        world2cam_4x4[..., 3, 3] = 1.0
        cam2world_4x4 = world2cam_4x4.inverse()

        return (
            rearrange(cam2world_4x4, "b t (i j) -> b t i j", i=4, j=4)
            if flatten
            else cam2world_4x4
        )

    def intrinsics(self, flatten: bool = False) -> torch.Tensor:
        """
        Returns the intrinsics matrix for the camera poses.
        Args:
            flatten (bool): Whether to flatten the intrinsics matrix.
        Returns:
            torch.Tensor: The intrinsics matrix. Shape (B, T, 3, 3) if flatten is False, else (B, T, 4).
        """
        if flatten:
            return self._K
        else:
            K = repeat(
                torch.eye(3, device=self._K.device),
                "i j -> b t i j",
                b=self._K.shape[0],
                t=self._K.shape[1],
            ).clone()
            K[:, :, 0, 0] = self._K[:, :, 0]
            K[:, :, 1, 1] = self._K[:, :, 1]
            K[:, :, 0, 2] = self._K[:, :, 2]
            K[:, :, 1, 2] = self._K[:, :, 3]
            return K

    def rays(self, resolution: int) -> Ray:
        """
        Returns the rays for the camera poses.
        Args:
            resolution (int): The resolution of the image.
        Returns:
            Ray: The rays object.
        """

        # Direction
        # compute ray direction in camera coordinates
        coord_w, coord_h = torch.meshgrid(
            torch.linspace(
                0,
                resolution - 1,
                resolution,
                device=self._K.device,
                dtype=self._K.dtype,
            ),
            torch.linspace(
                0,
                resolution - 1,
                resolution,
                device=self._K.device,
                dtype=self._K.dtype,
            ),
            indexing="xy",
        )
        coord_w = rearrange(coord_w, "h w -> 1 1 h w") + 0.5  # (1, 1, H, W)
        coord_h = rearrange(coord_h, "h w -> 1 1 h w") + 0.5  # (1, 1, H, W)

        fx, fy, px, py = rearrange(self._K * resolution, "b t i -> b t i 1").chunk(
            4, dim=-2
        )
        x = (coord_w - px) / fx
        y = (coord_h - py) / fy
        z = torch.ones_like(x)
        direction = torch.stack([x, y, z], dim=-1)
        # convert to world coordinates
        R_inv = rearrange(self._R, "b t i j -> b t j i")
        direction = einsum(
            R_inv,
            direction,
            "b t i j, b t h w j -> b t h w i",
        )

        # Origin
        origin = -einsum(R_inv, self._T, "b t i j, b t j -> b t i")
        origin = repeat(
            origin, "b t i -> b t h w i", h=resolution, w=resolution
        ).clone()
        return Ray(origin, direction)


def generate_points_in_sphere(n_points, radius):
    # Sample three independent uniform distributions
    samples_r = torch.rand(n_points)  # For radius distribution
    samples_phi = torch.rand(n_points)  # For azimuthal angle phi
    samples_u = torch.rand(n_points)  # For polar angle theta

    # Apply cube root to ensure uniform volumetric distribution
    r = radius * torch.pow(samples_r, 1 / 3)
    # Azimuthal angle phi uniformly distributed in [0, 2π]
    phi = 2 * math.pi * samples_phi
    # Convert u to theta to ensure cos(theta) is uniformly distributed
    theta = torch.acos(1 - 2 * samples_u)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def is_inside_fov(
    points_world_frame_chunk_to_denoise: Float[
        torch.Tensor, "b chunks_to_denoise chunks_candidate_neighbor num_samples 3"
    ],
    are_points_in_fov_chunk_candidate_neighbor: Float[
        torch.Tensor, "b chunks_to_denoise chunks_candidate_neighbor num_samples"
    ],
    cam2world_chunk_candidate_neighbor: Float[
        torch.Tensor, "b chunks_to_denoise chunks_candidate_neighbor chunk_size 4 4"
    ],
    intrinsics_chunk_candidate_neighbor: Float[
        torch.Tensor, "b chunks_to_denoise chunks_candidate_neighbor chunk_size 4"
    ],
    far_plane: float = 4.0,
) -> Float[torch.Tensor, "b chunks_to_denoise chunks_candidate_neighbor num_samples"]:

    world2cam = cam2world_chunk_candidate_neighbor.inverse()

    # if points_world_frame is visible in at least one frame of the chunk, return 1, else 0

    points_world_frame_homogeneous = torch.cat(
        [
            points_world_frame_chunk_to_denoise,
            torch.ones_like(points_world_frame_chunk_to_denoise[..., :1]),
        ],
        dim=-1,
    )

    points_camera_frame_homogeneous = einsum(
        world2cam,
        points_world_frame_homogeneous,
        "b c_d c_n t i j, b c_d c_n s j -> b c_d c_n t s i",
    )

    # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size, num_samples, 3)
    points_camera_frame = points_camera_frame_homogeneous[..., :3]
    x = points_camera_frame[..., 0]
    y = points_camera_frame[..., 1]
    z = points_camera_frame[..., 2]

    # Compute horizontal angle (yaw): measured with respect to the z-axis as the forward direction,
    # and the x-axis as left-right, resulting in a range of -180 to 180 degrees.
    azimuth = torch.atan2(x, z) * (
        180 / math.pi
    )  # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size, num_samples)

    # Compute vertical angle (pitch): measured with respect to the horizontal plane,
    # resulting in a range of -90 to 90 degrees.
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (
        180 / math.pi
    )  # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size, num_samples)

    # Check if both horizontal and vertical angles are within their respective FOV limits and whether the point is in front of the camera AND not beyond the far plane
    fx, fy, px, py = (
        intrinsics_chunk_candidate_neighbor[..., 0],
        intrinsics_chunk_candidate_neighbor[..., 1],
        intrinsics_chunk_candidate_neighbor[..., 2],
        intrinsics_chunk_candidate_neighbor[..., 3],
    )

    fov_x_min = torch.atan2(-px, fx) * (
        180 / math.pi
    )  # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size)
    fov_x_max = torch.atan2(1 - px, fx) * (
        180 / math.pi
    )  # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size)
    fov_y_min = torch.atan2(-py, fy) * (
        180 / math.pi
    )  # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size)
    fov_y_max = torch.atan2(1 - py, fy) * (
        180 / math.pi
    )  # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size)

    point_in_fov_frame = (
        (azimuth > fov_x_min[..., None])
        & (azimuth < fov_x_max[..., None])
        & (elevation > fov_y_min[..., None])
        & (elevation < fov_y_max[..., None])
        & (z >= 0)
        & (z < far_plane)
    )
    # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, chunk_size, num_samples)

    point_in_fov_chunk = point_in_fov_frame.any(
        dim=-2
    )  # shape = (b, chunks_to_denoise, chunks_candidate_neighbor, num_samples)

    point_in_fov_chunk = are_points_in_fov_chunk_candidate_neighbor * point_in_fov_chunk

    return point_in_fov_chunk


def get_frustum(
    K: torch.Tensor,
    uv_range: Sequence[float | int | Sequence[float | int]] = [0, 1],
    frustum_scale: float = 1.0,
    center_ray_mult: float = 10.0,
):
    """
    Get the frustum of the camera.
    Args:
        K: [..., 4] or [..., 3, 3]
        uv_range: [H, W] or [[H_min, H_max], [W_min, W_max]]
        frustum_scale: float
        center_ray_mult: float
    Returns:
        frustum: [..., 9, 2, 3]
            - dim[-1] is (x,y,z)
            - dim[-2] is (start, end)
            - dim[-3] is the edge index. Last edge is the center ray.
        frustum_width: [...]
        frustum_height: [...]
        frustum_center: [..., 3]
    """
    if K.shape[-1] == 4:
        pass
    else:
        assert K.shape[-2:] == (3, 3), f"K.shape must be (..., 3, 3) but got {K.shape}"
        fx, fy, cx, cy = K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]
        K = torch.stack([fx, fy, cx, cy], dim=-1)

    batch_size = K.shape[:-1]
    center_uv = K[..., None, 2:4].clone()

    corners_uv = get_corners(uv_range)
    corners_uv = corners_uv.expand(*batch_size, -1, -1).to(K.device, K.dtype)

    frustum_origins, frustum_dirs = get_rays_from_cameras(
        K=K[..., None, :],
        uv=torch.cat([corners_uv, center_uv], dim=-2),
        plucker=False,
    ).split([3, 3], dim=-1)

    frustum_origins, cp_origin = (
        frustum_origins[..., :-1, :],
        frustum_origins[..., -1, :],
    )
    frustum_dirs, cp_dir = frustum_dirs[..., :-1, :], frustum_dirs[..., -1, :]

    tl, tr, bl, br = (frustum_origins + frustum_dirs * frustum_scale).unbind(
        dim=-2
    )  # [..., 3], [..., 3], [..., 3], [..., 3]
    cp = cp_origin + cp_dir * frustum_scale * center_ray_mult  # [..., 3]

    origins = torch.zeros_like(tl)
    center_ray = torch.stack([origins, cp], dim=-2)  # [..., 2, 3]
    frustum = torch.stack(
        [
            torch.stack([tl, bl], dim=-2),  # [..., 2, 3]
            torch.stack([tr, br], dim=-2),  # [..., 2, 3]
            torch.stack([tl, tr], dim=-2),  # [..., 2, 3]
            torch.stack([bl, br], dim=-2),  # [..., 2, 3]
            torch.stack([origins, tl], dim=-2),  # [..., 2, 3]
            torch.stack([origins, bl], dim=-2),  # [..., 2, 3]
            torch.stack([origins, tr], dim=-2),  # [..., 2, 3]
            torch.stack([origins, br], dim=-2),  # [..., 2, 3]
            center_ray,
        ],
        dim=-3,
    )  # shape: [..., 9, 2, 3]:                     dim[-2] is (start, end)

    frustum_width = (tl - tr).norm(dim=-1)
    frustum_height = (tl - bl).norm(dim=-1)
    frustum_center = (tl + bl + tr + br) / 4

    return frustum, frustum_width, frustum_height, frustum_center


def get_corners(
    uv_range: Sequence[float | int | Sequence[float | int]] = [0, 1]
) -> torch.Tensor:
    """
    Get the corners of the image in normalized coordinates.
    Returns:
        np.ndarray: [4, 2]
            - [top_left, top_right, bottom_left, bottom_right]
    """
    assert len(uv_range) == 2
    if isinstance(uv_range[0], int) or isinstance(uv_range[0], float):
        assert isinstance(uv_range[1], int) or isinstance(
            uv_range[1], float
        ), "uv_range must be a sequence of two numbers or a sequence of two sequences of numbers"
        uv_range = [uv_range, uv_range]
    Hrange, Wrange = uv_range

    u, v = torch.meshgrid(
        torch.linspace(Wrange[0], Wrange[1], 2),
        torch.linspace(Hrange[0], Hrange[1], 2),
        indexing="xy",
    )
    uv = torch.stack([u, v], dim=-1).flatten(start_dim=-3, end_dim=-2)
    return uv


# @torch.autocast(device_type="cuda", enabled=False)
def get_rays_from_cameras(
    K: Tensor,
    uv: Tensor,
    R_wc: Tensor | None = None,
    T_wc: Tensor | None = None,
    plucker: bool = True,
):
    """
    K:        [..., 3, 3] or [..., 4]  Camera intrinsic matrix.
    uv:       [..., 2]     Pixel coordinates.
    R_wc:     [..., 3, 3]  Rotation matrix that maps points in camera frame to world frame
    T_wc:     [..., 3]     Translation vector that maps points in camera frame to world frame

    Returns:
        origins: [..., :3]  Origin vectors. If plucker is True, this will be the moment vector.
        dirs: [..., 3:]     Direction vectors. If plucker is True, this will be normalized.
    """
    K, uv = K.float(), uv.float()
    if K.shape[-1] == 4:
        fx, fy, cx, cy = K.unbind(-1)
    else:
        assert K.shape[-2:] == (3, 3), f"K.shape must be (..., 3, 3) but got {K.shape}"
        fx, fy, cx, cy = K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]

    x_cam = (uv[..., 0] - cx) / fx
    y_cam = (uv[..., 1] - cy) / fy
    z_cam = torch.ones_like(x_cam)
    dirs = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    if plucker:
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    if R_wc is not None:
        R_wc = R_wc.float()
        dirs = torch.einsum("...ij,...j->...i", R_wc, dirs)
    if T_wc is None:
        origins = torch.zeros_like(dirs)
    else:
        T_wc = T_wc.float()
        origins = T_wc.expand_as(dirs)

    if plucker:
        origins = torch.cross(origins, dirs, dim=-1)  # shape [..., 3]

    return torch.cat([origins, dirs], dim=-1)


def invert_extrinsics(R: Tensor, T: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Convert camera-world (cw) to world-camera (wc) coordinates.

    Args:
        R: [..., 3, 3]  Rotation matrix from camera to world frame
        T: [..., 3]     Translation vector from camera to world frame

    Returns:
        Rinv: [..., 3, 3]  Rotation matrix from world to camera frame
        Tinv: [..., 3]     Translation vector from world to camera frame
    """
    Rinv = R.transpose(-1, -2)
    Tinv = -torch.einsum("...ij,...j->...i", Rinv, T)
    return Rinv, Tinv


def rebase_reference_frame(R_wc: Tensor, T_wc: Tensor) -> Tuple[Tensor, Tensor]:
    R_rw, T_rw = invert_extrinsics(R_wc[:, 0], T_wc[:, 0])
    R_rc, T_rc = multiply_extrinsics(R_rw, T_rw, R_wc, T_wc)
    return R_rc, T_rc


def multiply_extrinsics(
    R1: Tensor, T1: Tensor, R2: Tensor, T2: Tensor
) -> Tuple[Tensor, Tensor]:
    R1R2 = torch.einsum("...ij,...jk->...ik", R1, R2)
    Tnew = torch.einsum("...ij,...j->...i", R1, T2) + T1
    return R1R2, Tnew


# Added by Chonghyuk Song (2025-07-23)
def generate_trajectory_orbit(
    num_frames: int,
    height: float,  # y-axis coordinate of camera center
    start_rotation_angle: float = 0,  # start rotation angle of camera (degrees)
    target_rotation_angle: float = 360,  # target rotation angle of camera (degrees)
    radius: float = 0.5,  # radius of circular orbit
    normalize_wrt_first_frame: bool = False,
) -> Float[torch.Tensor, "N 4 4"]:
    """
    Create a cam2world trajectory depicting an orbital motion - camera moves in a circular path in the xz plane
    while always looking at the center point (0,height,0).
    """
    angles = torch.linspace(
        start_rotation_angle, target_rotation_angle, num_frames
    )  # in degrees
    angles_rad = angles * (torch.pi / 180.0)  # in radians

    # create cam2world matrices
    cam2worlds = []
    for angle in angles_rad:
        # Calculate camera position on the circle
        cam_x = radius * torch.cos(angle)
        cam_z = radius * torch.sin(angle)
        cam_pos = torch.tensor([cam_x, height, cam_z], dtype=torch.float32)

        # Calculate rotation matrix to look at center
        # Forward direction = center - cam_pos (normalized)
        forward = -cam_pos  # looking at origin
        forward[1] = 0  # zero out y component
        forward = forward / torch.norm(forward)

        # Right direction = up cross forward
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        right = torch.cross(up, forward)
        right = right / torch.norm(right)

        # Recompute up to ensure orthogonality
        up = torch.cross(forward, right)

        # Create rotation matrix
        R = torch.stack([right, up, forward], dim=1)  # each vector is a row

        # Create full pose matrix
        cam2world = torch.eye(4, dtype=torch.float32)
        cam2world[:3, :3] = R
        cam2world[:3, 3] = cam_pos

        cam2worlds.append(cam2world)

    # Combine poses
    cam2worlds = torch.stack(cam2worlds)

    if normalize_wrt_first_frame:
        cam2worlds = torch.linalg.solve(cam2worlds[0:1], cam2worlds)

    return cam2worlds


def pairwise_squared_distances_batchwise(points: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances for a batch of point sets.

    Args:
        points: (B, N, D) tensor of B point sets each with N points of dimension D.

    Returns:
        (B, N, N) tensor of pairwise squared distances.
    """
    # ||x_i||^2
    norms = (points**2).sum(dim=-1, keepdim=True)  # (B, N, 1)

    # x_i · x_j
    dot_products = torch.bmm(points, points.transpose(1, 2))  # (B, N, N)

    # Apply the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    dists_squared = norms + norms.transpose(1, 2) - 2 * dot_products
    dists_squared = torch.clamp(dists_squared, min=0.0)  # avoid negatives

    return dists_squared


def compute_interpolant_chunks(
    conditions: Float[torch.Tensor, "B N 2 chunk_size ..."],
    orbit_radius: float = None,
    start_rotation_angle: float = 0,
    return_conditions: bool = True,
) -> Float[torch.Tensor, "B N chunk_size ..."]:

    device = conditions.device
    batch_size = conditions.shape[0]
    chunk_size = conditions.shape[3]
    num_chunk_pairs = conditions.shape[1]  # n

    conditions = rearrange(
        conditions, "b n pair chunk_size ... -> b (n pair chunk_size) ..."
    )
    cams = CameraPose.from_vectors(conditions)
    intrinsics = cams.intrinsics(flatten=True)
    intrinsics = rearrange(
        intrinsics, "b (n pc) ... -> (b n) pc ...", n=num_chunk_pairs, pc=2 * chunk_size
    )  # shape = (B*N_pairs, 2*chunk_size, 4)
    cam2worlds = cams.cam2world_4x4()
    cam2worlds = rearrange(
        cam2worlds, "b (n pc) i j -> (b n) pc i j", n=num_chunk_pairs, pc=2 * chunk_size
    )  # shape = (B*N_pairs, 2*chunk_size, 4, 4)
    cam2worlds_T = cam2worlds[..., :3, 3]  # shape = (B*N_pairs, 2*chunk_size, 3)

    # for each manual chunk pair, we compute the interpolant chunks as a set of camera poses that orbit around the centroid of the input conditions
    # the interpolant chunks orbit on a plane that best fits the viewing vectors of the input conditions
    #

    # 1) if the orbit radius is not provided, is set to be half the maximum span of the input conditions
    if orbit_radius is None:
        # compute the maximum span of the input conditions
        orbit_radii = (
            0.5
            * torch.max(
                torch.max(
                    pairwise_squared_distances_batchwise(cam2worlds_T), dim=-1
                ).values,
                dim=-1,
            ).values
        )  # shape = (B*N_pairs,)

    # 2) compute the centroid and the plane that best fits the camera centers
    #    the orbit will lie on this plane
    centroid = cam2worlds_T.mean(dim=1, keepdim=True)  # shape = (B*N_pairs, 1, 3)
    positions_centered = cam2worlds_T - centroid  # shape (B*N_pairs, 2*chunk_size, 3)
    U, S, Vh = torch.linalg.svd(
        positions_centered, full_matrices=False
    )  # Vh: (B*N_pairs, 3, 3)
    basis_x = Vh[:, 0, :]  # shape (B*N_pairs, 3): first in-plane axis
    basis_y = Vh[:, 1, :]  # shape (B*N_pairs, 3): second in-plane axis
    normal = Vh[
        :, 2, :
    ]  # shape (B*N_pairs, 3): plane normal (least principal component)
    R = torch.stack([basis_x, normal, basis_y], dim=-1)  # shape (B*N_pairs, 3, 3)
    orbit_center2world = torch.eye(4, device=device, dtype=torch.float32)
    orbit_center2world = repeat(
        orbit_center2world, "i j -> bn i j", bn=batch_size * num_chunk_pairs
    ).contiguous()
    orbit_center2world[..., :3, :3] = R
    orbit_center2world[..., :3, 3] = centroid[:, 0]
    orbit_center2world = rearrange(orbit_center2world, "bn i j -> bn () i j")

    # 3) compute orbit trajectory on this plane defined by orbit_center2world
    if orbit_radius is None:
        orbit_trajectories = torch.stack(
            [
                generate_trajectory_orbit(
                    chunk_size + 1,
                    height=0,
                    start_rotation_angle=start_rotation_angle,
                    target_rotation_angle=360 + start_rotation_angle,
                    radius=orbit_radius,
                    normalize_wrt_first_frame=False,
                )[1:]
                for orbit_radius in orbit_radii
            ],
            dim=0,
        ).to(
            device
        )  # shape (B*N_pairs,chunk_size,4,4)
    else:
        orbit_trajectories = generate_trajectory_orbit(
            chunk_size + 1,
            height=0,
            start_rotation_angle=start_rotation_angle,
            target_rotation_angle=360 + start_rotation_angle,
            radius=orbit_radius,
            normalize_wrt_first_frame=False,
        )[1:]
        orbit_trajectories = repeat(
            orbit_trajectories,
            "chunk_size i j -> bn chunk_size i j",
            bn=batch_size * num_chunk_pairs,
        ).to(device)

    # 4)normalize orbit trajectory s.t. center of orbit is at cam2world_orbit_center
    # it is currently centered at the origin
    cam2worlds_interpolant_chunks = torch.matmul(
        orbit_center2world, orbit_trajectories
    ).float()  # shape = (B*N_pairs, chunk_size, 4, 4)
    dummy = orbit_center2world.dtype

    # 5) convert back to conditions format
    if return_conditions:
        extrinsics_interpolant_chunks = torch.linalg.inv(cam2worlds_interpolant_chunks)[
            ..., :3, :
        ]
        extrinsics_interpolant_chunks = rearrange(
            extrinsics_interpolant_chunks, "... i j -> ... (i j)"
        )
        conditions_interpolant_chunks = torch.cat(
            [intrinsics[:, :chunk_size], extrinsics_interpolant_chunks], dim=-1
        )
        conditions_interpolant_chunks = rearrange(
            conditions_interpolant_chunks,
            "(b n) chunk_size ... -> b n chunk_size ...",
            n=num_chunk_pairs,
            chunk_size=chunk_size,
        )
        return conditions_interpolant_chunks
    else:
        cam2worlds_interpolant_chunks = rearrange(
            cam2worlds_interpolant_chunks,
            "(b n) chunk_size ... -> b n chunk_size ...",
            n=num_chunk_pairs,
            chunk_size=chunk_size,
        )
        return cam2worlds_interpolant_chunks

