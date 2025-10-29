import math
import torch
from torch import Tensor
from einops import rearrange, repeat
from typing import List, Optional, Tuple
from jaxtyping import Float, Bool
from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import numpy as np
import matplotlib.pyplot as plt

from utils.geometry_utils import CameraPose

from typing import Literal, Union

QueryAggregationType = Literal["mean", "last"]
QueryAggregation = Union[QueryAggregationType, int]


class BaseConditionDistance(ABC):
    """Base class for computing distances between conditions for retrieval"""

    def __init__(
        self,
        query_aggregation: QueryAggregation = "last",
        max_distance_threshold: Optional[float] = None,
    ):
        self.query_aggregation = query_aggregation
        self.max_distance_threshold = max_distance_threshold

    @abstractmethod
    def __call__(
        self, query_conditions: Tensor, candidate_conditions: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute distances between query and candidate conditions

        Args:
            query_conditions: Tensor of shape (B, h, C)
            candidate_conditions: Tensor of shape (B, N, C)

        Returns:
            distances: Tensor of shape (B, N) - distances between queries and each candidate
            mask: Optional tensor of shape (B, N) indicating if distances are above threshold
        """
        pass

    def reduce_distances(
        self, distances: Float[Tensor, "B N h"]
    ) -> Float[Tensor, "B N"]:
        """
        Return a query that is the target of the retrieval
        """
        if self.query_aggregation == "mean":
            return distances.mean(dim=1)
        elif self.query_aggregation == "last":
            return distances[:, :, -1]
        else:
            return distances[:, :, self.query_aggregation]

    def _compute_mask(
        self, distances: Float[Tensor, "B N"]
    ) -> Optional[Float[Tensor, "B N"]]:
        """
        Compute mask indicating if distances are above threshold
        """
        if self.max_distance_threshold is None:
            return None
        return (distances <= self.max_distance_threshold).float()


class RotationDistance(BaseConditionDistance):
    """Simple distance metric based on rotation angle differences between poses"""

    def __init__(
        self,
        query_aggregation: QueryAggregation = "last",
        max_distance_threshold: Optional[float] = None,
    ):
        super().__init__(query_aggregation, max_distance_threshold)

    def __call__(
        self,
        query_conditions: Float[Tensor, "B h 16"],
        candidate_conditions: Float[Tensor, "B N 16"],
    ) -> Tuple[Float[Tensor, "B N"], Optional[Float[Tensor, "B N"]]]:
        """
        Compute rotation angle differences between query and candidate poses

        Args:
            query_conditions: Query camera poses with shape [B, h, 16] (4 intrinsics + 12 extrinsics)
            candidate_conditions: Candidate camera poses with shape [B, N, 16]

        Returns:
            distances: Mean rotation angle differences with shape [B, N]
            mask: Optional mask indicating if distances are above threshold
        """
        # Convert to camera poses
        query_poses = CameraPose.from_vectors(query_conditions)
        candidate_poses = CameraPose.from_vectors(candidate_conditions)

        # Get rotation matrices
        query_rot = query_poses.extrinsics()[..., :3, :3].cpu().numpy()  # [B, h, 3, 3]
        cand_rot = (
            candidate_poses.extrinsics()[..., :3, :3].cpu().numpy()
        )  # [B, N, 3, 3]

        B, h, _, _ = query_rot.shape
        _, N, _ = candidate_conditions.shape

        # Compute rotation differences for each pair of poses
        distances = torch.zeros((B, N, h), device=query_conditions.device)

        for b in range(B):
            for n in range(N):
                for t in range(h):
                    # Get relative rotation between poses
                    rel_rot = query_rot[b, t] @ cand_rot[b, n].T
                    # Convert to angle
                    angle = abs(Rotation.from_matrix(rel_rot).magnitude())
                    # Store angle directly in the 3D tensor
                    distances[b, n, t] = torch.tensor(
                        angle, device=query_conditions.device
                    )

        # Reduce the time dimension according to the aggregation method
        distances = self.reduce_distances(distances)
        mask = self._compute_mask(distances)
        return distances, mask


class TranslationDistance(BaseConditionDistance):
    """Distance metric based on L2 distance between camera positions"""

    def __init__(
        self,
        query_aggregation: QueryAggregation = "last",
        max_distance_threshold: Optional[float] = None,
    ):
        super().__init__(query_aggregation, max_distance_threshold)

    def __call__(
        self,
        query_conditions: Float[Tensor, "B h 16"],
        candidate_conditions: Float[Tensor, "B N 16"],
    ) -> Tuple[Float[Tensor, "B N"], Optional[Float[Tensor, "B N"]]]:
        """
        Compute L2 distances between query and candidate camera positions

        Args:
            query_conditions: Query camera poses with shape [B, h, 16] (4 intrinsics + 12 extrinsics)
            candidate_conditions: Candidate camera poses with shape [B, N, 16]

        Returns:
            distances: Mean L2 distances between positions with shape [B, N]
            mask: Optional mask indicating if distances are above threshold
        """
        # Convert to camera poses
        query_poses = CameraPose.from_vectors(query_conditions)
        candidate_poses = CameraPose.from_vectors(candidate_conditions)

        # Get positions (last column of extrinsics)
        query_pos = query_poses.extrinsics()[..., :3, 3]  # [B, h, 3]
        cand_pos = candidate_poses.extrinsics()[..., :3, 3]  # [B, N, 3]

        # Compute L2 distances for each query-candidate pair
        B, h, _ = query_pos.shape
        _, N, _ = cand_pos.shape

        # Expand dimensions for broadcasting
        query_pos = query_pos.unsqueeze(1)  # [B, 1, h, 3]
        cand_pos = cand_pos.unsqueeze(2)  # [B, N, 1, 3]

        # Compute L2 distances
        distances = torch.norm(query_pos - cand_pos, dim=-1)  # [B, N, h]

        # Average over time dimension
        distances = self.reduce_distances(distances)
        mask = self._compute_mask(distances)
        return distances, mask


class ImageCenterDistance(BaseConditionDistance):
    """
    Distance metric based on L2 distance between image centers, where each center is computed
    by extending 2 units from the camera position along the optical axis (rotation direction).
    """

    def __init__(
        self,
        extension_distance: float = 2.0,
        query_aggregation: QueryAggregation = "last",
        max_distance_threshold: Optional[float] = None,
    ):
        super().__init__(query_aggregation, max_distance_threshold)
        self.extension_distance = extension_distance

    def _compute_image_centers(self, poses: CameraPose) -> Float[Tensor, "... 3"]:
        """
        Compute image centers by extending along optical axis

        Args:
            poses: Camera poses

        Returns:
            centers: Image centers with same shape as input positions
        """
        # Get camera positions and rotation matrices
        positions = poses.extrinsics()[..., :3, 3]  # [..., 3]
        rotations = poses.extrinsics()[..., :3, :3]  # [..., 3, 3]

        # Get optical axis (negative z-axis in camera coordinates)
        optical_axis = rotations[..., :3, 2]  # [..., 3]

        # Compute image centers by extending along optical axis
        centers = positions + self.extension_distance * optical_axis

        return centers

    def __call__(
        self,
        query_conditions: Float[Tensor, "B h 16"],
        candidate_conditions: Float[Tensor, "B N 16"],
    ) -> Tuple[Float[Tensor, "B N"], Optional[Float[Tensor, "B N"]]]:
        """
        Compute L2 distances between query and candidate image centers

        Args:
            query_conditions: Query camera poses with shape [B, h, 16] (4 intrinsics + 12 extrinsics)
            candidate_conditions: Candidate camera poses with shape [B, N, 16]

        Returns:
            distances: Mean L2 distances between image centers with shape [B, N]
            mask: Optional mask indicating if distances are above threshold
        """
        # Convert to camera poses
        query_poses = CameraPose.from_vectors(query_conditions)
        candidate_poses = CameraPose.from_vectors(candidate_conditions)

        # Get image centers
        query_centers = self._compute_image_centers(query_poses)  # [B, h, 3]
        cand_centers = self._compute_image_centers(candidate_poses)  # [B, N, 3]

        # Compute L2 distances for each query-candidate pair
        B, h, _ = query_centers.shape
        _, N, _ = cand_centers.shape

        # Expand dimensions for broadcasting
        query_centers = query_centers.unsqueeze(1)  # [B, 1, h, 3]
        cand_centers = cand_centers.unsqueeze(2)  # [B, N, 1, 3]

        # Compute L2 distances
        distances = torch.norm(query_centers - cand_centers, dim=-1)  # [B, N, h]

        # Average over time dimension
        distances = self.reduce_distances(distances)
        mask = self._compute_mask(distances)
        return distances, mask


@dataclass
class CombinedDistance(BaseConditionDistance):
    """
    Combines rotation and translation distances with specified weights

    Args:
        rotation_weight: Weight for rotation distance
        translation_weight: Weight for translation distance
    """

    rotation_weight: float = 1.0
    translation_weight: float = 1.0
    query_aggregation: QueryAggregation = "last"
    max_distance_threshold: Optional[float] = None

    def __post_init__(self):
        # Initialize rotation and translation distance metrics
        self.rotation_distance = RotationDistance(
            query_aggregation=self.query_aggregation
        )
        self.translation_distance = TranslationDistance(
            query_aggregation=self.query_aggregation
        )

    def __call__(
        self,
        query_conditions: Float[Tensor, "B h 16"],
        candidate_conditions: Float[Tensor, "B N 16"],
    ) -> Tuple[Float[Tensor, "B N"], Optional[Float[Tensor, "B N"]]]:
        """
        Compute weighted combination of rotation and translation distances

        Args:
            query_conditions: Query camera poses with shape [B, h, 16]
            candidate_conditions: Candidate camera poses with shape [B, N, 16]

        Returns:
            distances: Combined distances with shape [B, N]
            mask: Optional mask indicating if distances are above threshold
        """
        rot_dist, rot_mask = self.rotation_distance(
            query_conditions, candidate_conditions
        )
        trans_dist, trans_mask = self.translation_distance(
            query_conditions, candidate_conditions
        )

        # Normalize distances to similar scales
        rot_dist = rot_dist / np.pi  # Normalize rotation to [0, 1]
        trans_dist = (
            (trans_dist / trans_dist.max()) if trans_dist.max() > 0 else 0
        )  # Normalize translation to [0, 10]

        # Combine distances with weights
        combined_dist = (
            self.rotation_weight * rot_dist + self.translation_weight * trans_dist
        )

        mask = self._compute_mask(combined_dist)

        return combined_dist, mask


class TimeDecreasing(BaseConditionDistance):
    """Distance metric that assigns decreasing similarities based on temporal order"""

    def __init__(
        self,
        query_aggregation: QueryAggregation = "last",
        max_distance_threshold: Optional[float] = None,
    ):
        super().__init__(query_aggregation, max_distance_threshold)

    def __call__(
        self,
        query_conditions: Float[Tensor, "B h 16"],
        candidate_conditions: Float[Tensor, "B N 16"],
    ) -> Tuple[Float[Tensor, "B N"], Optional[Float[Tensor, "B N"]]]:
        """
        Compute distances that decrease linearly with time, ignoring actual poses.
        First frame gets minimal distance (maximal similarity), increasing linearly.

        Args:
            query_conditions: Query camera poses with shape [B, h, 16]
            candidate_conditions: Candidate camera poses with shape [B, N, 16]

        Returns:
            distances: Linearly increasing distances with shape [B, N]
            mask: Optional mask indicating if distances are above threshold
        """
        B, h, _ = query_conditions.shape
        _, N, _ = candidate_conditions.shape

        # Create linearly increasing distances from 0 to pi
        # This makes similarities decrease from 1 to 0 when converted by 1 - dist/pi
        distances = torch.linspace(0, np.pi, N, device=query_conditions.device)

        # Expand to match batch size
        distances = distances.unsqueeze(0).expand(B, -1)

        mask = None
        if self.max_distance_threshold is not None:
            mask = (distances <= self.max_distance_threshold).float()

        return distances, mask


class FOVDistance(BaseConditionDistance):
    """
    Distance metric based on field of view overlap between cameras.
    Computes the overlap between camera frustums and returns a distance
    that is inversely proportional to the overlap (smaller distance = more overlap).
    Uses actual camera intrinsics for accurate frustum computation.
    """

    def __init__(
        self,
        frustum_length: float = 2.0,
        num_samples: int = 500,
        resolution: int = 256,
        query_aggregation: QueryAggregation = "last",
        max_distance_threshold: Optional[float] = None,
        fix_intrinsics: bool = False,
    ):
        super().__init__(query_aggregation, max_distance_threshold)
        self.frustum_length = frustum_length
        self.num_samples = num_samples
        self.resolution = resolution
        self.fix_intrinsics = fix_intrinsics

    def _compute_frustum_corners(self, poses: CameraPose) -> Float[Tensor, "... 4 3"]:
        """
        Compute the 4 corners of the camera frustum at the given distance
        using actual camera intrinsics.

        Args:
            poses: Camera poses

        Returns:
            corners: Frustum corners with shape [..., 4, 3]
        """
        # Get world-to-camera extrinsics
        extrinsics = poses.extrinsics()  # [..., 3, 4]
        R_w2c = extrinsics[..., :3, :3]  # World-to-camera rotation
        T_w2c = extrinsics[..., :3, 3]  # World-to-camera translation

        # Convert to camera-to-world
        R_c2w = torch.transpose(R_w2c, -1, -2)  # Invert rotation by transposing
        cam_pos_world = -torch.matmul(R_c2w, T_w2c.unsqueeze(-1)).squeeze(
            -1
        )  # -R.T @ T

        # Get camera axes in world coordinates
        right = R_c2w[..., :3, 0]  # [..., 3]
        up = R_c2w[..., :3, 1]  # [..., 3]
        forward = R_c2w[..., :3, 2]  # [..., 3]

        # Get camera intrinsics
        intrinsics = poses._K  # [..., 4] - contains fx, fy, cx, cy

        # Extract individual components
        fx = intrinsics[..., 0]  # [...]
        fy = intrinsics[..., 1]  # [...]
        cx = intrinsics[..., 2]  # [...]
        cy = intrinsics[..., 3]  # [...]

        # Compute frustum corners
        # Start from camera position and extend along forward axis
        center = cam_pos_world + self.frustum_length * forward

        # Calculate the corners using intrinsics
        shape = intrinsics.shape[:-1]

        # Calculate frustum dimensions based on intrinsics
        # For a normalized image plane (0-1), the corners are at (0,0), (1,0), (1,1), (0,1)
        # Map these to rays through the camera using the intrinsics
        # Calculate width and height at frustum_length
        # Note: We need to project the corners of image space (0,0) and (1,1) to 3D
        # This is the correct formula:
        width_half = self.frustum_length / fx  # Half width at frustum_length
        height_half = self.frustum_length / fy  # Half height at frustum_length

        # Prepare for broadcasting
        right_offset = right.clone()  # [..., 3]
        up_offset = up.clone()  # [..., 3]

        # Reshaping for broadcasting
        width_half = width_half.reshape(*shape, 1)  # [..., 1]
        height_half = height_half.reshape(*shape, 1)  # [..., 1]

        # Scale the right and up vectors
        right_offset = right_offset * width_half  # [..., 3]
        up_offset = up_offset * height_half  # [..., 3]

        # Account for principal point offset
        cx_offset = (cx - 0.5).reshape(*shape, 1) * 2 * width_half * right
        cy_offset = (cy - 0.5).reshape(*shape, 1) * 2 * height_half * up

        # Center point adjusted for principal point offset
        center = center + cx_offset + cy_offset

        # Compute all 4 corners
        corners = torch.stack(
            [
                center + right_offset + up_offset,  # top right
                center + right_offset - up_offset,  # bottom right
                center - right_offset - up_offset,  # bottom left
                center - right_offset + up_offset,  # top left
            ],
            dim=-2,
        )  # [..., 4, 3]

        return corners

    def _compute_overlap(
        self,
        corners1: Float[Tensor, "B h 4 3"],
        corners2: Float[Tensor, "B N 4 3"],
        poses1: Optional[CameraPose] = None,
        poses2: Optional[CameraPose] = None,
    ) -> Float[Tensor, "B N h"]:
        """
        Compute frustum overlap using Monte Carlo sampling.
        Samples points inside each frustum and checks how many are contained in the other frustum.

        Args:
            corners1: First frustum corners [..., 4, 3]
            corners2: Second frustum corners [..., 4, 3]
            poses1: First camera poses (optional, will extract camera positions if provided)
            poses2: Second camera poses (optional, will extract camera positions if provided)

        Returns:
            overlap: Estimated overlap ratio [0, 1]
        """
        # Number of Monte Carlo samples
        # num_samples = 500
        num_samples = self.num_samples
        device = corners1.device

        # Get camera positions directly from poses if available
        def get_camera_positions(poses):
            # The extrinsics in CameraPose are world-to-camera, as documented in geometry_utils.py
            # To get camera position in world coordinates, we need to convert:
            # cam_pos_world = -R.T @ T (where R, T are the world-to-camera rotation and translation)
            R = poses.extrinsics()[..., :3, :3]  # World-to-camera rotation
            T = poses.extrinsics()[..., :3, 3]  # World-to-camera translation

            # Convert world-to-camera -> camera-to-world to get camera position
            R_inv = torch.transpose(R, -1, -2)  # Invert rotation by transposing
            cam_pos = -torch.matmul(R_inv, T.unsqueeze(-1)).squeeze(-1)  # -R.T @ T
            return cam_pos

        # Get camera positions
        cam_pos1: Float[Tensor, "B h 3"] = get_camera_positions(poses1)
        cam_pos2: Float[Tensor, "B N 3"] = get_camera_positions(poses2)

        # Function to generate random points inside a frustum
        def sample_points_in_frustum(corners, cam_pos, num_points):
            # Get batch dimensions
            batch_dims = corners.shape[:-2]  # All dimensions except the last two (4, 3)
            total_batch_size = (
                torch.prod(torch.tensor(batch_dims)).item() if batch_dims else 1
            )

            # Reshape for broadcasting
            corners_flat = corners.reshape(total_batch_size, 4, 3)  # [B, 4, 3]
            cam_pos_flat = cam_pos.reshape(total_batch_size, 1, 3)  # [B, 1, 3]

            # Initialize arrays to hold samples for each batch item
            all_samples = []

            # For each batch item
            for b in range(total_batch_size):
                # Get batch-specific corners and camera position
                b_corners = corners_flat[b]  # [4, 3]
                b_cam_pos = cam_pos_flat[b, 0]  # [3]

                # Generate samples
                samples = []
                for _ in range(num_points):
                    # Randomly choose 3 vertices from the frustum (camera + 2 corners)
                    # to form a tetrahedron, then sample within it

                    # Always include the camera position
                    vertices = [b_cam_pos]

                    # Randomly select 2 frustum corners
                    indices = torch.randperm(4)[:2]
                    vertices.extend([b_corners[i] for i in indices])

                    # Add a fourth vertex by interpolating between other corners
                    remaining_indices = set(range(4)) - set(indices.tolist())
                    r_idx = list(remaining_indices)[0]
                    vertices.append(b_corners[r_idx])

                    # Convert vertices to tensor
                    vertices = torch.stack(vertices)  # [4, 3]

                    # Sample barycentric coordinates
                    barycentric = torch.rand(4, device=device)
                    barycentric = barycentric / barycentric.sum()

                    # Compute point using barycentric coordinates
                    point = (vertices * barycentric.unsqueeze(-1)).sum(dim=0)  # [3]
                    samples.append(point)

                # Stack all samples for this batch item
                samples = torch.stack(samples)  # [num_points, 3]
                all_samples.append(samples)

            # Combine all batches
            all_samples: Float[Tensor, "Be num_points 3"] = torch.stack(all_samples)

            # Reshape to match original batch dimensions
            if batch_dims:
                all_samples: Float[Tensor, "B ? num_points 3"] = all_samples.reshape(
                    *batch_dims, num_points, 3
                )

            return all_samples

        # Function to check if points are inside a frustum
        def points_in_frustum(
            points: Float[
                Tensor, "B P Q 3"
            ],  # Points to test, B batches, P points per batch, Q query points per point
            corners: Float[
                Tensor, "B T 4 3"
            ],  # Frustum corners, B batches, T target cameras
            cam_pos: Float[
                Tensor, "B T 3"
            ],  # Camera positions, B batches, T target cameras
        ) -> Bool[
            Tensor, "B P Q T"
        ]:  # Returns boolean mask indicating which points are in which frustums
            """
            Check if points are inside camera frustums.

            Args:
                points: Points to test, shape [B, P, Q, 3] where:
                    B is batch size
                    P is number of points per batch
                    Q is number of query points per point
                corners: Frustum corners, shape [B, T, 4, 3] where:
                    B is batch size
                    T is number of target cameras
                cam_pos: Camera positions, shape [B, T, 3]

            Returns:
                Boolean mask of shape [B, P, Q, T] indicating which points are inside which frustums
            """
            device = points.device

            # Get shapes
            B, P, Q, _ = points.shape
            _, T, _, _ = corners.shape

            # Reshape for broadcasting
            # Points: [B, P, Q, 1, 3] to broadcast against T target cameras
            points = points.unsqueeze(3)  # [B, P, Q, 1, 3]

            # Corners: [B, 1, 1, T, 4, 3] to broadcast against P points and Q queries
            corners = corners.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T, 4, 3]

            # Camera positions: [B, 1, 1, T, 3] to broadcast against P points and Q queries
            cam_pos = cam_pos.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T, 3]

            # Initialize result tensor
            inside: Bool[Tensor, "B P Q T"] = torch.ones(
                (B, P, Q, T), device=device, dtype=torch.bool
            )

            # For each batch
            for b in range(B):
                # For each target camera
                for t in range(T):
                    # Get frustum planes for this camera
                    b_corners = corners[b, 0, 0, t]  # [4, 3]
                    b_cam_pos = cam_pos[b, 0, 0, t]  # [3]

                    # Define the 5 planes of the frustum (4 side planes and 1 far plane)
                    planes = []

                    # Create 4 side planes
                    for i in range(4):
                        # Get 3 points: camera, current corner, next corner
                        next_i = (i + 1) % 4
                        plane_points = torch.stack(
                            [b_cam_pos, b_corners[i], b_corners[next_i]]
                        )

                        # Compute plane normal and constant
                        v1 = plane_points[1] - plane_points[0]
                        v2 = plane_points[2] - plane_points[0]
                        normal = torch.cross(v1, v2)
                        normal = normal / (torch.norm(normal) + 1e-10)
                        d = -torch.sum(normal * plane_points[0])

                        planes.append((normal, d))

                    # Create far plane
                    far_center = b_corners.mean(dim=0)
                    # Calculate two edges of the rectangle
                    edge1 = b_corners[1] - b_corners[0]
                    edge2 = b_corners[3] - b_corners[0]
                    # Normal is cross product of edges
                    normal = torch.cross(edge1, edge2)
                    normal = normal / (torch.norm(normal) + 1e-10)

                    # We need to negate the distance to get the correct side of the plane
                    # because the normal points inward
                    normal = -normal

                    d = -torch.sum(normal * far_center)

                    planes.append((normal, d))

                    # Test each point against all planes
                    b_points = points[b]  # [P, Q, 1, 3]

                    # A point is inside if it's on the correct side of all planes
                    for normal, d in planes:
                        # Compute signed distance to plane
                        dist = torch.sum(normal * b_points, dim=-1) + d  # [P, Q, 1]
                        # Point is inside if dist <= 0
                        inside[b, :, :, t] = inside[b, :, :, t] & (
                            dist.squeeze(-1) <= 0
                        )

            return inside

        # Sample points in both frustums
        samples1: Float[Tensor, "B h num_samples 3"] = sample_points_in_frustum(
            corners1, cam_pos1, num_samples
        )
        samples2: Float[Tensor, "B N num_samples 3"] = sample_points_in_frustum(
            corners2, cam_pos2, num_samples
        )

        # Check which points from frustum1 are in frustum2
        in_frustum2: Bool[Tensor, "B h num_samples N"] = points_in_frustum(
            samples1, corners2, cam_pos2
        )
        # Check which points from frustum2 are in frustum1
        in_frustum1: Bool[Tensor, "B N num_samples h"] = points_in_frustum(
            samples2, corners1, cam_pos1
        )

        # Calculate intersection and union
        intersection: Float[Tensor, "B N h"] = in_frustum1.float().sum(
            dim=-2
        ) + in_frustum2.float().sum(dim=-2).transpose(-1, -2)
        union: int = 2 * num_samples

        # Compute intersection over union (IoU)
        iou: Float[Tensor, "B N h"] = intersection / union

        # # Account for view alignment
        # # Compute normals of far planes
        # def get_frustum_normal(corners):
        #     edge1 = corners[..., 0, :] - corners[..., 2, :]  # Diagonal from top-right to bottom-left
        #     edge2 = corners[..., 1, :] - corners[..., 3, :]  # Diagonal from bottom-right to top-left
        #     normal = torch.cross(edge1, edge2, dim=-1)
        #     normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-10)
        #     return normal

        # normal1 = get_frustum_normal(corners1)
        # normal2 = get_frustum_normal(corners2)

        # # Dot product of normals - negative for opposing views
        # alignment = torch.sum(normal1 * normal2, dim=-1)

        # # Use absolute alignment to include opposing views with high overlap
        # alignment_factor = torch.abs(alignment)

        # # Final overlap is IoU scaled by alignment factor
        # overlap = iou * alignment_factor
        overlap = iou

        # Ensure we never have exactly 0 or 1 for numerical stability
        overlap = torch.clamp(overlap, min=1e-6, max=1.0 - 1e-6)

        return overlap

    def __call__(
        self,
        query_conditions: Float[Tensor, "B h 16"],
        candidate_conditions: Float[Tensor, "B N 16"],
    ) -> Tuple[
        Float[Tensor, "B h N"], Float[Tensor, "B N"], Optional[Float[Tensor, "B N"]]
    ]:
        """
        Compute distances based on field of view overlap between cameras.
        Smaller distance means more overlap.

        Args:
            query_conditions: Query camera poses with shape [B, h, 16]
            candidate_conditions: Candidate camera poses with shape [B, N, 16]

        Returns:
            distances: Distances inversely proportional to overlap [B, N]
            mask: Optional mask indicating if distances are above threshold
        """

        # fix wrong fov of RE10k
        if self.fix_intrinsics:
            query_conditions[..., 0] = torch.max(
                query_conditions[..., 0], query_conditions[..., 1]
            )
            query_conditions[..., 1] = torch.max(
                query_conditions[..., 0], query_conditions[..., 1]
            )
            candidate_conditions[..., 0] = torch.max(
                candidate_conditions[..., 0], candidate_conditions[..., 1]
            )
            candidate_conditions[..., 1] = torch.max(
                candidate_conditions[..., 0], candidate_conditions[..., 1]
            )

        # Convert to camera poses
        query_poses = CameraPose.from_vectors(query_conditions)
        candidate_poses = CameraPose.from_vectors(candidate_conditions)

        # Compute frustum corners
        query_corners = self._compute_frustum_corners(query_poses)  # [B, h, 4, 3]
        cand_corners = self._compute_frustum_corners(candidate_poses)  # [B, N, 4, 3]

        # Compute overlap for all pairs
        overlap = self._compute_overlap(
            query_corners, cand_corners, poses1=query_poses, poses2=candidate_poses
        )  # [B, N, h]

        # Average overlap across query frames
        overlap_reduced = self.reduce_distances(overlap)
        # Convert overlap to distance (smaller distance = more overlap)
        # Use 1 - overlap to get distance in [0, 1] range
        distances = 1.0 - overlap_reduced

        mask = None
        if self.max_distance_threshold is not None:
            mask = distances <= self.max_distance_threshold

        return rearrange(overlap, "b n h -> b h n"), distances, mask


def retrieve_support_frames(
    query: Float[Tensor, "B h 16"],
    key: Float[Tensor, "B N 16"],
    value: Float[Tensor, "B N C H W"],
    n_support_frames: int,
    distance_fn: BaseConditionDistance,
) -> Tuple[
    Float[Tensor, "B n_support_frames 16"],
    Float[Tensor, "B n_support_frames C H W"],
    Optional[Float[Tensor, "B n_support_frames"]],
]:
    """
    Retrieve support frames based on a distance function between conditions

    Args:
        query: Query conditions with shape [B, h, 16] - query camera poses
        key: Database of candidate conditions with shape [B, N, 16]
        value: Database of candidate frames with shape [B, N, C, H, W]
        n_support_frames: Number of frames to retrieve
        distance_fn: Distance function to compute distances between conditions

    Returns:
        retrieved_key: Selected conditions with shape [B, n_support_frames, 16]
        retrieved_value: Selected frames with shape [B, n_support_frames, C, H, W]
        mask: Optional mask indicating if retrieved frames are within threshold
    """
    if key.shape[1] < n_support_frames:
        raise ValueError(
            f"Not enough candidate frames to retrieve support frames. \
        Expected at least {n_support_frames} candidates, but got only {key.shape[1]}. \
        Will incorporate explicit handling of this case in the future."
        )

    # Compute distances between current conditions and candidates
    distances, mask = distance_fn(query, key)  # [B, N], [B, N]

    # Get indices of closest candidates for each batch item
    _, indices = torch.topk(
        distances, k=n_support_frames, dim=1, largest=False
    )  # [B, k]

    # Retrieve the values and keys
    B = query.shape[0]
    retrieved_value = value[
        torch.arange(B).unsqueeze(1).to(indices.device), indices
    ]  # [B, k, C, H, W]
    retrieved_key = key[
        torch.arange(B).unsqueeze(1).to(indices.device), indices
    ]  # [B, k, 16]

    # If mask exists, retrieve corresponding mask values
    retrieved_mask = None
    if mask is not None:
        retrieved_mask = mask[
            torch.arange(B).unsqueeze(1).to(indices.device), indices
        ]  # [B, k]

    return retrieved_key, retrieved_value, retrieved_mask


def impute_support_frames(
    video: Float[Tensor, "batch sequence_length channels height width"],
    cond: Float[Tensor, "batch sequence_length condition_dim"],
    support_video: Float[Tensor, "batch num_retrieved channels height width"],
    support_cond: Float[Tensor, "batch num_retrieved condition_dim"],
    impute_indices: List[int],
    latent: Optional[Float[Tensor, "sequence_length latent_dim height_ width_"]] = None,
    support_latent: Optional[
        Float[Tensor, "batch num_retrieved latent_dim height_ width_"]
    ] = None,
    nonterminal: Optional[Bool[Tensor, "batch sequence_length"]] = None,
    valid_mask: Optional[Bool[Tensor, "batch num_retrieved"]] = None,
) -> Tuple[
    Float[Tensor, "batch new_length channels height width"],
    Float[Tensor, "batch new_length condition_dim"],
    Optional[Float[Tensor, "batch new_length latent_dim height_ width_"]],
    Optional[Bool[Tensor, "batch new_length"]],
    Optional[Bool[Tensor, "batch new_length"]],
]:
    batch_size = video.shape[0]
    sequence_length = video.shape[1]
    new_length = sequence_length + len(impute_indices)

    # Convert negative indices to positive and ensure they're valid
    impute_indices = [i if i >= 0 else new_length + i for i in impute_indices]
    impute_indices = sorted(impute_indices)

    if any(i >= new_length for i in impute_indices):
        raise ValueError(
            f"Impute indices {impute_indices} exceed new sequence length {new_length}"
        )

    # Initialize output tensors
    new_video = torch.zeros(
        (batch_size, new_length, *video.shape[2:]), device=video.device
    )
    new_cond = torch.zeros(
        (batch_size, new_length, *cond.shape[2:]), device=cond.device
    )
    if latent is not None:
        new_latent = torch.zeros(
            (batch_size, new_length, *latent.shape[2:]), device=latent.device
        )
    else:
        new_latent = None
    if nonterminal is not None:
        new_nonterminal = torch.zeros(
            (batch_size, new_length), device=nonterminal.device
        )
    else:
        new_nonterminal = None
    if valid_mask is not None:
        new_valid_mask = torch.zeros(
            (batch_size, new_length), dtype=bool, device=valid_mask.device
        )
    else:
        new_valid_mask = None

    # Create a mapping of new indices to source indices
    # -1 indicates positions where we'll insert retrieved frames
    index_map = [-1] * new_length
    orig_idx = 0
    for i in range(new_length):
        if i not in impute_indices:
            index_map[i] = orig_idx
            orig_idx += 1

    # Fill in the tensors using the index map
    for b in range(batch_size):
        retr_idx = 0
        for new_idx in range(new_length):
            if new_idx in impute_indices:
                # Place retrieved frame
                new_video[b, new_idx] = support_video[b, retr_idx]
                new_cond[b, new_idx] = support_cond[b, retr_idx]
                if latent is not None:
                    new_latent[b, new_idx] = support_latent[b, retr_idx]
                if nonterminal is not None:
                    new_nonterminal[b, new_idx] = 1
                if valid_mask is not None:
                    new_valid_mask[b, new_idx] = valid_mask[b, retr_idx]
                retr_idx += 1
            else:
                # Place original frame using the index map
                orig_idx = index_map[new_idx]
                new_video[b, new_idx] = video[b, orig_idx]
                new_cond[b, new_idx] = cond[b, orig_idx]
                if valid_mask is not None:
                    new_valid_mask[b, new_idx] = 1
                if latent is not None:
                    new_latent[b, new_idx] = latent[b, orig_idx]
                if nonterminal is not None:
                    new_nonterminal[b, new_idx] = nonterminal[b, orig_idx]

    return new_video, new_cond, new_latent, new_nonterminal, new_valid_mask


def rotation_angle(
    R1: Float[Tensor, "... t i j"],
    R2: Float[Tensor, "... t i j"],
    eps: float = 1e-8,
) -> Float[Tensor, "... t"]:
    # compute the axis-angle representation of the relative rotation between two rotation matrices (in degrees)
    R_rel = rearrange(R1, "... i j -> ... j i") @ R2
    rotation_angle_rad = torch.acos(
        torch.clip(
            (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2, -1.0 + eps, 1.0 - eps
        )
    )

    return rotation_angle_rad * 180.0 / math.pi


def dist_and_angle_between_cams(
    extrinsics1: Float[Tensor, "... t i j"],
    extrinsics2: Float[Tensor, "... t i j"],
) -> Tuple[Float[Tensor, "... t"], Float[Tensor, "... t"]]:
    """
    Compute the distance and angle between two sets of cameras, represented by their extrinsics.
    """

    R1, t1 = extrinsics1[..., :3, :3], extrinsics1[..., :3, 3:4]
    R2, t2 = extrinsics2[..., :3, :3], extrinsics2[..., :3, 3:4]

    # world-to-camera -> camera-to-world
    t1 = -rearrange(R1, "... i j -> ... j i") @ t1
    t2 = -rearrange(R2, "... i j -> ... j i") @ t2
    R1 = rearrange(R1, "... i j -> ... j i")
    R2 = rearrange(R2, "... i j -> ... j i")

    # shape = (..., T)
    translation_dist = torch.norm(t1[..., 0] - t2[..., 0], dim=-1)
    rotation_dist = rotation_angle(R1, R2)

    return translation_dist, rotation_dist


def test_identical_cameras_fov_distance():
    """
    Test that identical cameras have FOV distance of 0.
    """
    # Create a simple camera pose
    # Camera at (0,0,0) looking along positive z-axis
    R_w2c = torch.eye(3)  # Identity rotation (camera aligned with world)
    T_w2c = torch.zeros(3)  # Camera at origin

    # Create extrinsics matrix
    extrinsics = torch.zeros(3, 4)
    extrinsics[:3, :3] = R_w2c
    extrinsics[:3, 3] = T_w2c

    # Create intrinsics matrix K
    # Using normalized coordinates (0-1)
    fx = fy = 0.5  # 90-degree FOV
    cx = cy = 0.5  # Center of image

    # Create FOVDistance instance
    fov_distance = FOVDistance(frustum_length=2.0)

    # Create identical query and candidate conditions
    # Shape: [B, h, 16] where 16 is 4 intrinsics + 12 extrinsics
    # Flatten K and extrinsics to match the expected format
    query_conditions = (
        torch.cat([torch.tensor([fx, fy, cx, cy]), extrinsics.flatten()])
        .unsqueeze(0)
        .unsqueeze(0)
    )  # [1, 1, 16]
    candidate_conditions = query_conditions.clone()  # [1, 1, 16]

    # Compute distances
    distances, _ = fov_distance(query_conditions, candidate_conditions)

    # Check that distance is 0 (or very close to 0 due to numerical precision)
    assert torch.allclose(
        distances, torch.zeros_like(distances), atol=1e-5
    ), f"Expected distance 0 between identical cameras, got {distances}"

    print("Test passed: Identical cameras have FOV distance of 0")


if __name__ == "__main__":
    # Test code for comparing FOVDistance with RotationDistance for panorama trajectory
    import numpy as np
    import matplotlib.pyplot as plt

    def create_panorama_trajectory(
        num_frames=36, radius=5.0, center=np.array([0.0, 0.0, 0.0])
    ):
        """
        Create a panorama rotation-only trajectory (camera rotating in place)
        """
        # Camera positions all at the same point
        positions = np.array([center] * num_frames)

        # Rotations around the y-axis (full 360 degrees)
        angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

        # Create poses with rotation
        camera_poses = []
        intrinsics = []

        for i, angle in enumerate(angles):
            # Rotation matrix for rotation around y-axis
            R = np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )

            # Convert to camera-to-world (invert)
            R_cw = R.T
            t_cw = -R_cw @ positions[i]

            # Create 3x4 extrinsics matrix
            extrinsic = np.zeros((3, 4))
            extrinsic[:3, :3] = R_cw
            extrinsic[:3, 3] = t_cw

            # Add to the list
            camera_poses.append(extrinsic.flatten())

            # Create intrinsics (fx, fy, cx, cy)
            # Using typical realistic camera values
            # Set a narrower field of view by using a larger focal length
            # For a 60-degree horizontal FOV in normalized coordinates, fx should be ~0.87
            fx = fy = 0.87  # Normalized focal length (60° FOV)
            cx = cy = 0.5  # Center of the image
            intrinsics.append([fx, fy, cx, cy])

        # Convert to tensors
        camera_poses = torch.tensor(camera_poses, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        # Combine intrinsics and camera poses
        combined = torch.cat([intrinsics, camera_poses], dim=1)

        return combined

    def test_panorama_distances():
        """
        Test distance metrics on a panorama trajectory
        """
        print("Testing distance metrics on panorama trajectory...")

        # Create panorama trajectory
        num_frames = 36  # One frame every 10 degrees
        trajectory = create_panorama_trajectory(num_frames)

        # Add batch dimension
        trajectory = trajectory.unsqueeze(0)  # [1, num_frames, 16]

        # Select a reference frame (in the middle)
        reference_idx = np.random.randint(0, num_frames)
        reference = trajectory[:, reference_idx : reference_idx + 1, :]  # [1, 1, 16]

        # Initialize distance metrics
        # Use a reasonable frustum_length that will create visible differences in overlap
        fov_distance = FOVDistance(frustum_length=2.0, max_distance_threshold=0.7)
        rotation_distance = RotationDistance()

        # Debug: Directly examine the frustum calculations for a few frames
        print("\nDebugging frustum calculations and overlap:")

        # Convert to camera poses
        query_poses = CameraPose.from_vectors(reference)
        candidate_poses = CameraPose.from_vectors(trajectory)

        # Compute frustum corners for a subset of frames
        query_corners = fov_distance._compute_frustum_corners(
            query_poses
        )  # [1, 1, 4, 3]

        # Select frames to compare (reference, reference±30°, reference±60°, reference±90°, reference±180°)
        relative_angles = []

        for angle in relative_angles:
            # Convert angle to frame index (10 degrees per frame)
            offset = angle // 10
            frame_idx = (reference_idx + offset) % num_frames
            opposite_idx = (reference_idx - offset) % num_frames

            # Get the corners for these frames
            frame_corners = fov_distance._compute_frustum_corners(
                CameraPose.from_vectors(trajectory[:, frame_idx : frame_idx + 1, :])
            )  # [1, 1, 4, 3]

            opposite_corners = fov_distance._compute_frustum_corners(
                CameraPose.from_vectors(
                    trajectory[:, opposite_idx : opposite_idx + 1, :]
                )
            )  # [1, 1, 4, 3]

            # Compare with reference
            overlap_forward = fov_distance._compute_overlap(
                query_corners.unsqueeze(1),  # [1, 1, 1, 4, 3]
                frame_corners.unsqueeze(2),  # [1, 1, 1, 4, 3]
            )  # [1, 1, 1]

            overlap_backward = fov_distance._compute_overlap(
                query_corners.unsqueeze(1),  # [1, 1, 1, 4, 3]
                opposite_corners.unsqueeze(2),  # [1, 1, 1, 4, 3]
            )  # [1, 1, 1]

            print(f"\nAngle: {angle}° from reference:")
            print(
                f"  Forward (frame {frame_idx}) - overlap: {overlap_forward.item():.4f}, distance: {1-overlap_forward.item():.4f}"
            )
            print(
                f"  Backward (frame {opposite_idx}) - overlap: {overlap_backward.item():.4f}, distance: {1-overlap_backward.item():.4f}"
            )

        # Compute distances for all frames
        print("Computing distances for all frames...")
        print(reference.shape, trajectory.shape)
        fov_dists, fov_mask = fov_distance(reference, trajectory)
        rot_dists, rot_mask = rotation_distance(reference, trajectory)

        # Remove batch dimension for plotting
        fov_dists = fov_dists.squeeze(0).cpu().numpy()
        rot_dists = rot_dists.squeeze(0).cpu().numpy()

        # Print results for all frames
        print(f"\nReference frame: {reference_idx}")
        print("\nDistances from reference frame to all frames:")
        print("Frame\tAng.Diff\tFOV Dist\tRot Dist (rad)")

        # Calculate angular differences for all frames
        angles = np.linspace(0, 360, num_frames, endpoint=False)
        ref_angle = angles[reference_idx]
        for i in range(num_frames):
            ang_diff = np.abs(angles[i] - ref_angle)
            ang_diff = min(ang_diff, 360 - ang_diff)  # Get the smallest angle
            print(f"{i}\t{ang_diff:>7.1f}°\t{fov_dists[i]:.4f}\t{rot_dists[i]:.4f}")

        # Convert rotation angles to degrees for plotting
        relative_angles = np.abs(angles - angles[reference_idx])
        relative_angles = np.minimum(
            relative_angles, 360 - relative_angles
        )  # Get the smallest angle

        # Sort by relative angle for clearer plotting
        sort_idx = np.argsort(relative_angles)
        sorted_angles = relative_angles[sort_idx]
        sorted_fov = fov_dists[sort_idx]
        sorted_rot = rot_dists[sort_idx]

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_angles, sorted_fov, "bo-", label="FOV Distance")
        plt.plot(
            sorted_angles,
            sorted_rot / np.pi,
            "ro--",
            label="Rotation Distance (normalized)",
        )
        plt.xlabel("Relative Angle (degrees)")
        plt.ylabel("Distance")
        plt.title("Distance Metrics vs. Relative Angle from Reference")
        plt.legend()
        plt.grid(True)
        plt.savefig("panorama_distance_vs_angle.png")
        print("\nPlot saved as 'panorama_distance_vs_angle.png'")

        # Add a second plot showing distances vs. absolute frame numbers
        plt.figure(figsize=(10, 6))
        plt.plot(angles, fov_dists, "b-", label="FOV Distance")
        plt.plot(
            angles, rot_dists / np.pi, "r--", label="Rotation Distance (normalized)"
        )
        plt.axvline(
            x=angles[reference_idx], color="g", linestyle=":", label="Reference Frame"
        )
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Distance")
        plt.title("Distance Metrics for Panorama Trajectory")
        plt.legend()
        plt.grid(True)
        plt.savefig("panorama_distances.png")

        # Check which frames are considered similar by FOV
        if fov_mask is not None:
            fov_mask = fov_mask.squeeze(0).cpu().numpy()
            similar_frames = np.where(fov_mask > 0)[0]
            similar_angles = angles[similar_frames]
            ref_angle = angles[reference_idx]
            angle_diffs = np.abs(similar_angles - ref_angle)
            angle_diffs = np.minimum(
                angle_diffs, 360 - angle_diffs
            )  # Get the smallest angle
            max_angle_diff = angle_diffs.max() if len(angle_diffs) > 0 else 0

            print(
                f"\nFrames considered similar to reference by FOV (threshold={fov_distance.max_distance_threshold}):"
            )
            print(similar_frames)
            print(
                f"These frames span an angle range of {max_angle_diff:.1f}° relative to the reference"
            )

            # Sort by angular difference
            sort_idx = np.argsort(angle_diffs)
            sorted_similar_frames = similar_frames[sort_idx]
            sorted_angle_diffs = angle_diffs[sort_idx]

            # Print in a neat table
            print("\nSimilar frames sorted by angular difference:")
            print("Frame\tAng.Diff\tFOV Dist")
            for frame_idx, angle_diff in zip(sorted_similar_frames, sorted_angle_diffs):
                print(f"{frame_idx}\t{angle_diff:>7.1f}°\t{fov_dists[frame_idx]:.4f}")

            effective_fov = 2 * max_angle_diff if len(angle_diffs) > 0 else 0
            print(f"\nThis corresponds to an effective FOV of {effective_fov:.1f}°")

        # Add detailed debugging info for the frustum computation
        # Get a sample camera pose
        sample_pose = CameraPose.from_vectors(reference)

        # Get intrinsics
        intrinsics = sample_pose._K[0, 0]  # [fx, fy, cx, cy]
        print("\nCamera Intrinsics:")
        print(f"Focal length (fx, fy): {intrinsics[0]:.4f}, {intrinsics[1]:.4f}")
        print(f"Principal point (cx, cy): {intrinsics[2]:.4f}, {intrinsics[3]:.4f}")

        # Calculate theoretical horizontal FOV from intrinsics
        hfov_rad = 2 * np.arctan(0.5 / intrinsics[0].item())
        hfov_deg = hfov_rad * 180 / np.pi
        print(f"Theoretical horizontal FOV from intrinsics: {hfov_deg:.1f}°")

        # Also verify the range of FOV distances
        print(
            f"\nRange of FOV distances: [{fov_dists.min():.4f}, {fov_dists.max():.4f}]"
        )
        print(
            f"Is any FOV distance exactly 1? {'Yes' if np.any(fov_dists == 1.0) else 'No'}"
        )

        # Visualize frustum corners in 3D
        try:
            from mpl_toolkits.mplot3d import Axes3D

            # Sample a subset of frames for visualization
            viz_indices = [reference_idx]
            viz_indices.extend(
                [(reference_idx + i) % num_frames for i in range(4, 37, 8)]
            )

            # Calculate frustum corners for these frames
            viz_corners = []
            for idx in viz_indices:
                frame_corners = (
                    fov_distance._compute_frustum_corners(
                        CameraPose.from_vectors(trajectory[:, idx : idx + 1, :])
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )  # [4, 3]

                # Add the camera position as well
                cam_pos = candidate_poses.extrinsics()[0, idx, :3, 3].cpu().numpy()

                viz_corners.append((idx, frame_corners, cam_pos))

            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Plot each frustum
            colors = ["r", "g", "b", "c", "m", "y", "k"]
            for i, (idx, corners, cam_pos) in enumerate(viz_corners):
                color = colors[i % len(colors)]

                # Plot camera position
                ax.scatter(
                    cam_pos[0],
                    cam_pos[1],
                    cam_pos[2],
                    c=color,
                    marker="o",
                    s=100,
                    label=f"Frame {idx}",
                )

                # Plot frustum corners
                ax.scatter(
                    corners[:, 0], corners[:, 1], corners[:, 2], c=color, marker="^"
                )

                # Connect corners to form frustum
                # Connect corners to each other
                for j in range(4):
                    next_j = (j + 1) % 4
                    ax.plot(
                        [corners[j, 0], corners[next_j, 0]],
                        [corners[j, 1], corners[next_j, 1]],
                        [corners[j, 2], corners[next_j, 2]],
                        c=color,
                    )

                # Connect camera to corners
                for j in range(4):
                    ax.plot(
                        [cam_pos[0], corners[j, 0]],
                        [cam_pos[1], corners[j, 1]],
                        [cam_pos[2], corners[j, 2]],
                        c=color,
                        linestyle="--",
                    )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Camera Frustums for Selected Frames")
            plt.legend()
            plt.tight_layout()
            plt.savefig("camera_frustums_3d.png")
            print("\nFrustum visualization saved as 'camera_frustums_3d.png'")

        except ImportError:
            print("\nCould not create 3D visualization (mpl_toolkits not available)")

        print("\nTest completed successfully!")

    # Run the test
    test_panorama_distances()
    test_identical_cameras_fov_distance()

