from typing import Optional, List
from jaxtyping import Float
import wandb
import numpy as np
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from PIL import Image
from pathlib import Path
import imageio
import cv2

plt.set_loglevel("warning")

from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
)
from einops import rearrange, repeat
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)
from typing import Tuple
from utils import geometry_utils


# FIXME: clean up & check this util
def log_video(
    observation_hats: List[torch.Tensor] | torch.Tensor,
    observation_gt: Optional[torch.Tensor] = None,
    miscellany: Optional[List[torch.Tensor]] = None,
    step=0,
    namespace="train",
    prefix="video",
    postfix=[],
    captions=[],
    indent=0,
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
    n_frames=None,
    raw_dir=None,
    fps=24,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_gt: ground-truth observation tensor of shape (batch, frame, channel, height, width)
    :param observation_hats: list of predicted observation tensor of shape (batch, frame, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param postfix: a list of strings specify postfixes for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    if not logger:
        logger = wandb
    if isinstance(observation_hats, torch.Tensor):
        observation_hats = [observation_hats]
    if observation_gt is None:
        observation_gt_is_none = True
        observation_gt = torch.zeros_like(observation_hats[0])
    else:
        observation_gt_is_none = False
    observation_gt = observation_gt.type_as(observation_hats[0])

    if isinstance(context_frames, int):
        context_frames = torch.arange(context_frames, device=observation_gt.device)
    for observation_hat in observation_hats:
        observation_hat[:, context_frames] = observation_gt[:, context_frames]
        # observation_hat[:, -context_frames-1] = observation_gt[:, -context_frames-1]

    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        observation_gt_np, observation_hat_np = map(
            lambda x: (
                np.clip(x.detach().cpu().numpy(), a_min=0.0, a_max=1.0) * 255
            ).astype(np.uint8),
            (observation_gt, observation_hats[0]),
        )

        for i, (gt, hat) in enumerate(zip(observation_gt_np, observation_hat_np)):
            (raw_dir / f"{i + indent}").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(raw_dir / f"{i + indent}/data.npz", gt=gt, gen=hat)

            frames = [np.transpose(frame, (1, 2, 0)) for frame in hat]
            imageio.mimwrite(
                (raw_dir / f"{i + indent}") / "gen_preview.mp4",
                frames,
                fps=20,
                macro_block_size=None,
            )

    # Add red border of 1 pixel width to the context frames
    context_frames, indices = torch.meshgrid(
        context_frames,
        torch.tensor([0, -1], device=observation_gt.device, dtype=torch.long),
        indexing="ij",
    )
    for i, c in enumerate(color):
        c = c / 255.0
        for observation_hat in observation_hats:
            observation_hat[:, context_frames, i, indices, :] = c
            observation_hat[:, context_frames, i, :, indices] = c
            # observation_hat[:, -context_frames-1, i, indices, :] = c
            # observation_hat[:, -context_frames-1, i, :, indices] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    if miscellany is not None:
        video = (
            torch.cat([*observation_hats, observation_gt, *miscellany], -1)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        video = (
            torch.cat([*observation_hats, observation_gt], -1).detach().cpu().numpy()
        )
    if miscellany is not None:
        multiview_grid = (
            torch.cat([*observation_hats, observation_gt, *miscellany], -2)
            .detach()
            .cpu()
            .numpy()
        )
    else:
        multiview_grid = (
            torch.cat([*observation_hats, observation_gt], -2).detach().cpu().numpy()
        )

    # reshape to original shape
    if n_frames is not None:
        video = rearrange(
            video, "(b n) t c h w -> b (n t) c h w", n=n_frames // video.shape[1]
        )
        multiview_grid = rearrange(
            multiview_grid,
            "(b n) t c h w -> b (n t) c h w",
            n=n_frames // video.shape[1],
        )

    video = (np.clip(video, a_min=0.0, a_max=1.0) * 255).astype(np.uint8)
    multiview_grid = (np.clip(multiview_grid, a_min=0.0, a_max=1.0) * 255).astype(
        np.uint8
    )

    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    if isinstance(captions, str):
        captions = [captions] * n_samples
    for i in range(n_samples):
        name = f"{namespace}/{prefix}_{i + indent}" + (
            f"_{postfix[i]}" if i < len(postfix) else ""
        )
        caption = captions[i] if i < len(captions) else None
        logger.log(
            {
                name: wandb.Video(video[i], fps=1, caption=caption),
                "trainer/global_step": step,
            }
        )


def log_spatial_neighbors(
    spatial_neighbors: Float[torch.Tensor, "num_frame_pairs 2"],
    xs_pred: Float[torch.Tensor, "T C H W"],
    step=0,
    namespace="train",
    prefix="spatial_neighbors",
    logger=None,
):
    if not logger:
        logger = wandb

    def label_video_frames_float(
        frames,
        circle_color=(0, 0, 255),  # red in BGR
        circle_radius=22,
        circle_thickness=2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.6,
        font_thickness=2,
        corner_offset=(30, 30),
    ):
        """
        frames: float32/float64 ndarray with shape (T, C, H, W) and values in [0, 1]
        returns: same shape & dtype, still in [0, 1], with frame indices overlaid
        """
        if frames.dtype not in (np.float32, np.float64):
            raise TypeError("Input must be float32/float64 in [0, 1].")

        T, C, H, W = frames.shape
        # ---- 1. Re‑arrange to (T, H, W, C) and make a contiguous copy ----
        rgb = frames.transpose(0, 2, 3, 1).copy()  # keep float [0,1]

        # ---- 2. Convert once to uint8 for fast drawing ----
        draw = (rgb * 255).astype(np.uint8)  # contiguous

        cx, cy = corner_offset
        for idx in range(T):
            img = draw[idx]

            # Circle
            cv2.circle(
                img,
                (cx, cy),
                circle_radius,
                circle_color,
                thickness=circle_thickness,
                lineType=cv2.LINE_AA,
            )

            # Text centred in the circle
            text = str(idx)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            origin = (cx - tw // 2, cy + th // 2)
            cv2.putText(
                img,
                text,
                origin,
                font,
                font_scale,
                circle_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        # ---- 3. Back to float [0,1] in the original layout ----
        labelled = draw.astype(np.float32) / 255.0
        return labelled.transpose(0, 3, 1, 2)

    # label xs_pred with its frame index in the top left corner
    spatial_neighbors = spatial_neighbors.cpu().numpy()
    xs_pred = xs_pred.cpu().numpy()
    xs_pred = label_video_frames_float(xs_pred)

    # use spatial_neighbors to index into xs_pred and return a tensor of shape (num_frame_pairs, 2, C, H, W)
    xs_pred_spatial_neighbors = xs_pred[spatial_neighbors]
    xs_pred_spatial_neighbors = rearrange(
        xs_pred_spatial_neighbors, "n p c h w -> (n h) (p w) c"
    )
    xs_pred_spatial_neighbors = np.clip(
        xs_pred_spatial_neighbors * 255.0, a_min=0.0, a_max=255.0
    ).astype(np.uint8)

    name = f"{namespace}/{prefix}"
    logger.log(
        {
            name: wandb.Image(
                xs_pred_spatial_neighbors,
                caption=f"Each row is a pair of spatial neighbors",
            ),
            "trainer/global_step": step,
        }
    )


# def get_camera_params(conditions: torch.Tensor, ref_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Args:
#         conditions: [B, T, 12]
#         ref_mask: [B, T]
#     Returns:
#         K: [B, T, 4];  [fx, fy, cx, cy]
#         R_rc: [B, T, 3, 3]
#         T_rc: [B, T, 3]
#     """
#     camera_poses = conditions.clone()
#     K, RT_w2c = camera_poses.split([4, 12], dim=-1)
#     RT_w2c = rearrange(RT_w2c, "b t (i j) -> b t i j", i=3, j=4)
#     R_w2c = RT_w2c[..., :3, :3]
#     T_w2c = RT_w2c[..., :3, 3]
#     R_c2w, T_c2w = geometry_utils.invert_extrinsics(R_w2c, T_w2c)
#     R_c2w, T_c2w = geometry_utils.rebase_reference_frame(R_c2w, T_c2w)
#     return K, R_c2w, T_c2w


def visualize_conditions(
    conditions: torch.Tensor, plot_trajectory: bool = True
) -> torch.Tensor:
    """
    Visualize the conditions (camera poses) for a video.
    Args:
        conditions: Tensor of shape [B, T, 16] containing camera poses
                   where each pose is [intrinsic, rotation, translation] flattened
    Returns:
        Tensor of shape [B, T, 3, H, W] containing the visualization video
    """
    B, T = conditions.shape[:2]
    H, W = 256, 256  # Output image size
    # K, R_wc, T_wc = get_camera_params(conditions)

    camera_poses = geometry_utils.CameraPose.from_vectors(conditions)
    camera_poses.normalize_by_first()
    camera_poses.scale_within_bounds(1.0)
    K = camera_poses.intrinsics(flatten=False)
    R_w2c = camera_poses.extrinsics()[:, :, :3, :3]
    T_w2c = camera_poses.extrinsics()[:, :, :3, 3]
    R_c2w, T_c2w = geometry_utils.invert_extrinsics(R_w2c, T_w2c)

    # Get camera frustums
    frustum_trajectories = get_frustums(K, R_c2w, T_c2w, uv_range=[0, 1])

    # Create visualization videos
    videos = []
    for b in range(B):
        video = create_camera_path_video(
            frustums=frustum_trajectories[b],
            R_wc=R_c2w[b],
            T_wc=T_c2w[b],
            H=H,
            W=W,
            dpi=120,
            colors=None,
            plot_trajectory=plot_trajectory,
        )
        videos.append(video)

    return (
        torch.stack(videos, dim=0).to(conditions.device).float() / 255.0
    )  # shape: [B, T, 3, H, W]


@torch.no_grad()
def get_frustums(
    K: torch.Tensor, R_rc: torch.Tensor, T_rc: torch.Tensor, uv_range=[0, 1]
) -> torch.Tensor:
    """
    Args:
        K: [B, T, 3, 3];  [fx, fy, cx, cy]
        R_rc: [B, T, 3, 3]
        T_rc: [B, T, 3]
    Returns:
        frustum: [B, T, 8, 3]
    """
    frustum, frustum_width, frustum_height, frustum_center = geometry_utils.get_frustum(
        K, uv_range=uv_range, frustum_scale=0.1, center_ray_mult=5.0
    )
    frustum = (
        torch.einsum("...ij,...elj->...eli", R_rc, frustum) + T_rc[..., None, None, :]
    )
    return frustum


def create_camera_path_video(
    frustums,  # [T, 9, 2, 3], each: 9 edges (8 edges + 1 center ray), start/end points in world coords
    R_wc,  # [T, 3, 3], not used (already in world coords), kept just for shape check
    T_wc,  # [T, 3],    camera centers in world coords, for optional trajectory lines
    H=256,  # output image height
    W=256,  # output image width
    dpi=120,
    colors=None,  # None or [T, 3] in float RGB (0..1)
    plot_trajectory=True,
) -> torch.Tensor:
    """
    Returns a video of shape [T, 3, H, W] showing the camera path being revealed over time.
    Each frame shows all cameras up to the current time step.
    """
    if isinstance(frustums, torch.Tensor):
        frustums = frustums.detach().cpu().numpy()
    if isinstance(R_wc, torch.Tensor):
        R_wc = R_wc.detach().cpu().numpy()
    if isinstance(T_wc, torch.Tensor):
        T_wc = T_wc.detach().cpu().numpy()

    R_align = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ]
    )
    R_wc = np.einsum("ij,...jk->...ik", R_align, R_wc)
    T_wc = np.einsum("ij,...j->...i", R_align, T_wc)
    frustums = np.einsum("ij,...j->...i", R_align, frustums)

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.cm as cm

    # Total number of cameras
    T = frustums.shape[0]

    # Use viridis colormap if colors not provided
    if colors is None:
        colors = cm.get_cmap("viridis")(np.linspace(0, 1, T))[
            :, :3
        ]  # shape: [T, 3] in 0..1

    # Create output video array: (T, 3, H, W)
    video = np.zeros((T, 3, H, W), dtype=np.uint8)

    # Create a figure and canvas for rendering
    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0.1, right=1, top=1)  # Trim figure margins
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    # Pre-calc view limits using all points (frustums and camera centers)
    all_points = frustums.reshape(-1, 3)
    all_points = np.vstack([all_points, T_wc])
    min_xyz = all_points.min(axis=0)
    max_xyz = all_points.max(axis=0)
    center = (min_xyz + max_xyz) / 2
    size = max(max_xyz - min_xyz)

    # For each time step, render the scene with only cameras up to that time visible
    for t in range(T):
        ax.clear()

        # Plot cameras up to current time
        for i in range(t + 1):
            color = colors[i]
            # Plot each frustum edge with the color from viridis (or provided colors)
            edges = frustums[i]  # shape: [9, 2, 3]
            for edge in edges:
                start, end = edge
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=color,
                    alpha=0.5,
                )
            if plot_trajectory:
                # Plot camera center path connecting consecutive cameras using same color
                if i > 0:
                    ax.plot(
                        [T_wc[i - 1, 0], T_wc[i, 0]],
                        [T_wc[i - 1, 1], T_wc[i, 1]],
                        [T_wc[i - 1, 2], T_wc[i, 2]],
                        color=color,
                        alpha=0.5,
                    )

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        # Set consistent view limits
        ax.set_xlim(center[0] - size / 2, center[0] + size / 2)
        ax.set_ylim(center[1] - size / 2, center[1] + size / 2)
        ax.set_zlim(center[2] - size / 2, center[2] + size / 2)

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("--Y")

        # Render to image from canvas (extract RGBA, then drop alpha)
        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(H, W, 4)[..., :3]

        # Convert from HWC to CHW and store in video array
        video[t] = np.transpose(img, (2, 0, 1))

    plt.close(fig)
    return torch.from_numpy(video)


def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    batch, frame, channel, height, width = observation_hat.shape
    output_dict = {}
    # some metrics don't fully support fp16
    if observation_hat.dtype == torch.float16:
        observation_hat = observation_hat.to(torch.float32)
    if observation_gt.dtype == torch.float16:
        observation_gt = observation_gt.to(torch.float32)

    # reshape to (batch * frame, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(
        observation_hat, observation_gt, data_range=2.0
    )
    output_dict["ssim"] = structural_similarity_index_measure(
        observation_hat, observation_gt, data_range=2.0
    )
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


def is_grid_env(env_id):
    return "maze2d" in env_id or "diagonal2d" in env_id


def get_maze_grid(env_id):
    # import gym
    # maze_string = gym.make(env_id).str_maze_spec
    if "large" in env_id:
        maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    if "medium" in env_id:
        maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    if "umaze" in env_id:
        maze_string = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def get_random_start_goal(env_id, batch_size):
    maze_grid = get_maze_grid(env_id)
    s2i = {"O": 0, "#": 1, "G": 2}
    maze_grid = [[s2i[s] for s in r] for r in maze_grid]
    maze_grid = np.array(maze_grid)
    x, y = np.nonzero(maze_grid == 0)
    indices = np.random.randint(len(x), size=batch_size)
    start = np.stack([x[indices], y[indices]], -1) + 1
    x, y = np.nonzero(maze_grid == 2)
    goal = np.concatenate([x, y], -1)
    goal = np.tile(goal[None, :], (batch_size, 1)) + 1
    return start, goal


def plot_maze_layout(ax, maze_grid):
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle(
                        (i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black"
                    )
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def plot_start_goal(ax, start_goal: None):
    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (
            2 * num_points
        )
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle(
        (start_x, start_y), 0.16, facecolor="white", edgecolor="black"
    )
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle(
        (goal_x, goal_y), 0.16, facecolor="white", edgecolor="black"
    )
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)


def make_trajectory_images(
    env_id, trajectory, batch_size, start, goal, plot_end_points=True
):
    images = []
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        if is_grid_env(env_id):
            maze_grid = get_maze_grid(env_id)
        else:
            maze_grid = None
        plot_maze_layout(ax, maze_grid)
        ax.scatter(
            trajectory[batch_idx, :, 0],
            trajectory[batch_idx, :, 1],
            c=np.arange(trajectory.shape[1]),
            cmap="Reds",
        )
        if plot_end_points:
            start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)
        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = (
            np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            .copy()
            .reshape(img_shape)
        )
        images.append(img)

        plt.close()
    return images


def make_convergence_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(
        plan_history, trajectory, goal, open_loop_horizon
    )

    # animate the convergence of the first plan
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[0][frame]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    frames = tqdm(range(len(plan_history[0])), desc="Making convergence animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_convergence.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)
    return filename


def prune_history(plan_history, trajectory, goal, open_loop_horizon):
    dist = np.linalg.norm(
        trajectory[:, :2] - np.array(goal)[None],
        axis=-1,
    )
    reached = dist < 0.2
    if reached.any():
        cap_idx = np.argmax(reached)
        trajectory = trajectory[: cap_idx + open_loop_horizon + 1]
        plan_history = plan_history[: cap_idx // open_loop_horizon + 2]

    pruned_plan_history = []
    for plans in plan_history:
        pruned_plan_history.append([])
        for m in range(len(plans)):
            plan = plans[m]
            pruned_plan_history[-1].append(plan)
        plan = pruned_plan_history[-1][-1]
        dist = np.linalg.norm(plan.numpy()[:, :2] - np.array(goal)[None], axis=-1)
        reached = dist < 0.2
        if reached.any():
            cap_idx = np.argmax(reached) + 1
            pruned_plan_history[-1] = [p[:cap_idx] for p in pruned_plan_history[-1]]
    return trajectory, pruned_plan_history


def make_mpc_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(
        plan_history, trajectory, goal, open_loop_horizon
    )

    # animate the convergence of the plans
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    trajectory_colors = np.linspace(0, 1, len(trajectory))

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        control_time_step = 0
        while frame >= 0:
            frame -= len(plan_history[control_time_step])
            control_time_step += 1
        control_time_step -= 1
        m = frame + len(plan_history[control_time_step])
        num_steps_taken = 1 + open_loop_horizon * control_time_step
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[control_time_step][m]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            trajectory[:num_steps_taken, 0],
            trajectory[:num_steps_taken, 1],
            c=trajectory_colors[:num_steps_taken],
            cmap="Blues",
        )
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    num_frames = sum([len(p) for p in plan_history])
    frames = tqdm(range(num_frames), desc="Making MPC animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_mpc.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)

    return filename
