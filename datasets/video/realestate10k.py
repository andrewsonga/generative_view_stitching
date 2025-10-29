from typing import Tuple, List, Dict, Any, Literal, Optional
from jaxtyping import Float, Int64, Bool
from fractions import Fraction
from pathlib import Path
import random
import math
import io
from multiprocessing import Pool
import subprocess
from functools import partial
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.io import write_video
from torchvision.datasets.utils import (
    download_and_extract_archive,
)
import numpy as np
from pytubefix import YouTube
from utils.geometry_utils import CameraPose
from utils.print_utils import cyan
from utils.storage_utils import safe_torch_save
from utils.retrieval_utils import dist_and_angle_between_cams, impute_support_frames
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from .utils import read_video, rescale_and_crop, random_bool

VideoPreprocessingType = Literal["npz", "mp4"]
VideoPreprocessingMp4FPS: int = 10
DownloadPlan = Dict[str, Dict[str, List[float]]]


class RealEstate10KBaseVideoDataset(BaseVideoDataset):
    """
    RealEstate10K base video dataset.
    The dataset will be preprocessed to `_SUPPORTED_RESOLUTIONS` in the format of `_SUPPORTED_RESOLUTIONS[resolution]` during the download.
    """

    _ALL_SPLITS = ["training", "test"]
    # this originally comes from https://storage.cloud.google.com/realestate10k-public-files/RealEstate10K.tar.gz
    # but is served at the above URL to avoid the need for manual download + place in the right directory
    _DATASET_URL = "https://huggingface.co/kiwhansong/DFoT/resolve/main/datasets/RealEstate10K.tar.gz"
    _SUPPORTED_RESOLUTIONS: Dict[int, VideoPreprocessingType] = {
        64: "npz",
        256: "mp4",
    }

    def _should_download(self) -> bool:
        if self.resolution not in self._SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Resolution {self.resolution} is not supported. Supported resolutions: {list(self._SUPPORTED_RESOLUTIONS.keys())}. Please modify `_SUPPORTED_RESOLUTIONS` in the RealEstate10kBaseVideoDataset class to support this resolution."
            )

        return (
            super()._should_download()
            and not (self.save_dir / f"{self.split}_{self.resolution}").exists()
        )

    def download_dataset(self) -> None:
        print(cyan("Downloading RealEstate10k dataset..."))
        print(
            cyan(
                "Please read the NOTE in the `_download_videos` function at `datasets/video/realestate10k_video_dataset.py` before continuing."
            )
        )
        input("Press Enter to continue...")
        download_and_extract_archive(
            self._DATASET_URL,
            self.save_dir,
            filename="raw.tar.gz",
            remove_finished=True,
        )
        (self.save_dir / "RealEstate10K").rename(self.save_dir / "raw")
        (self.save_dir / "raw" / "train").rename(self.save_dir / "raw" / "training")

        for split in ["training", "test"]:
            plan = self._build_download_plan(split)
            self._download_videos(split, plan)
            self._preprocess_videos(split, plan)

        print(cyan("Finished downloading RealEstate10k dataset!"))

    def _download_videos(self, split: SPLIT, urls: List[str]) -> None:
        """
        NOTE: The RealEstate10k dataset is a collection of YouTube videos, and downloading them should be done with caution to ensure that the dataset is not lost.

        This function may fail due to the following reasons:
        - The video is not available anymore, deleted, or private on YouTube. In this case, you shall ignore the error and continue.
        - Bot detection from YouTube. You may meet the error "This request was detected as a bot. Use `use_po_token=True` to view. See more details at https://github.com/JuanBindez/pytubefix/pull/209". When this happens, you will miss all videos afterward, so you should try to resolve this by using `use_po_token=True` or `use_oauth=True`, or by using a proxy, or by giving a time delay between each download.

        The exact size of the RealEstate10k dataset may change across time as it relies on the availability of YouTube videos, so we provide the following statistics as of the time of our own download (You should expect a similar but not identical size):
        - training: 6132 / 6559 videos = 65798 / 71556 clips -> 65725 clips after preprocessing
        - test: 655 / 696 videos = 7148 / 7711 clips

        The camera poses are w2c in OpenCV convetions 

        """
        print(cyan(f"Downloading {split} videos from YouTube..."))
        download_dir = self.save_dir / "raw" / split
        download_dir.mkdir(parents=True, exist_ok=True)
        download_fn = partial(_download_youtube_video, download_dir=download_dir)
        with Pool(32) as pool:
            list(
                tqdm(
                    pool.imap(download_fn, urls),
                    total=len(urls),
                    desc=f"Downloading {split} videos",
                )
            )

    def _preprocess_videos(self, split: SPLIT, plan: DownloadPlan) -> None:
        print(
            cyan(
                f"Preprocessing {split} videos to resolutions {list(self._SUPPORTED_RESOLUTIONS.keys())}..."
            )
        )
        args = []
        for youtube_url, key_to_timestamps in plan.items():
            video_path = (
                self.save_dir / "raw" / split / f"{_youtube_url_to_id(youtube_url)}.mp4"
            )
            if not video_path.exists():
                continue
            for key, timestamps in key_to_timestamps.items():
                args.append((key, video_path, timestamps))
        preprocess_fn = partial(
            _preprocess_video,
            resolutions_to_preprocessing=self._SUPPORTED_RESOLUTIONS,
        )
        with Pool(32) as pool:
            list(
                tqdm(
                    pool.imap(preprocess_fn, args),
                    total=len(args),
                    desc=f"Preprocessing {split} videos",
                )
            )

    def _build_download_plan(self, split: SPLIT) -> DownloadPlan:
        """
        Builds a download plan for the specified split.
        Returns a dictionary with the following structure:
            {
                youtube_url: {
                    key: timestamps,
                }
            }
        """
        print(cyan(f"Building download plan & camera poses for {split}..."))
        plan = {}
        txt_files = list((self.save_dir / "raw" / split).glob("*.txt"))
        for txt_file in tqdm(
            txt_files, desc=f"Building download plan & camera poses for {split}"
        ):
            youtube_url, timestamps, cameras = self._read_txt_file(txt_file)
            if youtube_url not in plan:
                plan[youtube_url] = {
                    txt_file.stem: timestamps,
                }
            else:
                plan[youtube_url][txt_file.stem] = timestamps
            safe_torch_save(
                cameras, self.save_dir / f"{split}_poses" / f"{txt_file.stem}.pt"
            )
        return plan

    @staticmethod
    def _read_txt_file(file_path: Path) -> Tuple[str, List[float], torch.Tensor]:
        """
        Reads a txt file containing a video path, a list of timestamps, and a tensor of camera poses.
        """
        timestamps = []
        cameras = []
        youtube_url = ""
        with open(file_path, "r") as f:
            lines = f.readlines()
            assert len(lines) > 0, f"Empty file {file_path}"
            for idx, line in enumerate(lines):
                if idx == 0:
                    youtube_url = line.strip()
                else:
                    timestamp, *camera = line.split(" ")
                    timestamps.append(_timestamp_to_str(int(timestamp)))
                    cameras.append(np.fromstring(",".join(camera), sep=","))
            cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

        return youtube_url, timestamps, cameras

    def build_metadata(self, split: SPLIT) -> None:
        super().build_metadata(f"{split}_256")
        (self.metadata_dir / f"{split}_256.pt").rename(
            self.metadata_dir / f"{split}.pt"
        )

    def setup(self) -> None:
        self.transform = lambda x: x

    def video_path_to_preprocessed_path(self, video_path: Path) -> Path:
        return (
            self.save_dir / f"{self.split}_{self.resolution}" / video_path.name
        ).with_suffix("." + self._SUPPORTED_RESOLUTIONS[self.resolution])

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        
        # added to prevent error when end_frame is None
        if end_frame is None:
            end_frame = self.video_length(video_metadata)

        preprocessed_path = self.video_path_to_preprocessed_path(
            video_metadata["video_paths"]
        )
        match preprocessed_path.suffix:
            case ".npz":
                video = np.load(
                    preprocessed_path,
                )[
                    "video"
                ][start_frame:end_frame]
                return torch.from_numpy(video / 255.0).float()
            case ".mp4":
                video = read_video(
                    preprocessed_path,
                    pts_unit="sec",
                    start_pts=Fraction(start_frame, VideoPreprocessingMp4FPS),
                    end_pts=Fraction(end_frame - 1, VideoPreprocessingMp4FPS),
                )   # THWC

                # THWC -> TCHW
                return video.permute(0, 3, 1, 2) / 255.0


class RealEstate10KSimpleVideoDataset(
    RealEstate10KBaseVideoDataset, BaseSimpleVideoDataset
):
    """
    RealEstate10K simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
        self.setup()
    
    def exclude_videos_with_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split + "_{}".format(self.resolution)))

        return self.subsample(
            metadata,
            lambda video_metadata: self.video_metadata_to_latent_path(video_metadata)
            not in latent_paths,
            "videos that have already been preprocessed to latents",
        )


class RealEstate10KAdvancedVideoDataset(
    RealEstate10KBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    RealEstate10K advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
        self.fix_intrinsics = cfg.fix_intrinsics
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    @property
    def _training_frame_skip(self) -> int:
        if self.augmentation.frame_skip_increase == 0:
            return self.frame_skip
        assert (
            self.current_subepoch is not None
        ), "Subepoch should be given to the RealEstate10KAdvancedVideoDataset, to use frame skip schedule"
        return self.frame_skip + int(
            self.current_subepoch * self.augmentation.frame_skip_increase
        )

    def exclude_videos_without_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        latent_paths = set(self.get_latent_paths(self.split + "_{}".format(self.resolution)))
        
        return self.subsample(
            metadata,
            lambda video_metadata: self.video_metadata_to_latent_path(video_metadata)
            in latent_paths,
            "videos without latents",
        )

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"]
        path = self.save_dir / f"{self.split}_poses" / f"{path.stem}.pt"
        cond = torch.load(path, weights_only=False)[start_frame:end_frame]
        return cond

    def _augment(
        self,
        video: torch.Tensor,
        cond: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # 1) Horizontal flip augmentation
        if random_bool(self.augmentation.horizontal_flip_prob):
            video = video.flip(-1)
            # NOTE: extrinsics should also be flipped accordingly - the following is equivalent to:
            # E' = I' @ E @ I' where I' = diag([-1, 1, 1, 1]) (E is 4x4 extrinsics matrix)
            cond[:, [5, 6, 7, 8, 12]] *= -1
            if latent is not None:
                latent = latent.flip(-1)

        # 2) Back-and-forth video augmentation
        # 0 1 2 ... 2k+1 -> 0 2 4 ... 2k 2k+1 ... 3 1
        if random_bool(self.augmentation.back_and_forth_prob):
            video, cond = map(
                lambda x: torch.cat([x[::2], x[1::2].flip(0)], dim=0).contiguous(),
                (video, cond),
            )
            if latent is not None:
                latent = torch.cat([latent[::2], latent[1::2].flip(0)], dim=0).contiguous()
            
        # 3) Reverse video augmentation
        # 0 ... n -> n ... 0
        if random_bool(self.augmentation.reverse_prob):
            video, cond = map(lambda x: x.flip(0).contiguous(), (video, cond))
            if latent is not None:
                latent = latent.flip(0).contiguous()

        return video, cond, latent

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split != "training":
            return super().__getitem__(idx)

        video_idx, start_frame = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        frame_skip = (video_length - start_frame - 1) // (self.cfg.max_frames - 1)
        if self.split == "training":
            frame_skip = min(frame_skip, self._training_frame_skip)
        else:
            frame_skip = random.randint(self.frame_skip, frame_skip)

        assert frame_skip > 0, f"Frame skip {frame_skip} should be greater than 0"
        end_frame = start_frame + (self.cfg.max_frames - 1) * frame_skip + 1

        # load video, cond, and latents if necessary
        video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)
        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, start_frame, end_frame)
        else:
            latent = None
        lens = [len(x) for x in (video, cond, latent) if x is not None]
        assert len(set(lens)) == 1, "video, cond, latent must have the same length"

        # skip frames
        video, cond = video[::frame_skip], self._process_external_cond(cond, frame_skip)
        if self.use_preprocessed_latents:
            latent = latent[::frame_skip]

        # augment
        video, cond, latent = self._augment(video, cond, latent)

        output = {
            "videos": self.transform(video),            
            "latents": latent,
            "conds": cond,
            "nonterminal": torch.ones(self.cfg.max_frames, dtype=torch.bool),
        }

        return {key: value for key, value in output.items() if value is not None}

    def exclude_short_videos(
        self, metadata: List[Dict[str, Any]], min_frames: int
    ) -> List[Dict[str, Any]]:
        # if self.maximize_training_data is True,
        # include all videos with at least self.cfg.max_frames frames
        if self.maximize_training_data and self.split == "training":
            min_frames = min(min_frames, self.cfg.max_frames)
        return super().exclude_short_videos(metadata, min_frames)

    def _process_external_cond(
        self, external_cond: torch.Tensor, frame_skip: Optional[int] = None
    ) -> torch.Tensor:
        """
        Converts the raw camera poses to concat-flattened intrinsics and extrinsics.
        Args:
            external_cond (torch.Tensor): Raw camera poses. Shape (T, 18).
            frame_skip (Optional[int]): Frame skip. If None, uses self.frame_skip.
        Returns:
            torch.Tensor: Processed camera poses. Shape (T, 16).
        """
        poses = external_cond[:: frame_skip or self.frame_skip]

        if self.fix_intrinsics:
            poses[:, 0] = torch.max(poses[:, 0], poses[:, 1])
            poses[:, 1] = torch.max(poses[:, 0], poses[:, 1])
        return torch.cat(
            [
                poses[:, :4],
                poses[:, 6:],
            ],
            dim=-1,
        ).to(torch.float32)

class RealEstate10KAdvancedVideoDatasetWithLoops(RealEstate10KAdvancedVideoDataset):
    
    """
    RealEstate10K advanced video dataset that only contains sequences with looped camera trajectories
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        
        self.impute_indices = cfg.rag.impute_indices 
        self.n_support_frames = len(self.impute_indices)
        assert self.n_support_frames == 1, "For now, we only support retrieval of a single support frame"
        self.retrieval_window_end_frame_rel = cfg.rag.retrieval_window_end_frame_rel        
        self.show_bev = cfg.get("show_bev", False)  
        super().__init__(cfg, split, current_epoch)

        if self.split == "training":
            # TODO: potentially remove first assertion
            assert self.augmentation.frame_skip_increase == 0, "frame_skip_increase must be 0 to use clip filtering logic in num_loop_closures"

        self.n_frames = (
            1
            + ((cfg.max_frames - self.n_support_frames if split == "training" else cfg.n_frames) - 1)
            * cfg.frame_skip
        )

        """
        cfg.rag.retrieval_window_end_frame_rel defines the end frame of the retrieval window relative to the loop closing frame:
        e.g. retrieval_window_end_frame (absolute) = loop_closing_frame - cfg.rag.retrieval_window_end_frame_rel
        if cfg.rag.retrieval_window_end_frame_rel = 0, the window is range(0, loop_closing_frame)
        if cfg.rag.retrieval_window_end_frame_rel = h, the window is range(0, loop_closing_frame - h)
        """

    def load_bev(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        
        # added to prevent error when end_frame is None
        if end_frame is None:
            end_frame = self.video_length(video_metadata)

        path = video_metadata["video_paths"]
        path = self.save_dir / f"{self.split}_bev_vis" / f"{path.stem}.mp4"
        match path.suffix:
            case ".npz":
                video = np.load(
                    path,
                )[
                    "video"
                ][start_frame:end_frame]
                return torch.from_numpy(video / 255.0).float()
            case ".mp4":
                video = read_video(
                    path,
                    pts_unit="sec",
                    start_pts=Fraction(start_frame, VideoPreprocessingMp4FPS),
                    end_pts=Fraction(end_frame - 1, VideoPreprocessingMp4FPS),
                )   # THWC

                # THWC -> TCHW
                return video.permute(0, 3, 1, 2) / 255.0

    
    def load_video_and_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video and conditions from video_idx with given start_frame and end_frame (exclusive)
        """

        video = self.load_video(video_metadata, start_frame, end_frame)
        cond = self.load_cond(video_metadata, start_frame, end_frame)

        return video, cond      

    def load_video_and_cond_and_bev(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load video and conditions and bev from video_idx with given start_frame and end_frame (exclusive)
        """

        video = self.load_video(video_metadata, start_frame, end_frame)
        cond = self.load_cond(video_metadata, start_frame, end_frame)
        bev = self.load_bev(video_metadata, start_frame, end_frame)

        # extract the portion of video to the right of the original video
        bev = bev[..., bev.shape[2]:]

        # resize bev to match height of video
        video_height = video.shape[2]
        bev = F.interpolate(bev, size=(video_height, int(bev.shape[3] * video_height / bev.shape[2])), mode="bilinear", align_corners=False)

        # pad width to 3 times the height of the video
        pad_width = int(video_height * 3 - bev.shape[3])
        if pad_width > 0:
            bev = F.pad(bev, (pad_width, 0, 0, 0, 0, 0, 0, 0), "constant", 0)
        else:   
            bev = bev[..., :video_height * 3]

        return video, cond, bev

    def _augment(
        self,
        video: torch.Tensor,
        cond: torch.Tensor,
        bev: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # 1) Horizontal flip augmentation
        if random_bool(self.augmentation.horizontal_flip_prob):
            video = video.flip(-1)
            # NOTE: extrinsics should also be flipped accordingly - the following is equivalent to:
            # E' = I' @ E @ I' where I' = diag([-1, 1, 1, 1]) (E is 4x4 extrinsics matrix)
            cond[:, [5, 6, 7, 8, 12]] *= -1
            if bev is not None:
                bev = bev.flip(-1)
            if latent is not None:
                latent = latent.flip(-1)

        # 2) Back-and-forth video augmentation
        # 0 1 2 ... 2k+1 -> 0 2 4 ... 2k 2k+1 ... 3 1
        if random_bool(self.augmentation.back_and_forth_prob):
            if bev is not None:
                video, cond, bev = map(
                    lambda x: torch.cat([x[::2], x[1::2].flip(0)], dim=0).contiguous(),
                    (video, cond, bev),
                )
            else:
                video, cond = map(
                    lambda x: torch.cat([x[::2], x[1::2].flip(0)], dim=0).contiguous(),
                    (video, cond),
                )
            if latent is not None:
                latent = torch.cat([latent[::2], latent[1::2].flip(0)], dim=0).contiguous()
            
        # 3) Reverse video augmentation
        # 0 ... n -> n ... 0
        if random_bool(self.augmentation.reverse_prob):
            if bev is not None:
                video, cond, bev = map(lambda x: x.flip(0).contiguous(), (video, cond, bev))
            else:
                video, cond = map(lambda x: x.flip(0).contiguous(), (video, cond))
            if latent is not None:
                latent = latent.flip(0).contiguous()

        return video, cond, bev, latent

    def exclude_videos_without_loops(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return self.subsample(
            metadata,
            lambda video_metadata: self.num_loop_closures(video_metadata) > 0,
            "videos without loops"
        )
    
    def num_loop_closures(
        self, video_metadata: Dict[str, Any]
    ) -> int:
        
        # shape = (T, 18)
        cond = self.load_cond(video_metadata, 0, self.video_length(video_metadata))

        # flattened world-to-camera i.e. extrinsics (R|t)
        # shape = (T, 12)
        cond = self._process_external_cond(cond, 1)
        flattened_extrinsics = cond[..., 4:]
        
        # shape = (T, 3, 4)
        extrinsics = rearrange(flattened_extrinsics, "t (i j) -> t i j", i=3, j=4)

        # shape = (T, T)
        # the rows index the current frame
        # the columns index the previous frame
        pose_closeness = self.are_poses_close(rearrange(extrinsics, "t i j -> () t i j"), rearrange(extrinsics, "t i j -> t () i j"))

        # mask out entries in pose_closeness where past_frame_index >= current_frame_index - self.retrieval_window_end_frame_rel
        # shape = (T, T)
        assert pose_closeness.shape[0] > self.retrieval_window_end_frame_rel, "video is too short to contain any loop closures"
        pose_closeness = torch.tril(pose_closeness, diagonal=-self.retrieval_window_end_frame_rel)

        # update the metadata dict to include the indices of the loop closing frames and the corresponding support frames
        # let's build this as a dict where the keys are the loop closing frame indices and the values are the support frame indices
        loop_closures = {}
        loop_closing_frames = torch.nonzero(pose_closeness.sum(dim=-1) > 0)

        for loop_closing_frame in loop_closing_frames:
            # get the support frames
            support_frames = torch.nonzero(pose_closeness[loop_closing_frame.item()])[:, 0]
            loop_closures[loop_closing_frame.item()] = support_frames.tolist()
        
        video_metadata["loop_closures"] = loop_closures

        # create new field inside video_metadata called "clip_start_frames" which contains a list of the start indices of clips that contain loop closing frames
        #assert self.cfg.max_frames % 2 == 0, "max_frames must be even to use clip filtering logic below"
        #assert self.augmentation.frame_skip_increase == 0, "frame_skip_increase must be 0 to use clip filtering logic below"

        # dict with key, values of clip_start_frame and corresponding loop closing frame
        clip_start_frames_and_loop_closing_frames = {}
        
        if self.split == "training":
            # we want start indices such that the loop closing frames are in the latter half of the clip of length self.cfg.n_frames
            deltas = torch.arange(((self.cfg.max_frames - self.n_support_frames) // 2) * self.frame_skip, (self.cfg.max_frames - self.n_support_frames - 1e-8) * self.frame_skip, self.frame_skip)
            for loop_closing_frame in loop_closures.keys():
                clip_start_frames = loop_closing_frame - deltas

                # clip start frames should be between 0 and the video length
                clip_start_frames = clip_start_frames[(clip_start_frames >= 0) & (clip_start_frames + (self.cfg.max_frames - 1) * self.frame_skip <= self.video_length(video_metadata) - 1)].int().tolist()
                for clip_start_frame in clip_start_frames:
                    clip_start_frames_and_loop_closing_frames[clip_start_frame] = loop_closing_frame
        else:
            # we want the start index to encompass both the loop closing frame and the support frames
            deltas = torch.arange(0, (self.cfg.n_frames - 1e-8) * self.frame_skip, self.frame_skip)
            for loop_closing_frame in loop_closures.keys():
                clip_start_frames = loop_closing_frame - deltas
                support_frames = loop_closures[loop_closing_frame]

                # clip start frames should be between 0 and the video length
                clip_start_frames = clip_start_frames[(clip_start_frames >= 0) & (clip_start_frames + (self.cfg.n_frames - 1) * self.frame_skip <= self.video_length(video_metadata) - 1)].int().tolist()

                for clip_start_frame in clip_start_frames:
                    # at least one support_frame should be between clip_start_frame, loop_closing_frame (exclusive)

                    if any([support_frame in range(clip_start_frame, loop_closing_frame, self.frame_skip) for support_frame in support_frames]):
                        
                        clip_start_frames_and_loop_closing_frames[clip_start_frame] = loop_closing_frame

        video_metadata["clip_start_frames"] = clip_start_frames_and_loop_closing_frames

        # sum over the previous frames to get the number of loop closures
        #return (pose_closeness.sum(dim=-1) > 0).sum().item()
        return len(set(video_metadata["clip_start_frames"].values()))

    def are_poses_close(
        self,
        extrinsics1: Float[Tensor, "... t i j"],
        extrinsics2: Float[Tensor, "... t i j"],
        translation_threshold: float = 5,         # units: meters
        rotation_threshold: float = 100,             # units: degrees 
    ) -> Bool[Tensor, "... t"]:
        
        dist, angle = dist_and_angle_between_cams(extrinsics1, extrinsics2)

        return torch.logical_and(
            dist < translation_threshold,
            angle < rotation_threshold,
        )

    def on_before_prepare_clips(self) -> None:
        super().on_before_prepare_clips()

        # filter videos to include only ones with camera trajectories containing loops i.e. revisits the same location
        self.metadata = self.exclude_videos_without_loops(self.metadata)

    def prepare_clips(self) -> None:

        num_clips = torch.as_tensor(
            [
                len(video_metadata["clip_start_frames"])
                for video_metadata in self.metadata
            ]
        )
        self.cumulative_sizes = num_clips.cumsum(0).tolist()
        self.idx_remap = self._build_idx_remap()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # at validation / test time, samples clips of length self.n_frames as usual

        if self.split != "training":
            video_idx, clip_idx = self.get_clip_location(idx)
            video_metadata = self.metadata[video_idx]
            video_length = self.video_length(video_metadata)
            start_frame, loop_closing_frame = list(video_metadata["clip_start_frames"].items())[clip_idx]
            end_frame = min(start_frame + self.n_frames, video_length)

            video, latent, cond, bev = None, None, None, None
            if self.use_preprocessed_latents:
                latent = self.load_latent(video_metadata, start_frame, end_frame)

            if self.use_preprocessed_latents and self.split == "training":
                # do not load video if we are training with latents
                if self.external_cond_dim > 0:
                    cond = self.load_cond(video_metadata, start_frame, end_frame)

            else:
                if self.external_cond_dim > 0:

                    if self.show_bev:
                        # load video together with condition
                        video, cond, bev = self.load_video_and_cond_and_bev(
                            video_metadata, start_frame, end_frame
                        )
                    else:
                        # load video together with condition
                        video, cond = self.load_video_and_cond(
                            video_metadata, start_frame, end_frame
                        )
                else:
                    # load video only
                    video = self.load_video(video_metadata, start_frame, end_frame)

            lens = [len(x) for x in (video, cond, latent, bev) if x is not None]
            assert len(set(lens)) == 1, "video, cond, latent, bev must have the same length"
            pad_len = self.n_frames - lens[0]

            nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
            if pad_len > 0:
                if video is not None:
                    video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
                if latent is not None:
                    latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
                if cond is not None:
                    cond = F.pad(cond, (0, 0, 0, pad_len)).contiguous()
                if bev is not None:
                    bev = F.pad(bev, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
                nonterminal[-pad_len:] = 0

            if self.frame_skip > 1:
                if video is not None:
                    video = video[:: self.frame_skip]
                if latent is not None:
                    latent = latent[:: self.frame_skip]
                nonterminal = nonterminal[:: self.frame_skip]
            if cond is not None:
                cond = self._process_external_cond(cond)
            if bev is not None:
                bev = bev[:: self.frame_skip]
        else:
            # at training time, samples clips of length self.max_frame, where the self.n_support_frames are imputed frames
            video_idx, clip_idx = self.get_clip_location(idx)
            video_metadata = self.metadata[video_idx]
            video_length = self.video_length(video_metadata)
            start_frame, loop_closing_frame = list(video_metadata["clip_start_frames"].items())[clip_idx]
            end_frame = min(start_frame + self.n_frames, video_length)

            # load video, cond, and latents if necessary
            #video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)
            if self.show_bev:
                video, cond, bev = self.load_video_and_cond_and_bev(video_metadata, start_frame, end_frame)
            else:
                video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)

            if self.use_preprocessed_latents:
                latent = self.load_latent(video_metadata, start_frame, end_frame)
            else:
                latent = None
            lens = [len(x) for x in (video, cond, latent, bev) if x is not None]
            assert len(set(lens)) == 1, "video, cond, latent, bev must have the same length"
            pad_len = self.n_frames - lens[0]

            nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
            if pad_len > 0:
                if video is not None:
                    video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
                if latent is not None:
                    latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
                if cond is not None:
                    cond = F.pad(cond, (0, 0, 0, pad_len)).contiguous()
                if bev is not None:
                    bev = F.pad(bev, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
                nonterminal[-pad_len:] = 0

            # skip frames
            video, cond, bev = video[::self.frame_skip], self._process_external_cond(cond, self.frame_skip), bev[::self.frame_skip]
            if self.use_preprocessed_latents:
                latent = latent[::self.frame_skip]
                latent = latent[None]

            # retrieve one of the support frames
            support_frames = video_metadata["loop_closures"][loop_closing_frame]
            assert len(support_frames) >= 1, "No support frames found for the current clip"
            assert self.n_support_frames == 1, "n_support_frames assumed to be 1 for now"
            support_frame = random.choice(support_frames)
            if self.show_bev:
                support_video, support_cond, support_bev = self.load_video_and_cond_and_bev(video_metadata, support_frame, support_frame + 1)
            else:
                support_video, support_cond = self.load_video_and_cond(video_metadata, support_frame, support_frame + 1)
            support_cond = self._process_external_cond(support_cond, 1)

            if self.use_preprocessed_latents:
                support_latent = self.load_latent(video_metadata, support_frame, support_frame + 1)
                support_latent = support_latent[None]
            else:
                support_latent = None

            # impute the support frames into the original frames at specified indices
            if self.show_bev:
                video_and_bev = torch.cat([video, bev], dim=-1)
                support_video_and_bev = torch.cat([support_video, support_bev], dim=-1)
            else:
                video_and_bev = video
                support_video_and_bev = support_video

            if self.show_bev:
                video_and_bev, cond, latent, nonterminal = impute_support_frames(video_and_bev[None], cond[None], support_video_and_bev[None], support_cond[None], self.impute_indices, latent, support_latent, nonterminal[None])
                video = video_and_bev[..., :video.shape[-1]]
                bev = video_and_bev[..., video.shape[-1]:video.shape[-1] + bev.shape[-1]]
            else:
                video, cond, latent, nonterminal = impute_support_frames(video[None], cond[None], support_video[None], support_cond[None], self.impute_indices, latent, support_latent, nonterminal[None])
            
            video = video[0]
            cond = cond[0]
            if self.show_bev:
                bev = bev[0]

            if self.use_preprocessed_latents:
                latent = latent[0]
            nonterminal = nonterminal[0]

            # augment
            if self.show_bev:
                video, cond, bev, latent = self._augment(video, cond, bev, latent)
            else:
                video, cond, latent = self._augment(video, cond, latent)

        output = {
            "videos": self.transform(video),            
            "latents": latent,
            "conds": cond,
            "bevs": bev,
            "nonterminal": nonterminal,
        }

        return {key: value for key, value in output.items() if value is not None}

class RealEstate10KAdvancedMultiViewDataset(
    RealEstate10KBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    RealEstate10K dataset that loads the entire video and randomly selects n_frames for multi-view training.
    This class is designed for training models that need to learn from multiple random views of the same scene.
    It is an alternative to RealEstate10KAdvancedVideoDataset that selects random frames from the entire video
    instead of sequential frames.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
        self.fix_intrinsics = cfg.fix_intrinsics
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)
        self.generator = torch.Generator(device="cpu")

    @property
    def _training_frame_skip(self) -> int:
        if self.augmentation.frame_skip_increase == 0:
            return self.frame_skip
        assert (
            self.current_subepoch is not None
        ), "Subepoch should be given to the RealEstate10KAdvancedVideoDataset, to use frame skip schedule"
        return self.frame_skip + int(
            self.current_subepoch * self.augmentation.frame_skip_increase
        )

    def exclude_videos_without_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split + "_{}".format(self.resolution)))
        return self.subsample(
            metadata,
            lambda video_metadata: self.video_metadata_to_latent_path(video_metadata)
            in latent_paths,
            "videos without latents",
        )

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"]
        path = self.save_dir / f"{self.split}_poses" / f"{path.stem}.pt"
        cond = torch.load(path, weights_only=False)[start_frame:end_frame]
        return cond

    def _augment(
        self,
        video: torch.Tensor,
        cond: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Only keep horizontal flip augmentation as it's still valid for multi-view training
        if random_bool(self.augmentation.horizontal_flip_prob):
            video = video.flip(-1)
            # NOTE: extrinsics should also be flipped accordingly - the following is equivalent to:
            # E' = I' @ E @ I' where I' = diag([-1, 1, 1, 1]) (E is 4x4 extrinsics matrix)
            cond[:, [5, 6, 7, 8, 12]] *= -1
            if latent is not None:
                latent = latent.flip(-1)

        return video, cond, latent

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # if self.split != "training":
        #     return super().__getitem__(idx)

        video_idx, _ = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)

        # seed the generator with current idx if it's not training
        if self.split != "training":
            self.generator.manual_seed(idx)

        # Randomly select n_frames indices
        selected_indices = torch.randperm(video_length, generator=self.generator)[:self.cfg.max_frames]

        # Load the entire video and poses
        video, cond = self.load_video_and_cond(video_metadata, 0, video_length)
        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, 0, video_length)
        else:
            latent = None

        # Select frames based on random indices
        video = video[selected_indices]
        cond = cond[selected_indices]
        cond = self._process_external_cond(cond)
        if latent is not None:
            latent = latent[selected_indices]

        # Apply augmentations
        video, cond, latent = self._augment(video, cond, latent)

        output = {
            "videos": self.transform(video),
            "latents": latent,
            "conds": cond,
            "nonterminal": torch.ones(self.cfg.max_frames, dtype=torch.bool),
        }

        return {key: value for key, value in output.items() if value is not None}

    def exclude_short_videos(
        self, metadata: List[Dict[str, Any]], min_frames: int
    ) -> List[Dict[str, Any]]:
        # if self.maximize_training_data is True,
        # include all videos with at least self.cfg.max_frames frames
        if self.maximize_training_data and self.split == "training":
            min_frames = min(min_frames, self.cfg.max_frames* self.cfg.max_frames_multiplier)
        return super().exclude_short_videos(metadata, min_frames)

    def _process_external_cond(
        self, external_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Converts the raw camera poses to concat-flattened intrinsics and extrinsics.
        Args:
            external_cond (torch.Tensor): Raw camera poses. Shape (T, 18).
            frame_skip (Optional[int]): Frame skip. If None, uses self.frame_skip.
        Returns:
            torch.Tensor: Processed camera poses. Shape (T, 16).
        """
        poses = external_cond
        if self.fix_intrinsics:
            poses[:, 0] = torch.max(poses[:, 0], poses[:, 1])
            poses[:, 1] = torch.max(poses[:, 0], poses[:, 1])
        return torch.cat(
            [
                poses[:, :4],
                poses[:, 6:],
            ],
            dim=-1,
        ).to(torch.float32)


def _timestamp_to_str(timestamp: int) -> str:
    timestamp = int(timestamp / 1000)
    str_hour = str(int(timestamp / 3600000)).zfill(2)
    str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
    str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
    str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
    str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
    return str_timestamp


def _youtube_url_to_id(youtube_url: str) -> str:
    return youtube_url.split("=")[-1]


def _download_youtube_video(youtube_url: str, download_dir: Path) -> None:
    """
    Downloads a YouTube video to the specified directory.
    Retries with different clients if the download fails, to guarantee that it does not miss any available video.
    """

    def download_with_client(client: Optional[str] = None):
        yt = YouTube(youtube_url) if client is None else YouTube(youtube_url, client)
        yt.streams.filter(res="360p").first().download(
            download_dir, filename=f"{_youtube_url_to_id(youtube_url)}.mp4"
        )

    try:
        download_with_client(youtube_url)
    except Exception:
        try:
            download_with_client("WEB_EMBED")
        except Exception:
            try:
                download_with_client("IOS")
            except Exception as e:
                print(f"Error downloading {youtube_url}: {e}")


def _read_frame(video_path: Path, timestamp: str) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-ss",
        timestamp,
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "bmp",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image_bytes, _ = process.communicate()
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    image = image.convert("RGB")
    image_array = np.array(image)
    return torch.from_numpy(image_array)


def _preprocess_video(
    info: Tuple[str, Path, List[float]],
    resolutions_to_preprocessing: Dict[int, VideoPreprocessingType],
):
    key, video_path, timestamps = info
    try:
        frames = []
        for timestamp in timestamps:
            frames.append(_read_frame(video_path, timestamp))
        video = torch.stack(frames, dim=0)
        assert video.shape[0] == len(
            timestamps
        ), f"Number of frames {video.shape[0]} does not match the number of timestamps {len(timestamps)} for {key}"

        for resolution, preprocessing_type in resolutions_to_preprocessing.items():
            video_preprocessed = rescale_and_crop(video, resolution)
            save_path = (
                video_path.parent.parent.parent
                / f"{video_path.parent.name}_{resolution}"
                / f"{key}.{preprocessing_type}"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if preprocessing_type == "npz":
                np.savez_compressed(
                    save_path, video=video_preprocessed.transpose(0, 3, 1, 2).copy()
                )
            elif preprocessing_type == "mp4":
                write_video(
                    filename=save_path,
                    video_array=torch.from_numpy(video_preprocessed).clone(),
                    fps=VideoPreprocessingMp4FPS,
                )

    except Exception as e:
        print(f"Error processing {key}: {e}")

