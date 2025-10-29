from datasets.video import (
    MinecraftAdvancedVideoDataset,
    Kinetics600AdvancedVideoDataset,
    RealEstate10KAdvancedVideoDataset,
    RealEstate10KMiniAdvancedVideoDataset,
    RealEstate10KOODAdvancedVideoDataset,
)
from algorithms.dfot import (
    DFoTVideo,
    DFoTVideoPose,
    DFoTRAGVideoPose,
    GVSVideoPose,
    GVSScalableVideoPose,
    GVSNoOmniGuideVideoPose,
    StochSyncVideoPose,
)
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoGenerationExperiment(BaseLightningExperiment):
    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        dfot_video_pose=DFoTVideoPose,
        dfot_rag_video_pose=DFoTRAGVideoPose,
        sd_video=DFoTVideo,
        sd_video_3d=DFoTVideoPose,
        gvs_video_pose=GVSVideoPose,
        gvs_scalable_video_pose=GVSScalableVideoPose,
        gvs_no_omniguide_video_pose=GVSNoOmniGuideVideoPose,
        stochsync_video_pose=StochSyncVideoPose,
    )

    compatible_datasets = dict(
        # video datasets used in Diffusion-Forcing v1 and v2
        minecraft=MinecraftAdvancedVideoDataset,
        realestate10k=RealEstate10KAdvancedVideoDataset,
        realestate10k_ood=RealEstate10KOODAdvancedVideoDataset,
        realestate10k_mini=RealEstate10KMiniAdvancedVideoDataset,
        kinetics_600=Kinetics600AdvancedVideoDataset,
        # GVS datasets
        straight_line=RealEstate10KAdvancedVideoDataset,
        stairs=RealEstate10KAdvancedVideoDataset,
        panorama_1loop=RealEstate10KAdvancedVideoDataset,
        panorama_2loop=RealEstate10KAdvancedVideoDataset,
        circle_1loop=RealEstate10KAdvancedVideoDataset,
        circle_2loop=RealEstate10KAdvancedVideoDataset,
        staircase_circuit=RealEstate10KAdvancedVideoDataset,
        impossible_staircase=RealEstate10KAdvancedVideoDataset,
        indefinite_staircase_nframes1080_nloops9=RealEstate10KAdvancedVideoDataset,
        forward_orbit_backward=RealEstate10KAdvancedVideoDataset,
        forward_orbit_backward_loop_closing_window=RealEstate10KAdvancedVideoDataset,
    )

    data_module_cls = _data_module_cls
