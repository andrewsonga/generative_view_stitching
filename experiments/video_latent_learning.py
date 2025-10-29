from datasets.video import (
    MinecraftAdvancedVideoDataset,
    Kinetics600AdvancedVideoDataset,
    RealEstate10KAdvancedVideoDataset,
)
from algorithms.vae import ImageVAETrainer, VideoVAETrainer
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoLatentLearningExperiment(BaseLightningExperiment):
    """
    An experiment for training & validating the first stage model (e.g. VAE)
    that learns the latent representation of the data
    """

    compatible_algorithms = dict(
        image_vae=ImageVAETrainer,
        video_vae=VideoVAETrainer,
    )

    compatible_datasets = dict(
        minecraft=MinecraftAdvancedVideoDataset,
        kinetics_600=Kinetics600AdvancedVideoDataset,
        realestate10k=RealEstate10KAdvancedVideoDataset,
    )

    data_module_cls = _data_module_cls
