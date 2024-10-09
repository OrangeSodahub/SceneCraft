import torch
import random
import torch.nn.functional as F
import concurrent.futures
import multiprocessing
from torch.nn import Module, Parameter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Union, Sized
from pathlib import Path
from itertools import cycle
from rich.progress import track

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.utils.dataloaders import CacheDataloader as _CacheDataloader
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataParser
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, variable_res_collate
from nerfstudio.model_components.ray_generators import RayGenerator

from scenecraft.utils import LoggableConfig, markvar
from scenecraft.data.dataset import InputDataset, GuideDataset, DataparserOutputs, DataParserConfig


__all__ = ["SceneCraftDataManagerConfig", "SceneCraftDataManager"]


def nerf_collate_fn(batch: List[dict]) -> dict:
    pass


"""DataMananger with local dataset as groundtruth"""

@dataclass
class SceneCraftDataManagerConfig(VanillaDataManagerConfig, LoggableConfig):
    """DataManager Config"""
    _target: Type = field(default_factory=lambda: SceneCraftDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = field(default_factory=lambda: DataParserConfig())
    """Dataparser if not use random camera data"""
    # NOTE: here we only sample one camera at each iteration, which is
    # different from sample random number of rays of all cameras.
    train_num_images_to_sample_from: int = markvar(1)
    """Number of images to sample during training iteration."""
    # NOTE: here each sampled camera be used only once before resample
    # the next camera, setting it to -1 will use the same camera all the time.
    train_num_times_to_repeat_images: int = markvar(0)
    """When not training on all images, number of iterations before picking new"""
    nerf_batch_size: int = markvar(8)
    """Number of views to be rendered of nerf model per step"""
    guide_buffer_size: int = markvar(250)
    """The queue size of stored diffusion-generated guide images"""
    camera_res_scale_factor: float = markvar(1.0)
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    guide_camera_res_scale_factor: float = markvar(1.0)
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    condition_type: str = markvar(default="embedding")
    """conditioning image type"""


class CacheDataloader(_CacheDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = "coarse"

    def _get_batch_list(self):
        """Returns a list of batches from the dataset attribute."""

        assert isinstance(self.dataset, Sized)

        if self.stage == "coarse":
            data_index_list = self.dataset.center_index
        else:
            data_index_list = range(len(self.dataset))

        indices = random.sample(data_index_list, k=self.num_images_to_sample_from)
        batch_list = []
        results = []

        num_threads = int(self.num_workers) * 4
        num_threads = min(num_threads, multiprocessing.cpu_count() - 1)
        num_threads = max(num_threads, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            for idx in indices:
                res = executor.submit(self.dataset.__getitem__, idx)
                results.append(res)

            for res in track(results, description="Loading data batch", transient=True):
                batch_list.append(res.result())

        return batch_list


class SceneCraftDataManager(VanillaDataManager):
    """DataManager deleted eval dataset.

    Args:
        config: the DataManagerConfig used to instantiate class
    """
    config: SceneCraftDataManagerConfig
    train_dataset: Union[InputDataset, GuideDataset]
    eval_dataset: InputDataset
    dataparser_outputs: DataparserOutputs

    def __init__(
        self,
        config: SceneCraftDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        is_training: bool = True,
        world_size: int = 1,
        local_rank: int = 0,
        nerf_ranks: list = None,
        guide_ranks: list = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        # the root parent
        Module.__init__(self)

        self.config = config
        self.device = device
        self.is_training = is_training
        self.world_size = world_size
        self.local_rank = local_rank
        self.downscale_factor = kwargs.get("downscale_factor", 1.)
        self.train_num_rays_per_batch = kwargs.get("train_num_rays_per_batch")
        self.prompt = kwargs.get("prompt", None)

        self.train_count = 0
        # get dataset info
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser: DataParser = self.dataparser_config.setup()
        self.includes_time = self.dataparser.includes_time
        self.dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(training=is_training)
        assert self.dataparser_outputs.cameras is not None, \
            f"Missing values for `dataparser_outputs.cameras` for dataset `{self.dataparser.__class__.__name__}`."

        CONSOLE.print("Setting up training dataset...")

        self._setup_eval()
        if self.is_training:
            self.train_dataparser_outputs = self.dataparser_outputs
            if local_rank in nerf_ranks:
                self._setup_nerf()
            if local_rank in guide_ranks:
                self._setup_guide()
            self._data_mode = "bootstrap"
        else:
            self.train_dataset = self.eval_dataset

        self.image_size = self.eval_dataset.image_size

    def _get_valid_guide_sample(self):
        """Exclusively for NeRF model's guide dataset. This is a infinite loop of total
           length of guidance image while the images will be filled iteratively.
        """
        while True:
            selected = random.sample(list(self.train_dataset.guide_images.keys()), self.config.nerf_batch_size)
            # TODO: finish this
            # yield nerf_collate_fn([self.train_dataset[i] for i in selected])
            yield [self.train_dataset[i] for i in selected][0]

    def _setup_nerf(self):

        # setup train dataset
        self.train_dataset = InputDataset(
            dataparser_outputs=self.dataparser_outputs, scale_factor=self.config.camera_res_scale_factor,
            guide_buffer_size=self.config.guide_buffer_size, training=True)
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")
        cameras = self.dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break

        # setup dataloader
        self.iter_train_image_dataloader = self._get_valid_guide_sample()
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )

        # pixel sampler
        self.train_pixel_sampler = self.config.pixel_sampler.setup(num_rays_per_batch=self.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device), pose_optimizer=lambda *args: None)

        CONSOLE.print(f"InputDataset sets for local_rank = {self.local_rank}")

    def _setup_guide(self):

        # setup train dataset
        self.train_dataset = GuideDataset(dataparser_outputs=self.dataparser_outputs,
                                          scale_factor=self.config.guide_camera_res_scale_factor,
                                          prompt=self.prompt,
                                          source_type=self.config.condition_type)
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # setup dataloader
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_camera_optimizer = None

        CONSOLE.print(f"GuideDataset sets for local_rank = {self.local_rank}")

    def _setup_eval(self):
        # setup eval dataset
        self.eval_dataset = InputDataset(
            dataparser_outputs=self.dataparser_outputs, scale_factor=self.config.guide_camera_res_scale_factor, training=False)

        cameras = self.dataparser_outputs.cameras
        if len(cameras) > 1:
            for i in range(1, len(cameras)):
                if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                    CONSOLE.print("Variable resolution, using variable_res_collate")
                    self.config.collate_fn = variable_res_collate
                    break

        # setup dataloader
        if self.is_training:
            self.iter_eval_image_dataloader = cycle(torch.utils.data.dataloader.DataLoader(
                self.eval_dataset, batch_size=1, shuffle=True))
        else:
            self.iter_eval_image_dataloader = iter(torch.utils.data.dataloader.DataLoader(
                self.eval_dataset, batch_size=1, shuffle=False))

        CONSOLE.print(f"InputDataset (Eval) sets for local_rank = {self.local_rank}")

    def next_train(self, step: int, **kwargs) -> Union[Tuple[RayBundle, Dict], Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1

        # [1] sample pixels from all valid images
        if not kwargs.get("full_image", False) and isinstance(self.train_dataset, InputDataset):
            image_batch = self.train_dataset.get_all_guide_images()
            image_batch = self.train_pixel_sampler.sample(image_batch)
            ray_indices = image_batch["indices"]
            ray_bundle = self.train_ray_generator(ray_indices)
            image_batch["scale_factor"] = self.dataparser_outputs.dataparser_scale
            return ray_bundle, image_batch

        # [2] sample one fixed full image
        # `image_batch`: (`dict`) {`image_idx`, ...}
        if "image_idx" in kwargs:
            image_batch = self.train_dataset[kwargs["image_idx"]]
        else:
            image_batch = next(self.iter_train_image_dataloader)

        # Input to NeRF Model
        if isinstance(self.train_dataset, InputDataset):
            camera_index = image_batch["image_idx"]
            # `self.train_dataset.cameras` could be rescaled and `self.dataparser_outputs.cameras` won't.
            camera: Cameras = self.train_dataset.cameras[camera_index].to(self.device)
            scaling_factor = random.uniform(self.downscale_factor, 1.)
            camera.rescale_output_resolution(scaling_factor=scaling_factor)
            ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
            image_batch["scaling_factor"] = scaling_factor
            image_batch["image_size"] = camera.image_height, camera.image_width
            if scaling_factor != 1:
                if "guide_image" in image_batch:
                    image_batch["guide_image"] = F.interpolate(image_batch["guide_image"], (camera.image_height, camera.image_width), mode="bilinear")
                if "guide_depth" in image_batch and image_batch["guide_depth"][0] is not None:
                    image_batch["guide_depth"] = F.interpolate(image_batch["guide_depth"], (camera.image_height, camera.image_width), mode="nearest")
                image_batch["reference_depth"] = F.interpolate(image_batch["reference_depth"][None, None], (camera.image_height, camera.image_width), mode='nearest')[0][0]

            # camera parameters
            image_batch["intrin"] = camera.get_intrinsics_matrices().to(self.device) # 3×3
            image_batch["extrin"] = camera.camera_to_worlds # 3×4
            image_batch["scale_factor"] = self.dataparser_outputs.dataparser_scale
            if "neighbor_depth" in image_batch:
                image_batch["coords"] = camera.get_image_coords(pixel_offset=0.).to(self.device)
                neighbor_camera: Cameras = self.train_dataset.cameras[image_batch["neighbor_idx"]].to(self.device)
                image_batch["neighbor_extrin"] = neighbor_camera.camera_to_worlds

            # Here the image has the values in range [0, 1].
            return ray_bundle, image_batch

        # Input to Guidance Pipeline
        elif isinstance(self.train_dataset, GuideDataset):
            assert len(image_batch["image_idx"]) == self.config.train_num_images_to_sample_from, \
                f"Wrong number of images, expected {self.config.train_num_images_to_sample_from}, got {len(image_batch['image_idx'])}."
            if self._data_mode != "bootstrap":
                self.train_image_dataloader.stage = "refine"
            return image_batch

    def next_eval(self, step: int=None, **kwargs) -> Tuple[RayBundle, Dict]:
        if "image_idx" in kwargs:
            camera_index = kwargs["image_idx"]
            image_batch = dict()
        else:
            image_batch = next(self.iter_eval_image_dataloader)
            camera_index = image_batch["image_idx"]
        camera = self.eval_dataset.cameras[camera_index].to(self.device)
        ray_bundle = camera.generate_rays(camera_indices=0, keep_shape=True)
        image_batch["scale_factor"] = self.dataparser_outputs.dataparser_scale
        return ray_bundle, image_batch

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        if self.train_camera_optimizer is not None:
            return super().get_param_groups()
        else:
            return {}
