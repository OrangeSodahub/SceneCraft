from typing import List, Optional, Literal, Type, Union, Sequence
from pathlib import Path
from dataclasses import dataclass, field
from jaxtyping import Float
from collections import OrderedDict, defaultdict
from itertools import chain
from torch import Tensor
from typing import Dict, Iterator
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from numpy import typing as npt
from pylab import *
import platform
import h5py
import random
import pandas
import glob
import json
import os
import torch
import datasets
import bisect
import pandas as pd
import numpy as np
import torch.nn.functional as F

from itertools import chain
from torch.utils.data import Sampler
from accelerate.logging import MultiProcessAdapter
from datasets import Dataset as _Dataset
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig as _DataParserConfig, \
                                                        DataParser as _DataParser, \
                                                        DataparserOutputs as _DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path

from scenecraft.utils import transform_points_to_voxels, format_poses, qvec2rotmat, sample_trajectory, \
                       load_from_jsonl, LoggableConfig, markvar, format_intrinsic, colmap_to_nerfstudio, opengl_to_opencv


__all__ = ["ScannetDataset", "ScannetppDataset", "HypersimDataset"]


""" NeRF Input Dataset """
@dataclass
class DataparserOutputs(_DataparserOutputs):
    """Dataparser outputs for the which will be used by the DataManager
        for creating RayBundle and RayGT objects."""

    dataset_type: str = None
    """Type str of dataset."""
    image_ids: List[str] = None
    """Image ids"""
    image_filenames: List[Path] = None # this item isn't be used
    """Filenames for the images."""
    cameras: Cameras = None
    """Camera object storing collection of camera information in dataset."""
    source_filenames: List[Path] = None
    """Filenames for the source cond images."""
    target_filenames: Optional[List[Path]] = None
    """Filenames for the target RGB images."""
    label_filenames: List[Path] = None
    """Filenames for the labels."""
    item_ids: List[List[int]] = None
    """(Processed) item ids for each sample."""
    captions: List[str] = None
    """Captions for each image."""
    camera_poses: List[torch.Tensor] = None
    """Original camera poses before adjustment."""
    center_index: List[int] = None
    """image index for center cameras."""
    random_index: List[int] = None
    """image index for random cameras."""


class InputDataset(torch.utils.data.Dataset):
    """Dataset that returns guidance image, ... input to nerf pipeline
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    _dataparser_outputs: DataparserOutputs
    cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, guide_buffer_size: int = 50,
                 training: bool = False):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.training = training
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.image_size = (self.cameras[0].image_width.item(), self.cameras[0].image_height.item())

        self.guide_images = OrderedDict()
        self.guide_buffer_size = guide_buffer_size

        if training:
            # ordered same as `target_filenames`
            self.reference_depths: Dict[str, torch.Tensor] = dict()
            for image_id, label_filename in zip(self.image_ids, self.label_filenames):
                assert image_id == label_filename.name.removesuffix('.npz')
                depth = torch.from_numpy(np.load(str(label_filename))["depths"])
                if scale_factor != 1:
                    depth = F.interpolate(depth[None, None], self.image_size[::-1], mode="nearest")[0][0]
                self.reference_depths[image_id] = depth

    def get_all_guide_images(self):

        sample_idx, image, reference_depth, guide_depth = [], [], [], []
        keys = list(self.guide_images.keys())
        random.shuffle(keys)
        for key in keys:
            image_idx = int(key.split('-')[0])
            if image_idx not in sample_idx:
                sample_idx.append(image_idx)
            else:
                continue
            image_id = str(self.source_filenames[image_idx].name.removesuffix('.png'))
            image.append(self.guide_images[key]["rgb"].permute(0, 2, 3, 1))
            reference_depth.append(self.reference_depths[image_id])
            if self.guide_images[key]["depth"][0] is not None:
                guide_depth.append(self.guide_images[key]["depth"].permute(0, 2, 3, 1))

        sample_idx = torch.tensor(sample_idx, dtype=torch.long)
        image = torch.cat(image) # [B, H, W, C]
        reference_depth = torch.stack(reference_depth)

        perm_indices = torch.randperm(sample_idx.shape[0])[:50]
        sample_idx = sample_idx[perm_indices]
        image = image[perm_indices]
        reference_depth = reference_depth[perm_indices]

        guide_depth_dict = {}
        if guide_depth:
            guide_depth = torch.cat(guide_depth) # [B, H, W, C]
            guide_depth_dict["guide_depth"] = guide_depth

        return {"image_idx": sample_idx, "image": image, "reference_depth": reference_depth, **guide_depth_dict}
        
    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, key: Union[str, int]) -> Dict:
        """ here `image_idx` has a dynamic range [0, len(self.target_images) - 1]"""
        image_idx = int(key.split('-')[0]) if isinstance(key, str) else key
        image_id = self.image_ids[image_idx]
        data = {"image_idx": image_idx, "image_id": image_id}

        if self.training:
            if self.guide_images:
                guide_image = self.guide_images[key]["rgb"]
                guide_depth = self.guide_images[key]["depth"]
                data.update(guide_image=guide_image, guide_depth=guide_depth)
            # get current reference depth
            reference_depth = self.reference_depths[image_id]
            data.update(reference_depth=reference_depth)

        return data

    def put_guide_object(self, object: Dict[int, torch.Tensor], step: int):
        while self.full:
            self.guide_images.popitem(last=False)
        rgb = object["image"]
        depth = object["depth"]
        if rgb.shape[-2:] != self.image_size[::-1]:
            rgb = F.interpolate(rgb, self.image_size[::-1], mode="bilinear")
            if depth[0] is not None:
                depth = F.interpolate(depth, self.image_size[::-1], mode="nearest")
        self.guide_images[f"{object['idx']}-{step}"] = {"rgb": rgb, "depth": depth}

    @property
    def all_guide_indices(self) -> List[int]:
        return list(map(lambda x: int(x.split('-')[0]), self.guide_images.keys()))

    @property
    def full(self) -> bool:
        return len(self.guide_images) >= self.guide_buffer_size

    @property
    def image_ids(self) -> List[str]:
        return self._dataparser_outputs.image_ids

    @property
    def source_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """
        return self._dataparser_outputs.source_filenames

    @property
    def label_filenames(self) -> List[Path]:
        """
        Returns label filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """
        return self._dataparser_outputs.label_filenames


class GuideDataset(torch.utils.data.Dataset):
    """Dataset that returns samples containing images, prompts, ... input to guidance pipeline
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    _dataparser_outputs: DataparserOutputs

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, prompt: str = None,
                 source_type: str = "rgb"):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        # NOTE: in practice the guidance pipeline doesn't have to apply `scale_factor`
        self.scale_factor = scale_factor
        self.prompt = prompt
        self.source_type = ("indice" if source_type != "rgb" else source_type)
        self.image_size = self.get_image(self.source_filenames[0], "source").shape[:2][::-1]
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.scale_image_size = None
        if self._dataparser_outputs.captions is not None:
            CONSOLE.print(f"Loaded {len(self._dataparser_outputs.captions)} prompts.")
        else:
            assert self.prompt is not None and self.prompt != "", f"Invalid prompt input: {self.prompt}"
            CONSOLE.print(f"Use prompt: {prompt}.")

        self.center_index = dataparser_outputs.center_index
        self.random_index = dataparser_outputs.random_index
        assert len(self.center_index) + len(self.random_index) == len(self.cameras), \
            f"Loaded {len(self.center_index)} center poses and {len(self.random_index)} random poses but {len(self.cameras)} cameras."
        if len(self.center_index) == 0:
            self.center_index = self.random_index

    def get_image(self, image_filename: str, type: str, source_type: str = "rgb") -> Float[Tensor, "image_height image_width num_channels"]:
        if image_filename is None or image_filename == 'None':
            return None

        if type == "source" and source_type == "indice":
            assert image_filename.endswith('.npz'), f"Got unexpected image file type '{image_filename}', expect 'npz'."
            pil_image = Image.fromarray(np.load(image_filename)["labels"].astype(np.uint8))
        elif type == "depth":
            assert image_filename.endswith('.npz'), f"Got unexpected image file type '{image_filename}', exptect 'npz'."
            label_file = np.load(image_filename)
            pil_image = Image.fromarray(label_file["depths"]) if "depths" in label_file.files else None
        else:
            pil_image = Image.open(image_filename).convert("RGB")

        if pil_image is None:
            return None

        # check image shape, hard code here
        if pil_image.size[0] < pil_image.size[1]:
            width, height = pil_image.size
            if self._dataparser_outputs.dataset_type == "custom":
                pil_image = pil_image.crop((0, 0, width, height // 2))
            else:
                pil_image = pil_image.crop((0, height // 3, width, height // 3 * 2))
        width, height = pil_image.size
        newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
        if self.scale_factor != 1.0:
            pil_image = pil_image.resize(newsize, resample=(Image.NEAREST if type=="source" else Image.BILINEAR))
        # update image size info
        self.scale_image_size = newsize

        image = torch.from_numpy(np.array(pil_image, dtype="uint8").astype(np.float32)) # shape is (h, w) or (h, w, 3 or 4)
        # nomalize the image if not the indice image
        if (type == "source" and source_type == "rgb") or type == "target": image = image / 255.
        return image

    def __getitem__(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.
        """
        # conds and images
        source_filename = self.source_filenames[image_idx]
        source_image = self.get_image(str(source_filename), type="source")
        label_filename = self.label_filenames[image_idx]
        depth_image = self.get_image(str(label_filename), type="depth")
        if self.source_type == "indice":
            indice_image = self.get_image(str(label_filename), type="source", source_type=self.source_type)
        image_id = str(source_filename.name.removesuffix('.png'))
        data = {"image_idx": image_idx, "image_id": image_id, "image": source_image, "indice_image": indice_image, "prompt": self.prompts[image_idx]}

        if depth_image is not None:
            data.update(depth_image=depth_image)

        if self.target_filenames is not None:
            target_filename = self._dataparser_outputs.target_filenames[image_idx]
            target_image = self.get_image(str(target_filename), type="target")
            if target_image is None:
                target_image = torch.zeros((*self.scale_image_size[::-1], 3)) # to avoid errors in collate_fn
            data.update(target_image=target_image)

        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"

        return data

    def __len__(self):
        return len(self.cameras)

    @property
    def prompts(self) -> List[str]:
        if self._dataparser_outputs.captions:
            return self._dataparser_outputs.captions
        return [self.prompt] * len(self)

    @property
    def source_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """
        return self._dataparser_outputs.source_filenames

    @property
    def target_filenames(self) -> List[Path]:
        """
        Returns target rgb image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """
        return self._dataparser_outputs.target_filenames

    @property
    def label_filenames(self) -> List[Path]:
        """
        Returns label filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """
        return self._dataparser_outputs.label_filenames


""" StableDiffusion Dataset """
class FinetuneDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="scannet", version=VERSION, description="ScanNet Dataset."),
        datasets.BuilderConfig(name="scannetpp", version=VERSION, description="ScanNetPP Dataset."),
        datasets.BuilderConfig(name="hypersim", version=VERSION, description="HyperSim Dataset."),
        datasets.BuilderConfig(name="custom", version=VERSION, description="Custom Dataset."),
        datasets.BuilderConfig(name="scannet_online", version=VERSION, description="Online scannet Dataset."),
        datasets.BuilderConfig(name="scannetpp_online", version=VERSION, description="Online scannet++ Dataset."),
        datasets.BuilderConfig(name="hypersim_oneline", version=VERSION, description="Online hypersim Dataset."),
    ]

    BUILDER_PARAMS = {
        "scannet": dict(
            JSONL_PATH="./data/scannet/",
        ),
        "scannetpp": dict(
            JSONL_PATH="./data/scannetpp_processed/",
            COND_EMBED_PATH="./scenecraft/data/scannetpp_utils/meta_data/condition_embedding.npy",
        ),
        "hypersim": dict(
            JSONL_PATH="./data/hypersim/",
            COND_EMBED_PATH="./scenecraft/data/hypersim_utils/meta_data/condition_embedding.npy",
        ),
        "custom": dict(
            JSONL_PATH="./exp/",
            COND_EMBED_PATH="./scenecraft/data/hypersim_utils/meta_data/condition_embedding.npy",
        )
    }

    def _info(self):
        dataset_type = self.config.name.strip("_online")
        h, w = RAW_DATASETS[dataset_type].ori_h, RAW_DATASETS[dataset_type].ori_w
        if "online" in self.config.name:
            online_features = {
                        "labels": datasets.Array2D((h, w), dtype="int32"),
                        "cond_depths": datasets.Array2D((h, w), dtype="float32"),
                        "extrin": datasets.Array2D((4, 4), dtype="float32"),
                        "intrin": datasets.Array2D((4, 4), dtype="float32"),
            }
        else:
            online_features = {}

        return datasets.DatasetInfo(
            description="",
            features=datasets.Features({
                    "scene_id": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "target": datasets.Image() if self.config.name != "custom" else datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "source_image": datasets.Image(),
                    **online_features,
                    "depth": datasets.Value("string"),
                    "item_id": datasets.Value("string"),
                    "prompt": datasets.Value("string")}),
            homepage="", license="", citation="")

    def _split_generators(self, dl_manager):
        dataset_type = self.config.name.strip("_online")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(self.BUILDER_PARAMS[dataset_type]["JSONL_PATH"], "all.jsonl"),
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):

        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                source = np.load(data["source"])

                if "online" in self.config.name:
                    online_data = {
                        "labels": source["labels"],
                        "cond_depths": source["depths"],
                        "extrin": source["extrin"].reshape(4, 4),
                        "intrin": source["intrin"].reshape(4, 4),
                    }
                else:
                    online_data = {}

                yield key, {
                    "scene_id": data["scene_id"],
                    "id": data["id"],
                    "target": "" if self.config.name == "custom" else data["target"],
                    "source": data["source"],
                    "source_image": data["source"].replace("npz", "png"),
                    **online_data,
                    "depth": data["source"].replace("semantic_images", "depths").replace("npz", "bin"),
                    "item_id": data["item_id"],
                    "prompt": "This is one view of a room.",
                }


""" Diffuser ArrowDataset (from `FinetuneDataset`) """
class Dataset(_Dataset):
    def __init__(self, dataset: _Dataset, logger: MultiProcessAdapter, **kwargs):
        super().__init__(dataset._data)
        self.__dict__.update(dataset.__dict__)
        self.kwargs = kwargs
        self.logger = logger
        scene_id_column = self.get_column_data(column_name="scene_id", output_type="pandas")
        scene_sample_counts = scene_id_column.value_counts()
        self.scene_sample_counts_dict = {scene_id: counts for scene_id, counts in zip(
                                         scene_sample_counts.index, scene_sample_counts)}
        self.scene_sample_counts_dict = {scene_id: self.scene_sample_counts_dict[scene_id]
                                         for scene_id in self.get_column_data(column_name="scene_id")}
        self.scene_sample_counts_list = list(self.scene_sample_counts_dict.values())
        self.num_scenes = len(self.scene_sample_counts_list)
        assert self.num_scenes != 0, "Got unexpected zero scenes in dataset."
        logger.info(f"Loaded dataset info: number of scenes {self.num_scenes}, "
                    f"number of samples per scene {self.scene_sample_counts_list}, "
                    f"total samples {self.num_rows} (maybe not original dataset).")

    def get_unique_scene_ids(self) -> list:
        return self.unique("scene_id")

    def get_column_data(self, column_name: str, output_type: str = "list") -> Union[list, dict, pandas.array]:
        assert column_name in self.column_names, f"Unknown column name {column_name}."
        column_data = self._data.columns[self.column_names.index(column_name)] # column 7 is prompt
        if output_type == "dict":
            column_data = column_data.to_pydict()
        elif output_type == "list":
            column_data = column_data.to_pylist()
        elif output_type == "pandas":
            column_data = column_data.to_pandas()
        return column_data

    def _getitem(self, key):
        try:
            return super()._getitem(key)
        except OSError as e:
            print(f"Skipped current batch data due to {e}.")
            return None

    def __getitems__(self, keys: List) -> List:
        """Can be used to get a batch using a list of integers indices."""
        batch = self.__getitem__(keys)
        if batch is None:
            return None
        n_examples = len(batch[next(iter(batch))])
        return [{col: array[i] for col, array in batch.items()} for i in range(n_examples)]


@dataclass
class DataParserConfig(_DataParserConfig, LoggableConfig):
    """ Inherit the `LoggableConfig` """

    _target: Type = field(default_factory=lambda: DataParser)
    """target class to instantiate"""
    dataset_type: str = markvar("scannetpp")
    """Type of dataset."""
    data: Path = Path("outputs/scannetpp")
    """Directory to the root of training data."""
    scene_id: str = markvar("0a7cc12c0e")
    """Scene id."""
    scale_factor: float = markvar(1.0)
    """How much to scale the camera origins by."""
    scene_scale: float = markvar(1.5)
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = markvar(0.9)
    """The fraction of images to use for training. The remaining images are for eval."""

    # Scannet
    has_label: bool = True,
    """Whether the dataset has labels or not."""
    load_depth: bool = True,
    """Whether to load depth images or not."""
    load_frames: int = markvar(300)
    """Number of frames to load per scene."""
    use_frames: int = markvar(100)
    """Number of frames to use per scene."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

    # Scannet++
    images_dir: Path = Path("dslr/resized_images_2")
    """Relative path to the images directory (default: resized_images)"""
    masks_dir: Path = Path("dslr/resized_anon_masks_2")
    """Relative path to the masks directory (default: resized_anon_masks)"""
    transforms_path: Path = Path("dslr/nerfstudio/transforms_2.json")
    """Relative path to the transforms.json file"""


@dataclass
class DataParser(_DataParser):

    config: DataParserConfig

    def _trans_scene(self) -> npt.NDArray:
        vec = []
        for matrix, scene in zip(self.axis_align_matrixes, self.scene_names):
            mesh_path = os.path.join(self.label_dir, f'{scene}_vert.npy')
            unaligned_pts = np.load(mesh_path)[:, :3]
            unaligned_pts = np.concatenate([unaligned_pts, np.ones([unaligned_pts.shape[0], 1])], axis=-1)
            aligned_pts = np.dot(unaligned_pts, matrix.transpose()) # [N, 4]
            vec.append([np.min(aligned_pts[:, 0]), np.min(aligned_pts[:, 1]), np.min(aligned_pts[:, 2])])
        return np.array(vec)

    def _get_valid_ids(self, all_extrins: npt.NDArray) -> list:
        all_trans = np.array([extrins[:3, 3] for extrins in all_extrins])
        bx, by = self.kwargs['x_bound'][:2], self.kwargs['y_bound'][:2]
        all_ids = list(range(len(all_extrins)))
        all_valid_ids = list(filter(lambda i: all_trans[i][0] >= bx[0] and all_trans[i][0] <= bx[1]
                                          and all_trans[i][1] >= by[0] and all_trans[i][1] <= by[1], all_ids))
        return all_valid_ids

    def _filter_frames(self, save_ids: bool=False) -> None:
        save_dir = self.config.data / "selected_frames"
        all_frame_ids = []
        for i, scene in tqdm(enumerate(self.split_available_scenes)):
            fname = os.path.join(str(save_dir), f'{scene}.txt')
            try:
                frame_ids = np.loadtxt(fname).astype(int)
            except:
                scene_path = os.path.join(str(self.image_dir), scene)
                all_extrins = sorted(glob.glob(os.path.join(scene_path, '*.txt')))
                all_extrins.remove(os.path.join(scene_path, 'intrinsic.txt'))
                all_extrins = np.array([np.loadtxt(extrins, dtype=float) for extrins in all_extrins])
                all_extrins = np.matmul(self.axis_align_matrixes[i], all_extrins) # [N, 4, 4]
                all_extrins[:, :3, 3] -= self.trans_scene_vec[i]
                all_valid_ids = self._get_valid_ids(all_extrins)
                all_depths = sorted(glob.glob(os.path.join(scene_path, '*.png'))) if self.config.load_depth else None
                intrins = np.loadtxt(os.path.join(scene_path, 'intrinsic.txt'), dtype=float) # [4, 4]
                all_valid_ids, frame_ids = sample_trajectory(all_valid_ids, all_extrins, all_depths, intrins, self.config.image_size)
                # add more frames to the fixed number
                if len(frame_ids) < self.config.load_frames:
                    ids = np.setdiff1d(np.array(all_valid_ids), np.array(frame_ids))
                    add_ids = np.random.choice(ids, self.config.load_frames - len(frame_ids), replace=False)
                    frame_ids = sorted(np.concatenate([np.array(frame_ids), add_ids]).tolist())
                if save_ids:
                    os.makedirs(save_dir, exist_ok=True)
                    np.savetxt(fname, np.array(frame_ids), fmt='%d')
            tqdm.write(f'{len(frame_ids)} frames are selected from {scene}.')
            all_frame_ids.append(frame_ids)
        return all_frame_ids

    def _filter_scenes(self, save_data: bool=False) -> None:
        save_dir = self.config.data / "selected_scenes"
        fname = save_dir / "scannet.txt"
        try:
            data = np.loadtxt(fname)
        except:
            data = np.loadtxt('scenecraft/data/scannet_utils/scannet.txt')
            sum = np.sum(data[:, 3:5], axis=-1)
            data = data[np.argsort(sum)[::-1]]
            data = data[(data[:, 3] > 4.3) & (data[:, 3] < 8.) & (data[:, 4] > 4.3) & (data[:, 4] < 8.)]
            if save_data:
                os.makedirs(save_dir, exist_ok=True)
                np.savetxt(fname, data, fmt='%s')
        print(f"Number of selected scenes: {data.shape[0]}")
        scene_ids = data[:, 0].reshape(-1).astype(int)
        return scene_ids

    def _generate_dataparser_outputs(self, *args, **kwargs):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        if not (self.config.data / self.config.scene_id / "all.jsonl").exists():
            raise FileNotFoundError(f"JONSL file not found for scene {self.config.scene_id}.")
        data_info = load_from_jsonl(self.config.data / self.config.scene_id / "all.jsonl")
        if len(data_info) == 0:
            raise ValueError(f"Not found scene {self.config.scene_id} in JSONL file.")
        index = list(range(len(data_info)))
        center_index = list(filter(lambda idx: data_info[idx]["id"].startswith("center"), index))
        random_index = list(filter(lambda idx: not data_info[idx]["id"].startswith("center"), index))
        if len(random_index) == 0:
            self.config.auto_scale_poses = False
        CONSOLE.log(f"Loaded {len(data_info)} samples from scene {self.config.scene_id} "
                    f"which contains {len(center_index)} center samples and {len(random_index)} random samples")

        # load source files and camera data
        captions = None
        image_ids, target_image_paths, source_paths, item_ids = \
            list(zip(*map(lambda x: [str(x["id"]), Path(x["target"]) if x["target"] else None, Path(x["source"]), list(x["item_id"])], data_info)))
        if self.config.dataset_type == "custom" and "prompt" in data_info[0] and data_info[0]["prompt"] is not None:
            captions = list(map(lambda x: x["prompt"], data_info))
        source_image_paths = [source_path.with_suffix(".png") for source_path in source_paths]

        # load poses and transform from colmap to nerfstudio
        # https://github.com/nerfstudio-project/nerfstudio/issues/1286
        # https://github.com/nerfstudio-project/nerfstudio/issues/1504
        poses = list(map(lambda source_path: colmap_to_nerfstudio(np.load(source_path)["extrin"].reshape(4, 4))
                                                if RAW_DATASETS[self.config.dataset_type].coord_convention in ("opencv", "colmap") else
                                                colmap_to_nerfstudio(opengl_to_opencv(np.load(source_path)["extrin"].reshape(4, 4))), source_paths))

        orientation_method = self.config.orientation_method
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        original_poses = poses.clone()
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses, method=orientation_method, center_method=self.config.center_method,
        )

        # Scale the poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # assumes that the scene is centered at the origin
        if not self.config.auto_scale_poses:
            # Set aabb_scale to scene_scale * the max of the absolute values of the poses
            aabb_scale = self.config.scene_scale * float(torch.max(torch.abs(poses[:, :3, 3])))
        else:
            aabb_scale = self.config.scene_scale
        scene_box = SceneBox(aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]],
                dtype=torch.float32))

        # should be `PINHOLE` after preprocessing
        camera_type = CameraType.PERSPECTIVE

        # intrin = [[fl_x, 0, cx, 0], [0, fl_y, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        intrin_example = np.load(source_paths[0])["intrin"]
        fx = float(intrin_example[0][0])
        fy = float(intrin_example[1][1])
        cx = float(intrin_example[0][2])
        cy = float(intrin_example[1][2])
        height = int(cy * 2)
        width = int(cx * 2)
        distortion_params = camera_utils.get_distortion_params(0., 0., 0., 0., 0., 0.)

        cameras = Cameras(
            fx=fx, fy=fy, cx=cx, cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        dataparser_outputs = DataparserOutputs(
            dataset_type=self.config.dataset_type,
            image_ids=image_ids,
            captions=captions,
            # training and test
            target_filenames=list(target_image_paths),
            source_filenames=list(source_image_paths),
            label_filenames=list(source_paths),
            item_ids=item_ids,
            center_index=center_index,
            random_index=random_index,
            # undistorted cameras
            cameras=cameras,
            camera_poses=original_poses,
            scene_box=scene_box,
            mask_filenames=None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={},
        )
        return dataparser_outputs


""" Raw Dataset """

SPLIT_PATHS = {
    'scannet': {
        'train': 'scenecraft/data/scannet_utils/meta_data/scannetv2_train.txt',
        'val': 'scenecraft/data/scannet_utils/meta_data/scannetv2_val.txt',
        'test': 'scenecraft/data/scannet_utils/meta_data/scannetv2_test.txt',
    },
    'scannetpp': {
        'train': 'scenecraft/data/scannetpp_utils/meta_data/scannetpp_train.txt',
        'val': 'scenecraft/data/scannetpp_utils/meta_data/scannetpp_val.txt',
    },
    'hypersim': {
        'train': 'scenecraft/data/hypersim_utils/meta_data/hypersim_train.txt'
    },
}


class ScannetDataset(torch.utils.data.Dataset):

    CLASSES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'counter', 'desk', 'curtain',
               'refrigerator', 'television', 'showercurtain', 'whiteboard',
               'toilet', 'sink', 'bathtub', 'garbagebin', 'doorframe']

    CAT_IDS = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 24, 25, 28, 30, 33, 34, 36, 39, 41])

    COLORS = np.array(
        [[176, 224, 230], [255, 192, 203], [255, 228, 196], [255, 218, 185],
         [240, 230, 140], [255, 235, 205], [255, 250, 205], [240, 128, 128],
         [255, 99, 71],   [255, 69, 0],    [198, 255, 240], [210, 182, 198],
         [255, 165, 0],   [255, 215, 0],   [255, 250, 240], [255, 255, 224],
         [154, 205, 50],  [173, 216, 230], [236, 140, 56],  [179, 230, 211],
         [135, 206, 250], [240, 155, 25],  [224, 173, 246], [240, 255, 255],
         [255, 120, 243], [234, 211, 234]])

    rgb_image_suffix = ".jpg"
    cat2label = {cat: label for label, cat in enumerate(CLASSES)}
    label2cat = {label: cat for cat, label in cat2label.items()}
    cat_ids2class = {nyu40id: i for i, nyu40id in enumerate(CAT_IDS)}

    def __init__(self, root_dir: str='./data/scannet', split: str='train', load_depth: bool=False,
                 load_label: bool=True, align: bool=False, limit: int=None, voxel_size: float=-1, **kwargs) -> None:
        super().__init__()
        # kwargs contains the data augmentation configs
        self.kwargs = kwargs
        self.root_dir = root_dir
        self.load_label = load_label
        self.load_depth = load_depth
        self.align = align
        self.limit = limit
        self.voxel_size = voxel_size
        self.image_dir = os.path.join(self.root_dir, 'posed_images')
        self.label_dir = os.path.join(self.root_dir, 'scannet_instance_data')
        assert os.path.exists(self.image_dir), f'{self.image_dir} does not exist.'

        all_scenes = list(sorted(os.listdir(self.image_dir)))
        splits = ['train', 'val', 'test'] if split == 'all' else ['train', 'val'] if split == 'trainval' else [split]
        split_scenes = []
        for split in splits:
            with open(SPLIT_PATHS['scannet'][split], 'r') as file:
                split_scene = [line.strip() for line in file.readlines()]
                split_scenes += list(filter(lambda x: x in all_scenes, split_scene))
        self.split_available_scenes = sorted(split_scenes)
        self.axis_align_matrixes = np.array([np.load(os.path.join(self.label_dir, f'{scene}_axis_align_matrix.npy')
                                                   ).reshape(4, 4) for scene in self.split_available_scenes])
        print(f"======> Load {len(self.split_available_scenes)} scenes from {splits} splits.")

        self.voxel_data_dir = None
        if self.voxel_size != -1:
            self.voxel_data_dir = os.path.join(root_dir, 'voxel_data', f'size{voxel_size}')
            if not (os.path.exists(self.voxel_data_dir) and len(os.listdir(self.voxel_data_dir)) == len(self.split_available_scenes)):
                os.makedirs(self.voxel_data_dir, exist_ok=True)
                print(f"Start voxelization, voxel size was set to {voxel_size}, data will be saved to {self.voxel_data_dir}.")
                instance_data_path = os.path.join(root_dir, 'scannet_instance_data')
                assert os.path.exists(instance_data_path), f"Instance data {instance_data_path} doesn't exist, " \
                                "run python scenecraft/data/scannet_utils/_batch_load_scannet_data.py to fix it."
                for i, scene in tqdm(enumerate(self.split_available_scenes)):
                    if os.path.exists(os.path.join(self.voxel_data_dir, f'{scene}.npy')):
                        tqdm.write(f"Skipped scene {scene}.")
                        continue
                    points = np.load(os.path.join(instance_data_path, f'{scene}_vert.npy'))
                    labels = np.load(os.path.join(instance_data_path, f'{scene}_sem_label.npy'))
                    # align axis
                    aligned_points = np.ones((points.shape[0], 4))
                    aligned_points[:, :3] = points[:, :3]
                    aligned_points = np.dot(aligned_points, self.axis_align_matrixes[i].transpose())
                    aligned_points = np.concatenate([aligned_points[:, :3], points[:, 3:]], axis=1)
                    labels = np.array([self.cat_ids2class[cat_id] for cat_id in labels])
                    voxels = transform_points_to_voxels(aligned_points, voxel_size=[voxel_size] * 3, labels=labels)
                    print(f"Get {len(voxels)} voxels in scene {scene}.")
                    np.save(os.path.join(self.voxel_data_dir, f'{scene}.npy'), voxels)

    @staticmethod
    def get_relative_rgb_image_path(scene_id: str, id: str) -> str:
        return f'posed_images/{scene_id}/{id}.jpg'

    def get_scene_path(self, scene_id: str) -> str:
        return os.path.join(self.image_dir, scene_id)

    def get_scene_image_path(self, scene_id: str) -> str:
        return self.get_scene_path(scene_id)

    def __len__(self) -> int:
        return len(self.split_available_scenes)
  
    def __getitem__(self, idx: int, selected: List[int] = None) -> Sequence[npt.NDArray]:
        sampled_scene = self.split_available_scenes[idx]
        scene_path = self.get_scene_path(sampled_scene)
        # Align the axis and reset the origin to the corner of the scene
        all_extrins = sorted(glob.glob(os.path.join(scene_path, '*.txt')))
        all_extrins.remove(os.path.join(scene_path, 'intrinsic.txt'))
        all_extrins = np.array([np.loadtxt(extrins, dtype=float) for extrins in all_extrins])
        valid_ids = filter(lambda i: not np.any(np.isinf(all_extrins[i])), range(len(all_extrins)))
        valid_ids = np.array(list(valid_ids)).astype(np.int32)
        # Set limit
        N = len(valid_ids)
        limit = N if self.limit is None or self.limit >= N else self.limit
        replace = False if limit < N else True
        if selected is None:
            selected = np.random.choice(N, limit, replace=replace).astype(np.int32)
        valid_ids = valid_ids[selected]
        if self.align:
            all_extrins = np.matmul(self.axis_align_matrixes[idx], all_extrins[valid_ids]) # [N, 4, 4]
        else:
            all_extrins = all_extrins[valid_ids]
        all_images = sorted(glob.glob(os.path.join(scene_path, f'*{self.rgb_image_suffix()}')))
        all_depths = sorted(glob.glob(os.path.join(scene_path, '*.png'))) if self.load_depth else None
        intrins = np.loadtxt(os.path.join(scene_path, 'intrinsic.txt'), dtype=float) # [4, 4]
        # color images (single view) and extrinsics (per image)
        selected_ids = [all_images[i].split('/')[-1][:-4] for i in valid_ids]
        all_images = [np.array(Image.open(all_images[i])) for i in valid_ids]
        all_depthmaps = [np.array(Image.open(all_depths[i])) / 1000 for i in valid_ids] if self.load_depth else None
        intrins = np.broadcast_to(intrins, (len(all_images), 4, 4)) # [N, 4, 4]
        intrins.flags.writeable = True
        # meta infos
        metas = dict(scene_id=sampled_scene, selected_ids=selected_ids, voxel_size=self.voxel_size)
        if all_depthmaps is not None:
            metas.update(all_depthmaps=all_depthmaps)
        if self.load_label:
            bbox_name = 'aligned_bbox.npy' if self.align else 'unaligned_bbox.npy'
            box_label = np.load(os.path.join(self.label_dir, f'{sampled_scene}_{bbox_name}')) # [N, 6+class]
            box_label = np.array(list(filter(lambda x: int(x[-1]) in self.cat_ids2class, box_label)))
            box_num = box_label.shape[0]
            if box_num > 0:
                sample_boxes, sample_labels = box_label[:, :6], box_label[:, -1]
                sample_labels = np.array([self.cat_ids2class[sample_labels[i]] for i in range(box_num)])
                metas.update(obj_bbox=sample_boxes, obj_class=sample_labels)
        if self.voxel_data_dir is not None:
            voxel_data_path = os.path.join(self.voxel_data_dir, f'{sampled_scene}.npy')
            if os.path.exists(voxel_data_path):
                voxels = np.load(voxel_data_path)
                metas.update(voxels=voxels)
                points_path = os.path.join(self.root_dir, 'scannet_instance_data', f'{sampled_scene}_vert.npy')
                points = np.load(points_path)
                aligned_points = np.ones((points.shape[0], 4))
                aligned_points[:, :3] = points[:, :3]
                aligned_points = np.dot(aligned_points, self.axis_align_matrixes[idx].transpose())
                point_cloud_range = np.array([np.min(aligned_points[:, 0]), np.min(aligned_points[:, 1]), np.min(aligned_points[:, 2]),
                                              np.max(aligned_points[:, 0]), np.max(aligned_points[:, 1]), np.max(aligned_points[:, 2])])
                metas.update(point_cloud_range=point_cloud_range)
            else:
                print(f"Voxel data of scene {sampled_scene} not found.")

        # all_images: [N, H, W, 3], all_extrins: [N, 4, 4], intrins: [N, 4, 4]
        return np.array(all_images), np.array(all_extrins), np.array(intrins), metas


# historical settings
base_classes = ['wall', 'ceiling', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
               'table', 'door', 'window', 'bookshelf', 'counter', 'desk',
               'curtain', 'refrigerator', 'television', 'whiteboard', 'toilet',
               'sink', 'bathtub', 'doorframe']
base_colors = np.array(
        [[176, 224, 230], [136, 184, 200], [255, 192, 203], [255, 228, 196],
         [255, 218, 185], [240, 230, 140], [255, 235, 205], [255, 250, 205],
         [240, 128, 128], [255, 99, 71],   [255, 69, 0],    [198, 255, 240],
         [210, 182, 198], [255, 165, 0],   [255, 215, 0],   [255, 250, 240],
         [255, 255, 224], [154, 205, 50],  [173, 216, 230], [236, 140, 56],
         [234, 211, 234]])

colors = defaultdict(lambda: np.random.randint(0, 256, (3)))
for i, color in enumerate(base_colors):
    colors[i] = color


class ScannetppDataset(torch.utils.data.Dataset):

    NAME = "scannetpp"
    ALL_LABELS = list(map(lambda x: x[0], load_from_jsonl(Path("scenecraft/data/scannetpp_utils/meta_data/enum_labels.jsonl"))[:128]))
    CLASSES = list(OrderedDict.fromkeys(base_classes + ALL_LABELS).keys())
    COLORS = colors

    # refer to preprocess of scannetpp
    image_dir = "resized_images_2"
    anno_masks_dir = "resized_anno_masks_2"
    transform = "transforms_2"

    coord_convention = "OpenCV"
    rgb_image_suffix = ".JPG"
    cat2label = {cat: label for label, cat in enumerate(CLASSES)}
    label2cat = {label: cat for cat, label in cat2label.items()}
    ori_h = 584
    ori_w = 876

    def __init__(self, raw_root_dir: str='data/scannetpp', root_dir: str='data/scannetpp_processed', split: str='train',
                 type: str="dataset", limit: int=None, ignore_label: int=-100, voxel_size: float=-1., **kwargs) -> None:
        super().__init__()
        # kwargs contains the data augmentation configs
        self.scene_ids = kwargs.pop("scene_ids", [])
        if isinstance(self.scene_ids, str):
            self.scene_ids = [self.scene_ids]
        self.kwargs = kwargs
        self.type=type
        self.root_dir = root_dir
        self.raw_root_dir = root_dir if raw_root_dir == '' else raw_root_dir
        self.limit = limit
        self.ignore_label = ignore_label
        self.voxel_size = voxel_size
        self.data_dir = os.path.join(root_dir, 'data')
        self.raw_data_dir = os.path.join(raw_root_dir, 'data')

        all_scenes = list(sorted(os.listdir(self.data_dir)))
        splits = ['train', 'val'] if split == 'all' else [split]
        self.split_available_scenes = []
        for split in splits:
            with open(SPLIT_PATHS['scannetpp'][split], 'r') as file:
                split_scene = [line.strip() for line in file.readlines()]
                self.split_available_scenes += list(filter(lambda x: x in all_scenes, split_scene))
        if self.scene_ids:
            assert set(self.scene_ids).issubset(set(self.split_available_scenes)), f"Given scene {self.scene_ids} not entirely exist in splits {split}."
            self.split_available_scenes = list(filter(lambda x: x in self.scene_ids, self.split_available_scenes))
        print(f"======> Load {len(self.split_available_scenes)} scenes from {splits} splits.")

        self.voxel_data_dir = None
        if self.voxel_size != -1:
            self.voxel_data_dir = os.path.join(root_dir, 'voxel_data', f'size{voxel_size}')
            if not (os.path.exists(self.voxel_data_dir) and (
                    (not self.scene_ids and len(os.listdir(self.voxel_data_dir)) == len(self.split_available_scenes)) or
                    (self.scene_ids and set(self.scene_ids).issubset(set(os.listdir(self.voxel_data_dir)))))
                ):
                os.makedirs(self.voxel_data_dir, exist_ok=True)
                print(f"Start voxelization, voxel size was set to {voxel_size}, data will be saved to {self.voxel_data_dir}.")
                instance_data_path = os.path.join(root_dir, 'scannetpp_instance_data')
                assert os.path.exists(instance_data_path), f"Instance data {instance_data_path} doesn't exist, " \
                                "run python scenecraft/data/scannetpp_utils/_batch_load_scannetpp_data.py to fix it."
                avaliable_scenes = self.split_available_scenes
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    world_size = torch.distributed.get_world_size()
                    rank = torch.distributed.get_rank()
                    num_scenes_per_device = len(self.split_available_scenes) // world_size
                    remainder = len(self.split_available_scenes) % world_size
                    start_idx = rank * num_scenes_per_device
                    end_idx = (rank + 1) * num_scenes_per_device + remainder * (rank == world_size - 1)
                    avaliable_scenes = self.split_available_scenes[start_idx : end_idx]
                for scene in tqdm(avaliable_scenes):
                    if os.path.exists(os.path.join(self.voxel_data_dir, f'{scene}.npy')):
                        print(f"Skipped scene {scene} when generating voxels.")
                        continue
                    points = np.load(os.path.join(instance_data_path, f'{scene}_vert.npy'))
                    labels = np.load(os.path.join(instance_data_path, f'{scene}_sem_label.npy'))
                    try:
                        voxels = transform_points_to_voxels(points, voxel_size=[voxel_size] * 3, labels=labels)
                        print(f"Get {len(voxels)} voxels in scene {scene}.")
                        np.save(os.path.join(self.voxel_data_dir, f'{scene}.npy'), voxels)
                    except Exception as e:
                        print(f"Skipped scene {scene} due to {e}.")

    def get_scene_image_ids(self, scene_id: str) -> List[str]:
        scene_path = self.get_scene_path(scene_id)
        image_dir = self.get_scene_image_path(scene_path)
        image_ids = list(filter(lambda x: x.endswith('.JPG'), os.listdir(image_dir)))
        image_ids = list(map(lambda x: x.split('.')[0], image_ids))
        return image_ids

    @staticmethod
    def get_relative_rgb_image_path(scene_id: str, id: str) -> str:
        return f'data/{scene_id}/dslr/resized_images_2/{id}.JPG'

    def get_rgb_image_path(self, scene_id: str, id: str) -> str:
        return os.path.join(self.root_dir, self.get_relative_rgb_image_path(scene_id, id))

    def get_scene_path(self, scene_id: str) -> str:
        return os.path.join(self.data_dir, scene_id)

    def get_raw_scene_path(self, scene_id: str) -> str:
        return os.path.join(self.raw_data_dir, scene_id)

    def get_scene_image_path(self, scene_path: str) -> str:
        return os.path.join(scene_path, 'dslr', self.image_dir)

    def __len__(self) -> int:
        return len(self.split_available_scenes)

    def _get_sample_data(self, intrin: npt.NDArray, metas: dict, elevation: float=90., # 90 means horizontal
                         frames: int=120, center: npt.NDArray=None) -> Sequence[npt.NDArray]:
        center = [4.082, 1.860, 1.256]
        elevation = np.radians(elevation)
        extrins = []
        for _ in range(frames):
            azimuth = np.random.uniform(0, 2*np.pi)
            R_azimuth = np.array([[np.cos(azimuth), -np.sin(azimuth), 0],
                                  [np.sin(azimuth), np.cos(azimuth), 0],
                                  [0, 0, 1]])
            R_elevation = np.array([[1, 0, 0],
                                    [0, np.cos(elevation), -np.sin(elevation)],
                                    [0, np.sin(elevation), np.cos(elevation)]])
            R = np.dot(R_azimuth, R_elevation)

            extrin = np.eye(4)
            extrin[:3, :3] = R
            extrin[:3, 3] = center

            extrins.append(extrin)

        extrins = np.array(extrins)
        # OpenGL to OpenCV
        transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        extrins[..., :3, :3] @= transform

        intrins = np.broadcast_to(intrin, (len(extrins), 4, 4))
        metas.update(selected_ids=[f"center_{str(i).zfill(3)}" for i in range(frames)])
        return np.zeros((frames,)), np.array(extrins), np.array(intrins), metas

    def _get_dataset_data(self, sampled_scene: str, intrin: npt.NDArray, metas: dict, selected: List[int] = None) -> Sequence[npt.NDArray]:
        scene_path = self.get_scene_path(sampled_scene)
        raw_scene_path = self.get_raw_scene_path(sampled_scene)
        image_dir = self.get_scene_image_path(scene_path)

        colmap = []
        with open(os.path.join(raw_scene_path, 'dslr', 'colmap', 'images.txt'), 'rt') as f:
            for line in f:
                col = line.strip().split()
                if col[0] == '#' or len(col) > 10: continue
                colmap.append(col)
        colmap = np.array(sorted(colmap, key=lambda x: x[9])) # column 9 is the image file name
        quaternions = colmap[:, 1:5].astype(np.float32)
        rots = np.array([qvec2rotmat(qvec) for qvec in quaternions])
        trans = colmap[:, 5:8].astype(np.float32)
        all_extrins = np.linalg.inv(format_poses(rots, trans))
        all_images = sorted(glob.glob(os.path.join(image_dir, f'*{self.rgb_image_suffix}')))
        # Set limit
        N = len(all_images)
        limit = N if self.limit is None or self.limit >= N else self.limit
        replace = False if limit < N else True
        if selected is None:
            selected = np.random.choice(N, limit, replace=replace).astype(np.int32)
        splitor = '/' if platform.system() == 'Linux' else '\\'
        selected_ids = [all_images[i].split(splitor)[-1].removesuffix(self.rgb_image_suffix) for i in selected]
        all_images = [np.array(Image.open(all_images[i])) for i in selected]
        all_extrins = np.array(all_extrins)[selected]
        intrins = np.broadcast_to(intrin, (len(all_extrins), 4, 4))
        metas.update(selected_ids=selected_ids)
        # all_images: [N, H, W, 3], all_extrins: [N, 4, 4], intrins: [N, 4, 4]
        return np.array(all_images), np.array(all_extrins), np.array(intrins), metas

    def __getitem__(self, idx: int, selected: List[int] = None) -> Sequence[npt.NDArray]:
        sampled_scene = self.split_available_scenes[idx]
        scene_path = self.get_scene_path(sampled_scene)
        raw_scene_path = self.get_raw_scene_path(sampled_scene)
        # use transformation after undistortion
        transform = json.load(open(os.path.join(scene_path, 'dslr', 'nerfstudio', f'{self.transform}.json'), 'r'))
        fl_x, fl_y, cx, cy = transform['fl_x'], transform['fl_y'], transform['cx'], transform['cy']
        intrin = np.array([[fl_x, 0, cx, 0], [0, fl_y, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # meta infos
        metas = dict(scene_id=sampled_scene, voxel_size=self.voxel_size)
        segments_anno_file = self.kwargs.get('segments_anno_file', 'segments_anno.json')
        segments_anno = json.load(open(os.path.join(raw_scene_path, 'scans', segments_anno_file)))
        sample_boxes, sample_labels = [], []
        for seg in segments_anno['segGroups']:
            label = self.cat2label.get(seg['label'], self.ignore_label)
            centroid = seg['obb']['centroid']
            min = seg['obb']['min']
            max = seg['obb']['max']
            dxyz = (np.array(max) - np.array(min)).tolist()
            sample_labels.append(label)
            sample_boxes.append(centroid + dxyz)
        sample_labels = np.array(sample_labels)
        sample_boxes = np.array(sample_boxes)
        # filter the ignored labels
        sample_boxes = sample_boxes[sample_labels != self.ignore_label]
        sample_labels = sample_labels[sample_labels != self.ignore_label]
        if len(sample_labels) > 0:
            metas.update(obj_bbox=sample_boxes, obj_class=sample_labels)

        # voxels
        if self.voxel_data_dir is not None:
            voxel_data_path = os.path.join(self.voxel_data_dir, f'{sampled_scene}.npy')
            if os.path.exists(voxel_data_path):
                voxels = np.load(voxel_data_path)
                metas.update(voxels=voxels)
                points_path = os.path.join(self.root_dir, 'scannetpp_instance_data', f'{sampled_scene}_vert.npy')
                points = np.load(points_path)
                point_cloud_range = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]),
                                              np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
                metas.update(point_cloud_range=point_cloud_range)
            else:
                print(f"Voxel data of scene {sampled_scene} not found.")

        if self.type == "sample":
            raise RuntimeError("Not supported!")
        return self._get_dataset_data(sampled_scene, intrin, metas, selected)


semantic_label_descs_file = "thirdparty/ml-hypersim/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv"
semantic_label_descs = pd.read_csv(semantic_label_descs_file)
hypersim_colors_ = np.stack([semantic_label_descs[' semantic_color_r '].values,
                             semantic_label_descs[' semantic_color_g '].values,
                             semantic_label_descs[' semantic_color_b'].values], axis=-1)
hypersim_colors = defaultdict(lambda: np.random.randint(0, 256, (3)))
for i, color in enumerate(hypersim_colors_):
    hypersim_colors[i + 1] = color


# ADE20K labels (https://huggingface.co/datasets/huggingface/label-files/blob/main/ade20k-id2label.json)
nyu40_to_ade20k = [-1, 0, 3, 10, 7, 19, 23, 15, 14, 8, 62, -1, 45, 63, 33, 24, 18, 33, 57, 27, 28, -1, 5, 67, 50,
                   89, -1, 81, 18, 41, 43, 12, 15, 65, 47, 36, 37, 115, -1, -1, -1]
scannetpp_to_nyu40 = [1, 22, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 24, 25, 30, 33, 34, 36, 8]


class HypersimDataset(torch.utils.data.Dataset):

    NAME = "hypersim"
    CLASSES = list(map(lambda x: x.strip(), open(Path("scenecraft/data/hypersim_utils/meta_data/nyu40id.txt"), "r").readlines()))
    COLORS = hypersim_colors

    coord_convention = "OpenGL"
    rgb_image_suffix = ".hdf5"
    cat2label = {cat: label for label, cat in enumerate(CLASSES)}
    label2cat = {label: cat for cat, label in cat2label.items()}

    ori_h = 768
    ori_w = 1024

    def __init__(self, raw_root_dir: str='', root_dir: str='./data/hypersim', **kwargs) -> None:
        super().__init__()
        # kwargs contains the data augmentation configs
        self.scene_ids = kwargs.pop("scene_ids", [])
        if isinstance(self.scene_ids, str):
            self.scene_ids = [self.scene_ids]
        self.kwargs = kwargs
        self.data_dir = root_dir
        self.raw_data_dir = root_dir if raw_root_dir == '' else raw_root_dir

        all_scenes = list(sorted(os.listdir(self.data_dir)))
        all_scenes = list(filter(lambda x: x.startswith('ai'), all_scenes))
        with open(SPLIT_PATHS['hypersim']['train'], 'r') as file:
            avail_scene = [line.strip() for line in file.readlines()]
        self.split_available_scenes = list(filter(lambda x: x in avail_scene, all_scenes)).sort()
        if self.scene_ids:
            assert set(self.scene_ids).issubset(set(self.split_available_scenes)), f"Given scene {self.scene_ids} not entirely exist in splits {split}."
            self.split_available_scenes = list(filter(lambda x: x in self.scene_ids, self.split_available_scenes))
        assert len(self.split_available_scenes) > 0, "No scenes loaded!"
        print(f"======> Load {len(self.split_available_scenes)} scenes.")

        # load intrinsic parameters (refer to https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/jupyter/00_projecting_points_into_hypersim_images.ipynb)
        metadata_camera_parameters_csv_file = "scenecraft/data/hypersim_utils/meta_data/metadata_camera_parameters.csv"
        self.df_camera_parameters = pd.read_csv(metadata_camera_parameters_csv_file, index_col="scene_name")
        self.image_height = self.df_camera_parameters["settings_output_img_height"].values[0]
        self.image_width = self.df_camera_parameters["settings_output_img_width"].values[0]
        camera_trajectory_csv_file = "thirdparty/ml-hypersim/evermotion_dataset/analysis/metadata_camera_trajectories.csv"
        self.camera_trajectory = pd.read_csv(camera_trajectory_csv_file)
        self.scene_types = dict(zip(self.camera_trajectory['Animation'].values.tolist(), self.camera_trajectory['Scene type'].values.tolist()))

    def get_scene_image_ids(self, scene_id: str) -> List[str]:
        scene_path = os.path.join(self.data_dir, scene_id)
        detail_path = os.path.join(scene_path, "_detail")
        camera = os.listdir(detail_path)[0] # use the first camera
        image_path = os.path.join(scene_path, "images", f"scene_{camera}_final_preview")
        image_files = list(filter(lambda fname: fname.endswith('.color.jpg'), os.listdir(image_path)))
        image_ids = list(map(lambda fname: fname.removesuffix('.color.jpg'), image_files))
        return image_ids

    @staticmethod
    def get_relative_rgb_image_path(scene_id: str, id: str) -> str:
        # NOTE: hard code
        scene_path = os.path.join("data/hypersim", scene_id)
        detail_path = os.path.join(scene_path, "_detail")
        camera = os.listdir(detail_path)[0] # use the first camera
        return f'{scene_id}/images/scene_{camera}_final_preview/{id}.gamma.jpg'

    @staticmethod
    def get_rgb_image_path(scene_id: str, id: str) -> str:
        # NOTE: hard code
        return os.path.join("data/hypersim", HypersimDataset.get_relative_rgb_image_path(scene_id, id))

    def get_scene_path(self, scene_id: str) -> str:
        return os.path.join(self.data_dir, scene_id)

    def get_raw_scene_path(self, scene_id: str) -> str:
        return os.path.join(self.raw_data_dir, scene_id)

    def get_scene_image_path(self, scene_path: str) -> str:
        return os.path.join(scene_path, 'images')

    @staticmethod
    def get_rays(i, M_cam_from_uv, camera_orientations, camera_positions, return_tensors="pt", **kwargs) -> Sequence[Union[torch.Tensor, npt.NDArray]]:
        H, W = HypersimDataset.ori_h, HypersimDataset.ori_w
        if not isinstance(M_cam_from_uv, torch.Tensor):
            M_cam_from_uv = torch.Tensor(M_cam_from_uv)
            camera_orientations = torch.Tensor(camera_orientations)
            camera_positions = torch.Tensor(camera_positions)

        device = M_cam_from_uv.device
        M_cam_from_uv = matrix(M_cam_from_uv.cpu().numpy())
        camera_orientation = camera_orientations[i].cpu().numpy()
        camera_position = camera_positions[i].cpu().numpy()

        u_min, u_max, v_min, v_max = -1., 1., -1., 1.
        half_du = 0.5 * (u_max - u_min) / W
        half_dv = 0.5 * (v_max - v_min) / H

        u, v = meshgrid(linspace(u_min + half_du, u_max - half_du, W),
                        linspace(v_min + half_dv, v_max - half_dv, H)[::-1])

        uvs_2d = dstack((u, v, np.ones_like(u)))
        P_uv   = matrix(uvs_2d.reshape(-1, 3)).T

        P_world = camera_orientation * M_cam_from_uv * P_uv
        rays_d  = P_world.T.A
        rays_o  = ones_like(rays_d) * camera_position

        rays_d = rays_d.reshape(H, W, 3).transpose(1, 0, 2)
        rays_o = rays_o.reshape(H, W, 3).transpose(1, 0, 2)
        if return_tensors == "pt":
            rays_d = torch.Tensor(rays_d).to(device)
            rays_o = torch.Tensor(rays_o).to(device)
        return rays_o, rays_d

    def __len__(self) -> int:
        return len(self.split_available_scenes)

    def _get_camera_extrin_from_hdf5(self, camera_position_world: npt.NDArray, R_world_from_cam: npt.NDArray) -> npt.NDArray:
        t_world_from_cam = np.matrix(camera_position_world).T
        R_cam_from_world = np.matrix(R_world_from_cam).T
        t_cam_from_world = -R_cam_from_world * t_world_from_cam
        M_cam_from_world = np.matrix(np.block([[R_cam_from_world, t_cam_from_world], [np.matrix(np.zeros(3)), 1.0]]))
        return np.linalg.inv(M_cam_from_world)

    def __getitem__(self, idx: int) -> Sequence[npt.NDArray]:
        sampled_scene = self.split_available_scenes[idx] # e.g. "ai_037_002"
        scene_path = os.path.join(self.data_dir, sampled_scene)
        detail_path = os.path.join(scene_path, "_detail")
        camera = os.listdir(detail_path)[0] # use the first camera
        camera_path = os.path.join(detail_path, camera)
        camera_positions_hdf5_file    = os.path.join(camera_path, "camera_keyframe_positions.hdf5")
        camera_orientations_hdf5_file = os.path.join(camera_path, "camera_keyframe_orientations.hdf5")
        # images
        all_images = []
        image_path = os.path.join(scene_path, "images", f"scene_{camera}_final_preview")
        geometry_path = os.path.join(scene_path, "images", f"scene_{camera}_geometry_hdf5")
        image_files = list(filter(lambda fname: fname.endswith('.color.jpg'), os.listdir(image_path)))
        semantic_files = list(filter(lambda fname: fname.endswith('.semantic.hdf5'), os.listdir(geometry_path)))
        all_images = np.stack([np.array(Image.open(os.path.join(image_path, image_file))) for image_file in image_files]) # N×H×W
        selected_ids = list(map(lambda image_file: image_file[:-10], image_files))
        semantic_images = np.stack([np.array(h5py.File(os.path.join(geometry_path, semantic_file))['dataset'][:]).astype(np.uint8) for semantic_file in semantic_files])
        # camera parameters
        # for extrinsics, they're from hdf5 files
        # for intrinsics, refer to https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_bounding_box.py#L124
        fov_x = np.pi / 3.
        fov_y = 2.0 * np.arctan(self.image_height * np.tan(fov_x / 2.) / self.image_width)
        fx = self.image_width / (2 * np.tan(fov_x / 2))
        fy = self.image_height / (2 * np.tan(fov_y / 2)) # equal to fx
        cx = float(self.image_width / 2.)
        cy = float(self.image_height / 2.)
        intrin = np.array(format_intrinsic(fx, fy, cx, cy))
        df_ = self.df_camera_parameters.loc[sampled_scene]
        M_cam_from_uv = matrix([[ df_["M_cam_from_uv_00"], df_["M_cam_from_uv_01"], df_["M_cam_from_uv_02"] ],
                                [ df_["M_cam_from_uv_10"], df_["M_cam_from_uv_11"], df_["M_cam_from_uv_12"] ],
                                [ df_["M_cam_from_uv_20"], df_["M_cam_from_uv_21"], df_["M_cam_from_uv_22"] ]])
        metadata_scene_file = os.path.join(scene_path, "_detail", "metadata_scene.csv")
        metadata_scene = pd.read_csv(metadata_scene_file, index_col="parameter_name")
        meters_per_asset_unit = metadata_scene.loc["meters_per_asset_unit"].values[0]
        with h5py.File(camera_positions_hdf5_file,    "r") as f: camera_positions    = f["dataset"][:]
        with h5py.File(camera_orientations_hdf5_file, "r") as f: camera_orientations = f["dataset"][:]
        camera_positions = camera_positions * meters_per_asset_unit
        # filter cameras according to images
        if len(image_files) != camera_positions.shape[0]:
            image_ids = list(set(map(lambda image_file: int(image_file.split('.')[1]), image_files)))
            valid_ids = np.array(image_ids)
            camera_positions = camera_positions[valid_ids]
            camera_orientations = camera_orientations[valid_ids]
        all_extrins = np.array(list(map(self._get_camera_extrin_from_hdf5, camera_positions, camera_orientations)))
        intrins = np.broadcast_to(intrin, (len(all_extrins), 4, 4))
        # meta infos
        metas = dict(scene_id=sampled_scene, selected_ids=selected_ids, boxes_oriented=True,
                     semantic_images=semantic_images, M_cam_from_uv=M_cam_from_uv,
                     camera_orientations=camera_orientations, camera_positions=camera_positions)
        mesh_path = os.path.join(scene_path, "_detail", "mesh")
        bounding_box_object_aligned_2d_extents_file = os.path.join(mesh_path, "metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5")
        bounding_box_object_aligned_2d_orientation_file = os.path.join(mesh_path, "metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5")
        bounding_box_object_aligned_2d_positions_file = os.path.join(mesh_path, "metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5")
        with h5py.File(bounding_box_object_aligned_2d_extents_file,     "r") as f: bounding_box_object_aligned_2d_extents     = f["dataset"][:] # M×3
        with h5py.File(bounding_box_object_aligned_2d_orientation_file, "r") as f: bounding_box_object_aligned_2d_orientation = f["dataset"][:] # M×3×3
        with h5py.File(bounding_box_object_aligned_2d_positions_file,   "r") as f: bounding_box_object_aligned_2d_positions   = f["dataset"][:] # M×3
        frame_depth_files = list(filter(lambda fname: fname.endswith("depth_meters.hdf5"), os.listdir(geometry_path)))
        frame_depths = np.array(list(map(lambda fname: h5py.File(os.path.join(geometry_path, fname), "r")["dataset"][:], frame_depth_files)), dtype=np.float32)
        mask = np.isnan(frame_depths)
        frame_depths[~mask] = frame_depths[~mask] # [N, H, W] NOTE: do not multiply with `meters_per_asset_unit`
        bounding_box_object_aligned_2d_extents = np.array(bounding_box_object_aligned_2d_extents) * meters_per_asset_unit
        bounding_box_object_aligned_2d_orientation = np.array(bounding_box_object_aligned_2d_orientation)
        bounding_box_object_aligned_2d_positions = np.array(bounding_box_object_aligned_2d_positions) * meters_per_asset_unit
        anno_path = os.path.join("thirdparty/ml-hypersim/evermotion_dataset/scenes", sampled_scene, "_detail/mesh")
        si_file = os.path.join(anno_path, "mesh_objects_si.hdf5")
        sii_file = os.path.join(anno_path, "mesh_objects_sii.hdf5")
        with h5py.File(si_file, "r")    as f:  si    = f['dataset'][:]
        with h5py.File(sii_file, "r")   as f: sii    = f['dataset'][:]
        bounding_box_object_aligned_2d_labels = list(map(lambda i: si[sii == i][0] if np.any(sii == i) and i != 0 else -1, range(len(bounding_box_object_aligned_2d_positions))))
        bounding_box_object_aligned_2d_labels = np.array(bounding_box_object_aligned_2d_labels)
        mask = np.logical_or(np.any(np.isinf(bounding_box_object_aligned_2d_extents), axis=-1), bounding_box_object_aligned_2d_labels == -1)
        bounding_box_object_aligned_2d_extents = bounding_box_object_aligned_2d_extents[~mask]
        bounding_box_object_aligned_2d_orientation = bounding_box_object_aligned_2d_orientation[~mask]
        bounding_box_object_aligned_2d_positions = bounding_box_object_aligned_2d_positions[~mask]
        bounding_box_object_aligned_2d_labels = bounding_box_object_aligned_2d_labels[~mask]
        metas.update(frame_depths=frame_depths,
                     bbox_extents=bounding_box_object_aligned_2d_extents,
                     bbox_orientations=bounding_box_object_aligned_2d_orientation,
                     bbox_positions=bounding_box_object_aligned_2d_positions)
        obj_bbox = np.zeros((len(bounding_box_object_aligned_2d_extents), 6))
        obj_bbox[:, 3:] = bounding_box_object_aligned_2d_extents
        metas.update(obj_bbox=obj_bbox, obj_class=bounding_box_object_aligned_2d_labels)
        # all_images: [N, H, W, 3], all_extrins: [N, 4, 4], intrins: [N, 4, 4]
        return np.array(all_images), np.array(all_extrins), np.array(intrins), metas


class CustomSceneDataset(torch.utils.data.Dataset):

    NAME = "custom"
    # scannet++ convention
    ALL_LABELS = list(map(lambda x: x[0], load_from_jsonl(Path("scenecraft/data/scannetpp_utils/meta_data/enum_labels.jsonl"))[:128]))
    CLASSES_SCANNETPP = list(OrderedDict.fromkeys(base_classes + ALL_LABELS).keys())
    COLORS_SCANNETPP = colors
    cat2label_scannetpp = {cat: label for label, cat in enumerate(CLASSES_SCANNETPP)}
    label2cat_scannetpp = {label: cat for cat, label in cat2label_scannetpp.items()}

    # hypersim convention
    CLASSES_HYPERSIM = list(map(lambda x: x.strip(), open(Path("scenecraft/data/hypersim_utils/meta_data/nyu40id.txt"), "r").readlines()))
    COLORS_HYPERSIM = hypersim_colors
    cat2label_hypersim = {cat: label for label, cat in enumerate(CLASSES_HYPERSIM)}
    label2cat_hypersim = {label: cat for cat, label in cat2label_hypersim.items()}

    coord_convention = "OpenGL"
    CLASSES = CLASSES_HYPERSIM
    COLORS = COLORS_HYPERSIM
    cat2label = cat2label_hypersim
    label2cat = label2cat_hypersim

    ori_h = 768
    ori_w = 1024

    def __init__(self, custom_scene_id: str, fov: int = 60, height: int = 768, width: int = 1024, **kwargs) -> None:
        super().__init__()
        # kwargs contains the data augmentation configs
        self.scene_ids = kwargs.pop("scene_ids", [])
        if isinstance(self.scene_ids, str):
            self.scene_ids = [self.scene_ids]
        self.kwargs = kwargs
        self.scene_id = custom_scene_id
        self.scene_path = os.path.join("data/custom", custom_scene_id)

        # layouts
        bboxes_file = os.path.join(self.scene_path, "layout.json")
        self.bboxes = load_from_json(Path(bboxes_file))
        CONSOLE.print(f"======> Load {len(self.bboxes['bboxes'])} bounding boxes from scene {custom_scene_id}.")

        # given existing cameras!
        # could be useful to scene editing
        if kwargs.get("semantic_images_path", None) is not None:
            intrins = []
            all_extrins = []
            npz_files = list(sorted(filter(lambda f: f.endswith(".npz"), os.listdir(kwargs["semantic_images_path"]))))
            for npz_file in npz_files:
                data = np.load(os.path.join(kwargs["semantic_images_path"], npz_file))
                intrins.append(data["intrin"].reshape(4, 4))
                all_extrins.append(data["extrin"].reshape(4, 4))
            self.intrins = np.stack(intrins)
            self.all_extrins = np.stack(all_extrins)

            return

        # cameras
        camera_files = list(sorted(filter(lambda fname: fname.startswith("cameras") and fname.endswith(".json"), os.listdir(self.scene_path))))
        layout_file = os.path.join(self.scene_path, "layout.json")
        self.cameras = [load_from_json(Path(self.scene_path) / camera_file) for camera_file in camera_files]
        self.layout = load_from_json(Path(layout_file))
        # TODO: more elegant way
        if height is not None and width is not None:
            self.render_size = (height, width)
        else:
            self.render_size = (self.cameras[0]['render_height'], self.cameras[0]['render_width'])
        for i, camera_file in enumerate(camera_files):
            if "fix" in camera_file:
                self.cameras[i]["camera_path"] = [
                    {"camera_to_world": np.array(list(map(lambda x: float(x), key_camera["matrix"].strip("[]").split(",")))).reshape(4, 4).transpose().reshape(-1).tolist(),
                     "fov": key_camera["fov"],
                     "aspect": key_camera["aspect"]}
                    for key_camera in self.cameras[i]["keyframes"]
                ]
        self.prompts = None
        self.camera_paths = list(chain.from_iterable([camera['camera_path'] for camera in self.cameras]))
        if "prompt" in self.cameras[0]:
            self.prompts = list(chain.from_iterable([[camera["prompt"]] * len(camera["camera_path"]) for camera in self.cameras]))
            assert len(self.prompts) == len(self.camera_paths), f"Got {len(self.prompts)} promtps and {len(self.camera_paths)} camera frames"
        self.ids = [f"random_{i:03d}" for i in range(len(self.camera_paths))]
        CONSOLE.print(f"======> Load {len(camera_files)} paths and {len(self.camera_paths)} frames from scene {custom_scene_id} with render size = {self.render_size}.")

        # cameras NOTE extrin in OpenGL convention
        # https://github.com/nerfstudio-project/nerfstudio/blob/ae6c46cfeaebbe28f9cd48fa986755e27b5f0ae2/nerfstudio/viewer/utils.py#L60
        # Fov from nerfstudio is vertical_fov: https://github.com/nerfstudio-project/nerfstudio/blob/3f395466534050342f95d4fb837f27ee3cc1d688/nerfstudio/scripts/blender/nerfstudio_blender.py#L68-L88
        if fov is None:
            fov = self.camera_paths[0]["fov"]
        fov = np.radians(fov)
        focal_length = self.render_size[0] / (2 * np.tan(fov / 2))
        cx = float(self.render_size[1] / 2.)
        cy = float(self.render_size[0] / 2.)
        intrin = np.array(format_intrinsic(focal_length, focal_length, cx, cy))
        self.all_extrins = np.stack(list(map(lambda x: np.array(x["camera_to_world"]).reshape(4, 4), self.camera_paths)))

        # generate center poses
        # self.generate_center_poses()
        self.intrins = np.broadcast_to(intrin, (len(self.all_extrins), 4, 4))

    @classmethod
    def switch_to_scannetpp_format(cls):
        cls.CLASSES = cls.CLASSES_SCANNETPP
        cls.COLORS = cls.COLORS_SCANNETPP
        cls.cat2label = cls.cat2label_scannetpp
        cls.label2cat = cls.label2cat_scannetpp

    def generate_center_poses(self, elevation: float=90., frames: int=120):
        center = self.all_extrins[0][:3, 3]
        elevation = np.radians(elevation)
        extrins = []
        for _ in range(frames):
            azimuth = np.random.uniform(0, 2*np.pi)
            R_azimuth = np.array([[np.cos(azimuth), -np.sin(azimuth), 0],
                                  [np.sin(azimuth), np.cos(azimuth), 0],
                                  [0, 0, 1]])
            R_elevation = np.array([[1, 0, 0],
                                    [0, np.cos(elevation), -np.sin(elevation)],
                                    [0, np.sin(elevation), np.cos(elevation)]])
            R = np.dot(R_azimuth, R_elevation)

            extrin = np.eye(4)
            extrin[:3, :3] = R
            extrin[:3, 3] = center

            extrins.append(extrin)

        # NOTE in OpenGL format
        extrins = np.array(extrins)
        self.all_extrins = np.concatenate([self.all_extrins, extrins])
        self.ids += [f"center_{i:03d}" for i in range(len(extrins))]

    def __len__(self) -> int:
        return len(self.all_extrins)

    def __getitem__(self, idx: int) -> Sequence[npt.NDArray]:
        all_extrins = self.all_extrins[idx : idx + 1]
        intrins = self.intrins[idx : idx + 1]
        all_images = np.zeros((all_extrins.shape[0], *self.render_size, 3))
        sample_boxes = np.array(self.bboxes["bboxes"]) # [N, 6]
        sample_boxes = sample_boxes[:, [0, 1, 2, 4, 3, 5]]
        sample_labels = np.array(list(map(int, self.bboxes["labels"])))
        # convert scannet++ id to hypersim id
        # sample_labels = np.array(scannetpp_to_nyu40)[sample_labels.astype(np.int32)]
        metas = dict(customed=True,
                     scene_id=self.scene_id, selected_ids=self.ids[idx : idx + 1],
                     obj_bbox=sample_boxes, obj_class=sample_labels)
        # all_images: [N, H, W, 3], all_extrins: [N, 4, 4], intrins: [N, 4, 4]
        return np.array(all_images), np.array(all_extrins), np.array(intrins), metas


RAW_DATASETS: Dict[str, Union[ScannetDataset, ScannetppDataset, HypersimDataset]] = {
    'scannet': ScannetDataset, 'scannetpp': ScannetppDataset, 'hypersim': HypersimDataset, 'custom': CustomSceneDataset}

SCALE_H: Dict[str, int] = {'scannetpp': 512, 'hypersim': 576}


# NOTE: Inherit from `RandomSampler`. Don't use `RandomSampler` directly or it will be forced back to `SeedableRandomSampler`
# after `acceleactor.prepare` and the behavior will be same as normal RandomSampler if distributed is used.
class CustomSampler(Sampler[int]):

    def __init__(self, dataset: Dataset, batch_size: int, replacement: bool = False,
                 num_samples: int = None, generator = None, **kwargs) -> None:
        self.sequential_input = kwargs.pop("sequential_input", False)
        self.data_source = dataset
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.batch_size = batch_size
        self.num_scenes = dataset.num_scenes
        self.scene_sample_counts_list = dataset.scene_sample_counts_list
        assert sum(self.scene_sample_counts_list) == self.num_samples, \
            f"number of samples in dataset {self.num_samples} isn't equal to {sum(self.scene_sample_counts_list)}."

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # replacement is False.
        assert self.num_samples == n, f"num_samples should be equal to n."

        if not self.sequential_input:
            indices = [list(range(int(sum(self.scene_sample_counts_list[:i])), int(sum(self.scene_sample_counts_list[:i + 1]))))
                       for i in range(self.num_scenes)]
            indices = [random.sample(scene, len(scene)) for scene in indices]
            indices = [scene_indices[:(len(scene_indices) // self.batch_size * self.batch_size)] for scene_indices in indices]
            indices = list(chain(*indices))
            num_batches = int(len(indices) / self.batch_size)
            indices = [indices[int(i * self.batch_size) : int((i + 1) * self.batch_size)] for i in range(num_batches)]
            batch_indices = list(range(num_batches))
            random.shuffle(batch_indices)
            indices = [indices[batch_indice] for batch_indice in batch_indices]
            indices = list(chain(*indices))

        # each batch contains multiviews which are neighboring views
        else:
            acc = [sum(self.scene_sample_counts_list[:i]) for i in range(self.num_scenes + 1)]
            left = int(self.batch_size * 1)
            indices = [list(range(sum(self.scene_sample_counts_list[:i]), sum(self.scene_sample_counts_list[:i + 1])))
                       for i in range(self.num_scenes)]
            indices = [scene_indices[:(len(scene_indices) // self.batch_size * self.batch_size)] for scene_indices in indices]
            indices = list(chain(*indices))
            random.shuffle(indices)
            num_batches = n // self.batch_size
            curr_views = indices[:num_batches]

            def get_batch_views(global_idx):
                scene_idx = bisect.bisect_right(acc, global_idx) - 1
                start_idx = max(acc[scene_idx], min(global_idx - left, acc[scene_idx] + self.scene_sample_counts_list[scene_idx] - 1 - self.batch_size * 2))
                end_idx = min(start_idx + self.batch_size * 2, acc[scene_idx] + self.scene_sample_counts_list[scene_idx] - 1)
                batch_idx = random.sample(list(range(start_idx, end_idx)) + 
                                            (random.choices(list(range(start_idx, end_idx)), k=self.batch_size - (end_idx - start_idx))
                                            if self.batch_size > (end_idx - start_idx) else []), self.batch_size)
                return batch_idx

            indices = list(chain(*map(lambda x: get_batch_views(x), curr_views)))

        yield from indices

    def __len__(self) -> int:
        return self.num_samples
