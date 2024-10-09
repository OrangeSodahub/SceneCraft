import argparse
import functools
import glob
import json
import os
import sys
import copy
import numpy as np
import yaml
import torch
import mediapy as media
import open3d as o3d
from dataclasses import dataclass, field
from numpy import typing as npt
from functools import partial
from pathlib import Path
from collections import defaultdict
from concurrent import futures
from itertools import chain
from typing import Sequence, Union, List, Literal, Callable, Optional, Tuple
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, default_collate, ConcatDataset
from tqdm import tqdm
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, \
                          TimeElapsedColumn, TimeRemainingColumn
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.scripts.render import BaseRender
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.scripts.exporter import ExportPointCloud as _ExportPointCloud, ExportTSDFMesh as _ExportTSDFMesh
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import get_mesh_from_filename

from scenecraft.data.dataset import RAW_DATASETS, SPLIT_PATHS
from scenecraft.data.datamanager import SceneCraftDataManagerConfig
from scenecraft.pipeline import SceneCraftPipeline
from scenecraft.trainer import SceneCraftTrainerConfig
from scenecraft.utils import (get_corner_bbox, draw_bbox_on_image, colorize_depth, opengl_to_opencv,
                              project_3d_to_2d, seed_everything, setup_logger, generate_point_cloud)


logger = None


# to handle the string data
def collate_fn(save_path, save_scene_names, batch):
    view_batched_meta_info_keys = ['selected_ids', 'semantic_images', 'camera_orientations',
                                   'camera_positions', 'frame_depths', ]

    new_batch, scene_ids, image_ids = [], [], []
    for _batch in batch:
        images, extrins, intrins, metas = _batch
        if save_path is not None:
            assert os.path.exists(save_path), f"{save_path} does not exist."
            scene_save_path = os.path.join(save_path, metas['scene_id']) \
                              if save_scene_names else save_path
            remains = [True] * images.shape[0]
            for i, image_id in enumerate(metas['selected_ids']):
                if os.path.exists(os.path.join(scene_save_path, f"{image_id}.png")) and \
                   os.path.exists(os.path.join(scene_save_path, f"{image_id}.npz")):
                    print(f"Skipped sample {image_id} of scene {metas['scene_id']} since already exist.")
                    remains[i] = False
            for key in view_batched_meta_info_keys:
                if key in metas:
                    if isinstance(metas[key], list):
                        metas[key] = [metas[key][i] for i in range(len(metas[key])) if remains[i]]
                    if isinstance(metas[key], np.ndarray):
                        metas[key] = metas[key][remains]
            images = images[remains]
            extrins = extrins[remains]
            intrins = intrins[remains]
            print(f"Skipped {remains.count(False)} samples in total.")
        scene_ids.append(metas.pop('scene_id'))
        image_ids.append(metas.pop('selected_ids'))
        new_batch.append((images, extrins, intrins, metas,))
    return default_collate(new_batch), scene_ids, image_ids


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = 1 << 16,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[SceneCraftTrainerConfig], SceneCraftTrainerConfig]] = None,
) -> Tuple[SceneCraftTrainerConfig, SceneCraftPipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config: SceneCraftTrainerConfig = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    # TODO: change the load_dir
    # TODO: support download model from huggingface
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline: SceneCraftPipeline = config.pipeline.setup(device=device, is_training=False, test_mode=test_mode)
    pipeline.eval()

    # load checkpointed information
    # `pipeline` will only load the nerf model while the diffusion won't be loaded.
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step

    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["meta_infos"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}.")

    return config, pipeline, load_path, load_step


@dataclass
class DatasetRender(BaseRender):
    """Render all images in the dataset."""

    scene_id: str = "0a7cc12c0e"

    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""

    def render_scene(self):

        def update_config(config: SceneCraftTrainerConfig) -> SceneCraftTrainerConfig:
            data_manager_config: SceneCraftDataManagerConfig = config.pipeline.datamanager
            setattr(data_manager_config, "camera_res_scale_factor", 1.)
            setattr(data_manager_config, "guide_camera_res_scale_factor", 1.)
            return config

        _, pipeline, _, _ = eval_setup(self.load_config,
                                            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
                                            test_mode="inference",
                                            update_config_callback=update_config)

        with Progress(
            TextColumn(f":movie_camera: Rendering scene {self.scene_id} :movie_camera:"),
            BarColumn(),
            TaskProgressColumn(
                text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                show_speed=True,
            ),
            ItersPerSecColumn(suffix="fps"),
            TimeRemainingColumn(elapsed_when_finished=False, compact=False),
            TimeElapsedColumn(),
        ) as progress:
            for _ in progress.track(range(len(pipeline.datamanager.eval_dataset))):
                # no grad
                outputs = pipeline.get_inference()

                for output_name in list(outputs.keys()):
                    if output_name in ["image_idx", "image_id", "accumulation", "scale_factor"]: continue

                    is_depth = "depth" in output_name

                    image_id = outputs["image_id"][0].replace(".", "_")
                    output_path: Path = self.output_path / output_name / image_id
                    output_path.parent.mkdir(exist_ok=True, parents=True)

                    output_image = outputs[output_name]
                    if is_depth:
                        # Divide by the dataparser scale factor
                        output_image.div_(pipeline.datamanager.dataparser_outputs.dataparser_scale)

                    # Map to color spaces / numpy
                    if is_depth:
                        output_image = colormaps.apply_depth_colormap(
                                output_image, accumulation=outputs["accumulation"],
                                near_plane=self.depth_near_plane,
                                far_plane=self.depth_far_plane,
                                colormap_options=self.colormap_options).cpu().numpy()
                    else:
                        output_image = output_image.cpu().numpy()

                    # Save to file
                    if self.image_format == "png":
                        media.write_image(output_path.with_suffix(".png"), output_image, fmt="png")
                    elif self.image_format == "jpeg":
                        media.write_image(
                            output_path.with_suffix(".jpg"), output_image, fmt="jpeg", quality=self.jpeg_quality)
                    else:
                        raise ValueError(f"Unknown image format {self.image_format}")

        table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))
        table.add_row(f"Outputs {self.scene_id}", str(self.output_path / self.scene_id))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))


@dataclass
class ExportPointCloud(_ExportPointCloud):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normal"
    """Name of the normal output."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = True
    """If true, saves in the frame of the transform.json file, if false saves in the frame of the scaled 
        dataparser transform"""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        def update_config(config: SceneCraftTrainerConfig) -> SceneCraftTrainerConfig:
            data_manager_config: SceneCraftDataManagerConfig = config.pipeline.datamanager
            setattr(data_manager_config, "camera_res_scale_factor", 1.)
            setattr(data_manager_config, "guide_camera_res_scale_factor", 1.)
            return config

        _, pipeline, _, _ = eval_setup(self.load_config,
                                            test_mode="inference",
                                            update_config_callback=update_config)

        if not pipeline.model.config.predict_normals and self.normal_method == "model_output":
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{self.normal_output_name}' not supported in current model.")
            self.normal_method = "open3d"

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=(-1, -1, -1),
            bounding_box_max=(1, 1, 1),
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(_ExportTSDFMesh):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        def update_config(config: SceneCraftTrainerConfig) -> SceneCraftTrainerConfig:
            data_manager_config: SceneCraftDataManagerConfig = config.pipeline.datamanager
            setattr(data_manager_config, "camera_res_scale_factor", 1.)
            setattr(data_manager_config, "guide_camera_res_scale_factor", 1.)
            return config

        _, pipeline, _, _ = eval_setup(self.load_config,
                                            test_mode="inference",
                                            update_config_callback=update_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            resolution=self.resolution,
            batch_size=self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=(-1, -1, -1),
            bounding_box_max=(1, 1, 1),
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


class LayoutRenderModel(nn.Module):

    dataset: str = ""

    def __init__(self) -> None:
        super().__init__()
        self.dummy_param = nn.Parameter(torch.Tensor(0))

    @staticmethod
    def _get_rays(H, W, K, c2w, normalize_dir: bool=False, format: str = "OpenCV") -> Sequence[torch.Tensor]:
        H, W = int(H), int(W)
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy') # pytorch's meshgrid has indexing='ij'
        i = i.t().to(c2w.device)
        j = j.t().to(c2w.device)
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1).to(c2w.device)
        if format == "OpenGL" or format == "Blender":
            dirs[..., 1:] = -dirs[..., 1:]
        # Rotate ray directions from camera frame to the world frame
        rays_d = c2w[:3, :3].matmul(dirs[..., None]).squeeze(-1)
        if normalize_dir:
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    @staticmethod
    def _filter_voxels_by_pixels(metas, **kwargs):
        voxels, voxel_size = metas['voxels'], metas['voxel_size']
        point_cloud_range = metas['point_cloud_range']
        # voxels: shape [M, 4], x y z label.
        voxels, labels = voxels[:, :3], voxels[:, -1]
        centroids_3d = voxels * voxel_size + voxel_size / 2 + point_cloud_range[:3]
        centroids_2d = project_3d_to_2d(centroids_3d, metas['extrin'], metas['intrin']) # [M, 3]
        valid = (centroids_2d[:, 0] > -(kwargs['width'] / 2)) & (centroids_2d[:, 0] < kwargs['width'] * 3 / 2 ) & \
                (centroids_2d[:, 1] > -(kwargs['height'] / 2)) & (centroids_2d[:, 1] < kwargs['height'] * 3 / 2) & \
                (centroids_2d[:, 2] > 0)
        return centroids_3d[valid], labels[valid]

    def _compute_intersections_chunk(self, rays_o, rays_d, metas, return_depth: bool=False, voxelized: bool=False, **kwargs):
        oriented = metas.get('boxes_oriented', False)
        bbox, cls = metas['obj_bbox'], metas['obj_class']
        rays_o = rays_o[:, None]
        rays_d = rays_d[:, None]

        if oriented:
            bbox_orientations = metas["bbox_orientations"]   # bbox_to_world [N, 3, 3]
            bbox_positions = metas["bbox_positions"]         # in world [N, 3]
            R_world_to_bbox = torch.linalg.inv(bbox_orientations)
            # Rotate ray directions from world frame to the bbox frame
            rays_d = (R_world_to_bbox[None, ...] @ rays_d[..., None])[..., 0]
            rays_o = (R_world_to_bbox[None, ...] @ (rays_o - bbox_positions[None, ...])[..., None])[..., 0]

        # compute ts
        bbox_min = bbox[:, :3] - bbox[:, 3:] / 2 # [N, 3]
        bbox_max = bbox[:, :3] + bbox[:, 3:] / 2
        t_mins = (bbox_min - rays_o) / rays_d # [W*H, N, 3]
        t_maxs = (bbox_max - rays_o) / rays_d
        ts = torch.stack([t_mins, t_maxs], dim=-1) # [M, N, 3, 2]
        t_mins_max = ts.min(-1)[0].max(-1)[0] # [M, N] last of entrance
        t_maxs_min = ts.max(-1)[0].min(-1)[0] # [M, N] first of exit
        # The first of exit of intersected boxes should be in front of the image.
        is_intersects = (t_mins_max < t_maxs_min) & (t_maxs_min > 0) # [M, N]
        if not oriented and not metas.get("customed", False):
            corner_3d = get_corner_bbox(bbox) # [N, 8, 3]
            if oriented:
                # Transform points from bbox frame to world frame
                corner_3d = (bbox_orientations[:, None] @ corner_3d[..., None])[..., 0] + bbox_positions[:, None]
            corner_2d = project_3d_to_2d(corner_3d, metas['extrin'], metas['intrin'])
            # Filter the case the camera inside the box and the box behined the camera
            unseen_bbox = torch.all(corner_2d[..., 2] <= 0, dim=1) # [N]
            origin_inside_bbox = (rays_o[0] >= bbox_min) & (rays_o[0] <= bbox_max) # [N, 3]
            origin_inside_bbox = origin_inside_bbox.all(-1) # [N]
            is_intersects[:, unseen_bbox | origin_inside_bbox] = False
            del corner_2d, corner_3d, unseen_bbox, origin_inside_bbox
        del t_mins, t_maxs, t_mins_max, t_maxs_min

        # Only care about the rays that intersects boxes > 0
        keep = is_intersects.sum(-1) > 0 # [M]
        t_nears = ts.min(-1)[0][keep] # [L, N, 3]
        rays_o = rays_o[keep] # [L, 1, 3]
        rays_d = rays_d[keep] # [L, 1, 3]
        intersects = rays_o[..., None, :] + t_nears[..., None] * rays_d[..., None, :] # [L, N, 3, 3]
        eps = torch.tensor([1e-4, 1e-4, 1e-4], dtype=torch.float32, device=bbox_min.device)
        bbox_min_expanded = (bbox_min[..., None, :] - eps).repeat(1, 3, 1)
        bbox_max_expanded = (bbox_max[..., None, :] + eps).repeat(1, 3, 1)
        valid_intersects = (intersects >= bbox_min_expanded) & (intersects <= bbox_max_expanded)
        valid_intersects = valid_intersects.all(-1) # [L, N, 3]
        is_positive = t_nears >= 0 # [L, N, 3]
        valid_intersects &= is_positive
        del intersects, is_positive, bbox_min_expanded, bbox_max_expanded

        # Find the nearest valid intersected plane and nearest intersected bbox.
        t_nears[~valid_intersects] = 1e10
        del valid_intersects
        t_nears = t_nears.min(-1)[0] # [L, N]
        sorted_min = torch.argsort(t_nears, dim=-1) # [L, N]
        first_min, second_min = sorted_min[:, 0], sorted_min[:, 1] if sorted_min.shape[1] > 1 else None # [L]

        return_dict = dict()
        # assign class index to pixel
        nearest_bbox_idx = torch.zeros_like(is_intersects[:, 0]).long() - 1 # [M]
        nearest_bbox_idx[keep] = first_min.long()
        return_dict.update(nearest_bbox_idx=nearest_bbox_idx)

        # assign depth value to pixel
        if return_depth:
            nearest_distance_2d = torch.zeros_like(is_intersects[:, 0]).to(bbox_min) - 1 # [M]
            nearest_distance_2d[keep] = torch.gather(t_nears, dim=1, index=first_min.unsqueeze(-1)).squeeze(-1)
            return_dict.update(nearest_distance_2d=nearest_distance_2d)

        return return_dict

    def _compute_intersections(self, rays_o, rays_d, metas, return_depth: bool=True, wh: bool=False, **kwargs
                              ) -> Union[npt.NDArray, Sequence[npt.NDArray]]:
        voxelized = False
        wh_shape = rays_o.shape
        if 'voxels' in metas:
            voxelized = True
            filtered_voxels, filtered_labels = self._filter_voxels_by_pixels(metas, **kwargs)
            filtered_voxels = torch.concat([filtered_voxels, torch.zeros_like(filtered_voxels) + metas['voxel_size']], dim=1)
            metas.update(obj_bbox=filtered_voxels, obj_class=filtered_labels)
            del filtered_voxels, filtered_labels

        remove_labels = kwargs.pop('remove_labels', [])
        if remove_labels:
            remove_label_ids = [RAW_DATASETS[self.dataset].CLASSES.index(label) for label in remove_labels if label in RAW_DATASETS[self.dataset].CLASSES]
            filtered_bboxes, filtered_labels = zip(*filter(lambda x: x[1] not in remove_label_ids, zip(metas['obj_bbox'], metas['obj_class'])))
            metas.update(obj_bbox=list(filtered_bboxes), obj_class=list(filtered_labels))

        rays_d[rays_d == 0] = 1e-8
        rays_o_flattened = rays_o.reshape(-1, 3) # [M, 3]
        rays_d_flattened = rays_d.reshape(-1, 3)
        del rays_o, rays_d

        rays_chunk = int(kwargs.get('rays_chunk', rays_o_flattened.shape[0]))
        all_outputs = defaultdict(list)
        for i in range(0, rays_o_flattened.shape[0], rays_chunk):
            chunk_outputs = self._compute_intersections_chunk(rays_o_flattened[i : i + rays_chunk],
                                                              rays_d_flattened[i : i + rays_chunk],
                                                              metas, return_depth, voxelized, **kwargs)
            for output_name, output in chunk_outputs.items():
                all_outputs[output_name].append(output)
        all_outputs = {k: torch.cat(v) for k, v in all_outputs.items()}

        nearest_bbox_idx_2d = all_outputs['nearest_bbox_idx'].reshape(wh_shape[:2]) # [W, H]
        depth_image = all_outputs['nearest_distance_2d'].reshape(wh_shape[:2]).cpu().numpy() if return_depth else None
        index_image = nearest_bbox_idx_2d.cpu().numpy()
        nearest_bbox_idx_2d[nearest_bbox_idx_2d != -1] = metas['obj_class'][nearest_bbox_idx_2d[nearest_bbox_idx_2d != -1]].long()
        nearest_bbox_idx_2d[nearest_bbox_idx_2d == -1] = 0
        label_image = nearest_bbox_idx_2d.cpu().numpy()
        del all_outputs, nearest_bbox_idx_2d

        if not wh:
            label_image = label_image.transpose(-1, -2)
            index_image = index_image.transpose(-1, -2)
            depth_image = depth_image.transpose(-1, -2) if depth_image is not None else None

        return label_image, index_image, depth_image

    @staticmethod
    def get_unique_labels_of_interest(label_image: npt.NDArray, roi: float = 1.) -> npt.NDArray:
        label_image_main_area = label_image
        if roi != 1.:
            height, width = label_image.shape
            vertical_margin = int(height * (1 - roi) / 2)
            horizontal_margin = int(width * (1 - roi) / 2)
            label_image_main_area = label_image[vertical_margin : height - vertical_margin, horizontal_margin : width - horizontal_margin]
        unique_labels = np.unique(label_image_main_area)
        return unique_labels

    def draw_outline(self, semantic_image, label_image, index_image, depth_image, labels, colors, cls: torch.Tensor, voxelized: bool=False) -> Sequence[npt.NDArray]:
        label_image, index_image, depth_image, labels = \
            map(lambda x: torch.tensor(x, device=cls.device) if x is not None else x, [label_image, index_image, depth_image, labels])

        # TODO: support draw outline on voxels and remove this
        if voxelized:
            return semantic_image

        for label in labels:
            label_indice = (label_image == label).nonzero()
            label_indice = label_indice[(label_indice[:, 0] < label_image.shape[0] - 1) &
                                        (label_indice[:, 1] < label_image.shape[1] - 1)]
            i, j = label_indice[:, 0], label_indice[:, 1]
            outline_index = index_image[i, j] != index_image[i + 1, j + 1]
            outline_indice = label_indice[outline_index]
            outline_indice = outline_indice[(label_image[outline_indice[:, 0], outline_indice[:, 1]] == label) &
                                            (label_image[outline_indice[:, 0] + 1, outline_indice[:, 1] + 1] == label) &
                                            (index_image[outline_indice[:, 0], outline_indice[:,  1]] != -1) &
                                            (index_image[outline_indice[:, 0] + 1, outline_indice[:, 1] + 1] != -1)]
            outline_indice = outline_indice.cpu().numpy()
            color = colors[label]
            outline_color = np.clip(color - 30, a_min=0, a_max=255)
            i, j = outline_indice[:, 0], outline_indice[:, 1]
            semantic_image[i, j] = outline_color

        return semantic_image

    def _forward(self, height, width, images, extrins, intrins, metas, rays_chunk,
                 scene_path, save_image, i, image_id, **kwargs):
        if kwargs.get("type") == "sample":
            scene_path = os.path.join(scene_path, "samples")
            if not os.path.exists(scene_path):
                os.makedirs(scene_path, exist_ok=True)
        # NOTE: Generally, it should normalize the ray direction here, when rendering
        # the depth map, it must. But for rendering semantic map, it could do or not.
        if hasattr(RAW_DATASETS[self.dataset], "get_rays"):
            rays_o, rays_d = RAW_DATASETS[self.dataset].get_rays(i=i, **metas)
        else:
            rays_o, rays_d = self._get_rays(height, width, intrins[i], extrins[i],
                                            format=RAW_DATASETS[self.dataset].coord_convention) # [w, h, 3]
        metas.update(extrin=copy.deepcopy(extrins[i]), intrin=intrins[i])

        if RAW_DATASETS[self.dataset].coord_convention in ("OpenGL", "Blender"):
            metas["extrin"] = opengl_to_opencv(metas["extrin"])
        label_image, index_image, depth_image = self._compute_intersections(rays_o, rays_d, metas, height=height,
                                                                            width=width, rays_chunk=rays_chunk, **kwargs)
        # hypersim dataset
        if "semantic_images" in metas and "frame_depths" in metas:
            semantic_image = metas["semantic_images"][i].cpu().numpy()
            mask = (label_image == 0) & ((semantic_image == 1) | (semantic_image == 2) | (semantic_image == 22))
            label_image[mask] = semantic_image[mask]
            frame_depths = metas["frame_depths"][i].cpu().numpy()
            mask = (depth_image == -1.) & ~np.isnan(frame_depths)
            depth_image[mask] = frame_depths[mask]
        colors = np.array([RAW_DATASETS[self.dataset].COLORS[j] for j in range(label_image.max() + 1)])
        semantic_image = colors[label_image.astype(np.int32)]
        # add outlines of objects
        # TODO: support outlines in voxels
        unique_labels = self.get_unique_labels_of_interest(label_image, roi=1.)
        semantic_image = self.draw_outline(semantic_image, label_image, index_image, depth_image, unique_labels, colors,
                                                                        metas['obj_class'], voxelized=('voxels' in metas))
        np.savez_compressed(f'{scene_path}/{image_id}.npz', unique_labels=unique_labels.astype(np.uint8),
                                                            labels=label_image.astype(np.uint8),
                                                            depths=depth_image.astype(np.float32),
                                                            extrin=extrins[i].cpu().numpy().astype(np.float32),
                                                            intrin=intrins[i].cpu().numpy().astype(np.float32))
        # save semantic images
        if save_image:
            image_count = 0
            merged_image = Image.new('RGB', (width, height * (int(kwargs.get("type") == "dataset")
                                                              + 1 + int(depth_image is not None))))
            if not metas.get("customed", False):
                if kwargs.get("type") == "dataset":
                    rgb_image = Image.fromarray(images[i].cpu().numpy().astype(np.uint8))
                    rgb_image = draw_bbox_on_image(rgb_image, metas['obj_bbox'].cpu().numpy(),
                                                    extrins[i].cpu().numpy(), intrins[i].cpu().numpy(),
                                                    labels=metas['obj_class'].cpu().numpy().astype(np.uint8),
                                                    label2cat=RAW_DATASETS[self.dataset].label2cat,
                                                    bbox_orientations=metas.get("bbox_orientations", None),
                                                    bbox_positions=metas.get("bbox_positions", None),
                                                    voxelized=('voxels' in metas))
                    merged_image.paste(rgb_image, (0, image_count * height))
                    image_count += 1
            semantic_image = Image.fromarray(semantic_image.astype(np.uint8))
            merged_image.paste(semantic_image, (0, image_count * height))
            image_count += 1
            if depth_image is not None:
                merged_image.paste(Image.fromarray(colorize_depth(depth_image).astype(np.uint8)).convert('RGB'), (0, image_count * height))
            merged_image.save(f'{scene_path}/{image_id}.png')

    def forward(self, inputs: Sequence[Union[torch.Tensor, dict]], scene_ids: list, image_ids: List[list],
                rays_chunk: int, output_path: str, save_image: bool=False, **kwargs):
        # images: [B, N, H, W, 3], extrins: [B, N, 4, 4], scene_ids: [B]
        # NOTE: only support B=1 due to different number of bbox/voxels.
        scene_name = scene_ids[0]
        image_ids = image_ids[0]
        images, extrins, intrins, metas = inputs
        height, width = RAW_DATASETS[self.dataset].ori_h, RAW_DATASETS[self.dataset].ori_w
        if 'obj_bbox' not in metas:
            print(f"Skipped {scene_name} since there is no bbox in it !")
            return
        N = images.shape[0]
        save_scene_names = kwargs.get("save_scene_names", True)
        scene_path = os.path.join(output_path, scene_name) if save_scene_names else output_path
        # check if skip
        if os.path.exists(scene_path) and not metas.get("customed", False):
            files = os.listdir(scene_path)
            if len(files) == N * (2 if save_image else 1):
                print(f"Skipped {scene_name} with {N} frames since it is ready.")
                return
        os.makedirs(scene_path, exist_ok=True)
        for i, image_id in tqdm(enumerate(image_ids), leave=False):
            try:
                self._forward(height, width, images, extrins, intrins, metas, rays_chunk,
                              scene_path, save_image, i, image_id, **kwargs)
                print(f"Rendered sample {image_id} of scene {scene_name}")
            except Exception as e:
                print(f"{e}! Skippedd sample {image_id}.")


@dataclass
class LayoutRender:
    """
    Render layout.
    """

    dataset_type: str = ""
    """Type of dataset."""
    model: LayoutRenderModel = LayoutRenderModel()
    """Model to do the rendering."""

    # Must be a diffusion process
    @torch.no_grad()
    @check_main_thread
    def render_scene(self, output_path: str, scene_id: str, **kwargs) -> None:
        self.model.dataset = self.dataset_type
        # output_path: outputs/${dataset}/${scene_id}
        self.model.eval()
        output_semantic_path = os.path.join(output_path, "semantic_images")
        dataset = RAW_DATASETS[self.dataset_type](scene_ids=scene_id,
                                            custom_scene_id=scene_id,
                                            split="all",
                                            **kwargs)
        if hasattr(dataset, "_get_sample_data"):
            sample_dataset = RAW_DATASETS[self.dataset_type](scene_ids=scene_id,
                                                        custom_scene_id=scene_id,
                                                        split="all",
                                                        type="sample",
                                                        **kwargs)
            dataset = ConcatDataset([sample_dataset, dataset])
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=16,
                                collate_fn=partial(collate_fn, output_semantic_path, False),
                                pin_memory=True,
                                shuffle=False)

        print(f"Render layout begins.")
        for inputs in tqdm(dataloader):
            inputs, scene_ids, image_ids = inputs
            inputs = tuple(map(lambda x: x.float().squeeze(0) if isinstance(x, torch.Tensor)
                                else {k: v.float().squeeze(0) for k, v in x.items()} if isinstance(x, dict) else x, inputs))
            try:
                self.model(inputs, scene_ids, image_ids, rays_chunk=8192, output_path=output_semantic_path,
                           save_image=True, return_depth=True, save_scene_names=False)
            except Exception as e:
                print(f"Skipped {scene_ids} due to {e}.")

        print("Generate JSONL files.")
        jsonl_path = os.path.join(output_path, "all.jsonl")
        ids = list(map(
                    lambda fname: fname.removesuffix(".npz"), filter(lambda fname: fname.endswith(".npz"), os.listdir(output_semantic_path))))
        info = []
        for i, id in enumerate(ids):
            item_id = np.load(os.path.join(output_semantic_path, f"{id}.npz"))['unique_labels']
            info_dict = dict(scene_id=scene_id, id=id,
                            target=dataset.get_rgb_image_path(scene_id, id) if self.dataset_type != "custom" and not id.startswith("center") else None,
                            source=os.path.join(output_semantic_path, f"{id}.npz"),
                            item_id=item_id.reshape(-1).tolist())
            if hasattr(dataset, "prompts") and dataset.prompts is not None:
                info_dict.update(prompt=dataset.prompts[i])
            info.append(info_dict)
        info = list(sorted(info, key=lambda x: x["id"]))
        with open(jsonl_path, "w") as f:
            f.write('\n'.join((map(json.dumps, info))))

    @classmethod
    @torch.no_grad()
    def render_dataset(cls):
        assert args.batch_size == 1
        custom_scene_id = None
        if args.dataset == "custom":
            assert args.scene_id is not None, f"scene_id required for custom dataset"
            custom_scene_id = args.scene_id
        global logger

        if args.generate_caption:
            logger = setup_logger(os.path.basename(__file__))
            generate_captions(args.dataset, args.data_root, args.output_path)
            return

        if args.generate_json:
            logger = setup_logger(os.path.basename(__file__))
            generate_json(args.dataset, args.data_root, args.output_path, args.jsonl_path, args.limit, args.roi)
            return

        # Run!
        from torch.utils.data.distributed import DistributedSampler
        from torch.nn.parallel import DistributedDataParallel

        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=local_rank)
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

        logger = setup_logger(os.path.basename(__file__))

        if args.dataset == "custom":
            logger.info(f"Dataset convention was set to {args.dataset_convention}")
            if args.dataset_convention == "scannetpp":
                RAW_DATASETS["custom"].switch_to_scannetpp_format()
            elif args.dataset_convention == "hypersim":
                pass
            else:
                raise ValueError(f"Unknown `dataset_convention`== {args.dataset_convention}")

        os.makedirs(args.output_path, exist_ok=True)
        dataset = RAW_DATASETS[args.dataset](raw_root_dir=args.raw_data_root,
                                                root_dir=args.data_root,
                                                type=args.camera_type,
                                                align=True,
                                                limit=args.limit,
                                                split=args.split,
                                                voxel_size=args.voxel_size,
                                                scene_ids=args.scene_id,
                                                custom_scene_id=custom_scene_id)

        cls.model.dataset = args.dataset
        model = DistributedDataParallel(cls.model.cuda(), device_ids=[local_rank], output_device=local_rank)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sampler=sampler,
                                collate_fn=partial(collate_fn, args.output_path, True),
                                shuffle=False,
                                pin_memory=True)

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        if not isinstance(args.remove_labels, list):
            args.remove_labels = [args.remove_labels]

        for inputs in tqdm(dataloader):
            inputs, scene_ids, image_ids = inputs
            inputs = tuple(map(lambda x: x.float().squeeze(0) if isinstance(x, torch.Tensor)
                                else {k: v.float().squeeze(0) for k, v in x.items()} if isinstance(x, dict) else x, inputs))
            try:
                model(inputs, scene_ids, image_ids, args.rays_chunk, dataset=args.dataset,
                        output_path=args.output_path, save_image=args.save_image, save_depth=args.save_depth,
                        remove_labels=args.remove_labels, type=args.camera_type)
            except Exception as e:
                print(f"Skipped {scene_ids} due to {e}.")

        torch.distributed.destroy_process_group()


def generate_captions(dataset: str, data_root, output_path, type="lavis"):
    if type == "lavis":
        from lavis.models import load_model_and_preprocess
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    elif type == "diffusers":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="auto")
        if not model.device == torch.device('cpu'):
            model.to(torch.float16)
    dataset = RAW_DATASETS[dataset](root_dir=data_root, split="all")

    def caption(images: "Image") -> List[str]:
        if type == "lavis":
            images = torch.stack([vis_processors["eval"](image).to(device) for image in images])
            texts = model.generate({"image": images})
        elif type == "diffusers":
            inputs = processor(images=images, text=texts, return_tensors="pt").to(model.device, model.dtype)
            ids = model.generate(**inputs)
            texts = list(map(lambda text: text.strip(), processor.batch_decode(ids, skip_special_tokens=True)))
        return texts

    # scannetpp
    os.makedirs(output_path, exist_ok=True)
    for scene in tqdm(dataset.split_available_scenes):
        if os.path.isfile(os.path.join(output_path, f"{scene}.json")):
            continue
        try:
            image_ids = dataset.get_scene_image_ids(scene)
            image_paths = list(map(lambda id: dataset.get_rgb_image_path(scene, id), image_ids))
            images = list(map(lambda path: Image.open(path), image_paths))
            # captioning
            # [Errno 24] Too many open files ... -> ulimit -n 10240
            bs = 500 if len(images) < 1500 else 200
            texts = list(chain(*[caption(images[i : i+bs]) for i in range(0, len(images), bs)]))
            captions = dict(zip(image_ids, texts))
            tqdm.write(f"Processed scene {scene} with {len(texts)} captions.")
            with open(f"{output_path}/{scene}.json", "w") as f:
                f.write(json.dumps(captions))
        except Exception as e:
            tqdm.write(f"{e}! Scene {scene} skipped.")
            if "Too many open files" in str(e):
                tqdm.write("Run `ulimit -n 10240` to fix it!")


def process_scene(dataset, source_path, roi, scene, label, caption):

    id = label.split('/')[-1][:-4]
    print(f"Processing {scene}, id {id}.")
    image_path = os.path.join(source_path, RAW_DATASETS[dataset].get_relative_rgb_image_path(scene, id))
    source_image_path = label[:-3] + 'png'
    assert os.path.exists(image_path), f"{image_path} doesn't exist."
    assert os.path.exists(source_image_path), f"{source_image_path} doesn't exist."
    if roi == 1:
        item_id = np.load(label)['unique_labels']
    else:
        # Need more time
        label_image = np.load(label)['labels']
        item_id = LayoutRender.get_unique_labels_of_interest(label_image, roi).astype(np.uint8)
    depth = np.load(label)["depths"]
    max_depth = float(depth.max())
    return {
        'scene_id': scene,
        'id': id,
        'target': image_path,
        'source': label,
        'caption': caption,
        'item_id': item_id.reshape(-1).tolist(),
        'max_depth': max_depth,
    }


def generate_json(dataset, source_path, target_path, jsonl_path, limit, roi):
    process_scene_partial = functools.partial(process_scene, dataset, source_path, roi)

    def _process_split(scenes):
        fetch_labels = lambda scene: list(sorted(glob.glob(os.path.join(target_path, scene, '*.npz'))))
        fetch_scenes = lambda scene, all_labels: [scene] * len(all_labels)
        all_labels = list(map(fetch_labels, scenes))
        num_labels = list(map(lambda scene: len(scene), all_labels))
        if limit != -1:
            invalid_scene_ids = list(filter(lambda i: num_labels[i] != limit, range(len(num_labels))))
            if len(invalid_scene_ids) != 0:
                invalid_scenes = [scenes[i] for i in invalid_scene_ids]
                logger.warning(f"Removed scenes whose labels are accidentally abnormal: {', '.join(invalid_scenes)}")
                scenes = list(filter(lambda x: x not in invalid_scenes, scenes))
                all_labels = [all_labels[i] for i in range(len(all_labels)) if i not in invalid_scene_ids]
        all_scenes = list(chain(*map(fetch_scenes, scenes, all_labels)))
        try:
            caption_path = os.path.join(source_path, "captions")
            scene_captions = list(map(lambda scene: json.load(open(os.path.join(caption_path, f"{scene}.json"), "r", encoding="UTF-8")), scenes))
            all_captions_dict = dict(zip(scenes, scene_captions))
            all_image_ids = list(map(lambda label: label.split('/')[-1].removesuffix('.npz'), chain(*all_labels)))
            all_captions = list(map(lambda scene, id: all_captions_dict[scene][id], all_scenes, all_image_ids))
        except Exception as e:
            all_captions = [""] * len(list(chain(*all_labels)))
            print(f"{e}! Load captions failed.")
        with futures.ProcessPoolExecutor(max_workers=16) as executor:
            split_info = executor.map(process_scene_partial, all_scenes, chain(*all_labels), all_captions)
        return list(split_info), scenes

    split_scenes = []
    all_scenes = list(os.listdir(target_path))
    for split_path in SPLIT_PATHS[dataset].values():
        with open(split_path, 'r') as file:
            split_scene = [line.strip() for line in file.readlines()]
            # Filter scenes that not be converted due to e.g. cuda out of memory.
            split_scene = list(filter(lambda x: x in all_scenes, split_scene))
            split_scenes.append(split_scene)

    info_list, updated_split_scenes = zip(*map(_process_split, split_scenes))
    splits = list(SPLIT_PATHS[dataset].keys())
    if jsonl_path is None:
        jsonl_path = source_path
    all_info = []
    for info, split in zip(info_list, splits):
        info = list(sorted(info, key=lambda x: (x["scene_id"], x["id"])))
        info = list(filter(lambda x: x["max_depth"] <= 20., info))
        all_info.extend(info)
        with open(f"{jsonl_path}/{split}.jsonl", "w") as f:
            f.write('\n'.join(map(json.dumps, info)))
    with open(f"{jsonl_path}/all.jsonl", "w") as f:
        f.write('\n'.join(map(json.dumps, all_info)))
    logger.info(f"Processed {splits} splits that contains "
                f"{', '.join([str(len(split_scene)) for split_scene in updated_split_scenes])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # runtime
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--launcher', default='pytorch', type=str, required=False)
    # render
    parser.add_argument(
        "--dataset", type=str, default="scannet", required=False, help="Dataset type, scannet or scannetpp.")
    parser.add_argument(
        "--raw-data-root", type=str, default="", required=False, help="Path to raw dataset without preprocess.")
    parser.add_argument(
        "--data-root", type=str, default="./data/scannet", required=False, help="Path to local dataset.")
    parser.add_argument(
        "--output-path", type=str, default="./data/scannet/semantic_images", required=False)
    parser.add_argument(
        "--dataset-convention", type=str, default="hypersim", required=False)
    parser.add_argument(
        "--camera-type", type=str, default="dataset", required=False)
    parser.add_argument("--split", type=str, default="train", required=False)
    parser.add_argument("--voxel-size", type=float, default=-1., required=False)
    parser.add_argument("--scene-id", metavar='N', type=str, nargs='+', default=[], required=False)
    parser.add_argument("--remove-labels", type=list, default=[], required=False)
    parser.add_argument("--rays-chunk", type=int, default=8192, required=False)
    parser.add_argument("--save-image", action="store_true", required=False)
    parser.add_argument("--save-depth", action="store_true", required=False)
    parser.add_argument("--limit", type=int, default=None, required=False)
    parser.add_argument("--generate-json", action="store_true", required=False)
    parser.add_argument("--generate-caption", action="store_true", required=False)
    parser.add_argument("--jsonl-path", type=str, default=None, required=False)
    parser.add_argument("--roi", type=float, default=1., required=False)
    args = parser.parse_args()
    seed_everything(0)
    LayoutRender.render_dataset()
