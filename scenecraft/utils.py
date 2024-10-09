import functools
import gc
import os
import logging
import random
import json
import cv2
import numpy as np
import torch
import PIL
import sys
import open3d as o3d
import torch.nn.functional as F
from numpy import typing as npt
from einops import rearrange
from dataclasses import dataclass, fields, field
from logging import Logger
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from pathlib import Path
from typing import List, Union, Optional, Callable, cast, Any, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from nerfstudio.utils import profiler
from nerfstudio.utils import comms
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.utils.rich_utils import CONSOLE

from accelerate.state import PartialState
from accelerate.logging import MultiProcessAdapter


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier(groups = None):
    if not _distributed_available():
        return
    elif groups is not None:
        torch.distributed.barrier(groups)
    else:
        torch.distributed.barrier()


def rank_zero_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.distributed.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper


@rank_zero_only
def rank_zero_info(*args, fn: Callable=print, **kwargs):
    fn(*args, **kwargs)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def markvar(*args, **kwargs):
    """ Like `dataclasses.field` """
    if len(kwargs) == 0:
        assert len(args) != 0, f"No arguments!"
        kwargs = dict(default=args[0])
    return field(**kwargs, metadata={'log': True})


def is_local_main_process() -> bool:
    return comms.get_local_rank() == 0


def check_local_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(*args, **kwargs):
        ret = None
        if is_local_main_process():
            ret = func(*args, **kwargs)
        return ret

    return wrapper


def check_nerf_process(func: Callable, nerf_rank: int = 1) -> bool:

    def wrapper(*args, **kwargs):
        ret = None
        if comms.get_rank() == nerf_rank:
            ret = func(*args, **kwargs)
        return ret

    return wrapper


@dataclass
class LoggableConfig:
    """ Like `PrintableConfig` """
    def log(self, logger: Logger):
        log_vars = [field_info.name for field_info in fields(self) if field_info.metadata.get("log")]
        for var in log_vars:
            logger.info(f"  {' '.join([e.capitalize() if i == 0 else e for i, e in enumerate(var.split('_'))])} = {getattr(self, var)}")


def shifted_expotional_decay(a, b, c, r):
    return a * torch.exp(-b * r) + c


def shifted_cosine_decay(a, b, c, r):
    return a * torch.cos(b * r + c) + a


def load_from_json(filename: Path):
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def load_from_jsonl(filename: Path):
    assert filename.suffix == ".jsonl"
    if not filename.exists():
        return None

    data = []
    with open(filename, encoding="utf-8") as f:
        for row in f:
            data.append(json.loads(row))
    return data


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min
def seed_everything(seed=None):
    seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        raise ValueError(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")

    print(f"seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def format_intrinsic(fx, fy, cx, cy, length: int = 1):
    K = np.zeros((length, 4, 4), dtype=np.float32)
    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = cx
    K[..., 1, 2] = cy
    K[..., 2, 2] = 1.0
    K[..., 3, 3] = 1.0
    return K


def colmap_to_nerfstudio(c2w):
    # Convert from COLMAP's camera coordinate system to nerfstudio/instant-ngp
    c2w[..., 0:3, 1:3] *= -1
    c2w = c2w[..., np.array([1, 0, 2, 3]), :]
    c2w[..., 2, :] *= -1
    return c2w


def opengl_to_opencv(c2w):
    transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if isinstance(c2w, torch.Tensor):
        transform = torch.Tensor(transform).to(c2w)
    c2w[..., :3, :3] @= transform
    return c2w


def project_3d_to_2d_np(points_3d, extrin, intrin):
    points_3d_homogeneous = np.concatenate((points_3d, np.ones([*points_3d.shape[:-1], 1])), -1)
    projection = intrin @ np.linalg.inv(extrin)
    points_2d_homogeneous = points_3d_homogeneous @ projection.T # (projection @ points_3d_homogeneous.T).T
    points_2d = np.concatenate([points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3],
                                points_2d_homogeneous[..., 2:3]], axis=-1)
    return points_2d # [N, 8, 3]


def project_3d_to_2d(points_3d, extrin, intrin):
    points_3d_homogeneous = torch.concat((points_3d, torch.ones([*points_3d.shape[:-1], 1]).to(points_3d)), -1)
    projection = intrin @ torch.linalg.inv(extrin)
    points_2d_homogeneous = points_3d_homogeneous @ projection.T
    points_2d = torch.concat([points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3],
                              points_2d_homogeneous[..., 2:3]], axis=-1)
    return points_2d # [N, 8, 3]


def project_target_to_source_np(extrin_target, extrin_source, intrin_target, intrin_source,
                                target, is_uv = True):
    if is_uv:
        target_camera = np.linalg.inv(intrin_target[:3, :3]) @ np.array(target + [1])
    else:
        target_camera = target
    target_world = extrin_target @ np.concatenate([target_camera, np.array([1])], axis=0)
    source_camera = np.linalg.inv(extrin_source) @ target_world
    source = intrin_source[:3, :3] @ source_camera[:3]
    source /= source[2]
    return source[:2]


def project_target_to_source(extrin_target, extrin_source, intrin_target, intrin_source,
                             target, is_uv = True):
    ori_dtype = extrin_target.dtype
    if is_uv:
        target_h = torch.concat([target, torch.ones_like(target[..., -1:])], dim=-1)
        target_camera = torch.linalg.inv(intrin_target[..., :3, :3].float()) @ target_h[..., None].float()
        target_camera = target_camera[..., 0].to(ori_dtype)
    else:
        target_camera = target
    target_camera_h = torch.concat([target_camera, torch.ones_like(target_camera[..., -1:])], dim=-1)
    target_world = extrin_target @ target_camera_h[..., None]
    source_camera = torch.linalg.inv(extrin_source.float()) @ target_world.float()
    source = intrin_source[..., :3, :3] @ source_camera[..., :3, :].to(ori_dtype)
    source = source[..., 0]
    source[..., :2] /= source[..., 2:3]
    return source[..., :2], source[..., 2:3]


def get_corner_bbox_np(bbox: npt.NDArray) -> npt.NDArray:
    min_corner = bbox[:, :3] - bbox[:, 3:] / 2
    max_corner = bbox[:, :3] + bbox[:, 3:] / 2
    dim_x, dim_y, dim_z = np.split(max_corner - min_corner, 3, -1)
    dim_zeros = np.zeros_like(dim_x)
    # get all 8 corners
    corners = np.stack([
        min_corner,
        min_corner + np.concatenate([dim_x, dim_zeros, dim_zeros], axis=-1),
        min_corner + np.concatenate([dim_zeros, dim_y, dim_zeros], axis=-1),
        min_corner + np.concatenate([dim_zeros, dim_zeros, dim_z], axis=-1),
        max_corner,
        max_corner - np.concatenate([dim_x, dim_zeros, dim_zeros], axis=-1),
        max_corner - np.concatenate([dim_zeros, dim_y, dim_zeros], axis=-1),
        max_corner - np.concatenate([dim_zeros, dim_zeros, dim_z], axis=-1),
    ], axis=1)
    return corners


def get_corner_bbox(bbox: torch.Tensor) -> torch.Tensor:
    min_corner = bbox[:, :3] - bbox[:, 3:] / 2
    max_corner = bbox[:, :3] + bbox[:, 3:] / 2
    dim_x, dim_y, dim_z = torch.split(max_corner - min_corner, 1, -1)
    dim_zeros = torch.zeros_like(dim_x)
    # get all 8 corners
    corners = torch.stack([
        min_corner,
        min_corner + torch.concat([dim_x, dim_zeros, dim_zeros], axis=-1),
        min_corner + torch.concat([dim_zeros, dim_y, dim_zeros], axis=-1),
        min_corner + torch.concat([dim_zeros, dim_zeros, dim_z], axis=-1),
        max_corner,
        max_corner - torch.concat([dim_x, dim_zeros, dim_zeros], axis=-1),
        max_corner - torch.concat([dim_zeros, dim_y, dim_zeros], axis=-1),
        max_corner - torch.concat([dim_zeros, dim_zeros, dim_z], axis=-1),
    ], axis=1)
    return corners


# From https://github.com/scannetpp/scannetpp/common/utils/colmap.py
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def to_matrix(x: torch.Tensor):
    h, w = x.shape[-2:]
    if h < w:
        x = torch.cat([x, torch.zeros_like(x[..., -1:, :])], dim=-2)
        x[..., -1, -1] = 1
    elif w < h:
        x = torch.cat([x, torch.zeros_like(x[..., -1:])], dim=-1)
        x[..., -1, -1] = 1
    return x


def colorize_depth(depth: Union[npt.NDArray, Image.Image], depth_min: int = -1, depth_max: int = -1,
                   output_type: str = "np") -> npt.NDArray:
    if isinstance(depth, Image.Image):
        depth = np.array(depth)

    depth = 1.0 / (depth + 1e-6)
    invalid_mask = (depth <= 0) | (depth >= 1e6) | np.isnan(depth) | np.isinf(depth)
    if depth_min < 0 or depth_max < 0:
        depth_min = np.percentile(depth[~invalid_mask], 5)
        depth_max = np.percentile(depth[~invalid_mask], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min + 1e-10)
    depth_scaled_uint8 = np.uint8(np.clip(depth_scaled * 255, 0, 255))
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0
    depth_color = depth_color[..., ::-1]

    if output_type == "pil":
        depth_color = Image.fromarray(depth_color.astype(np.uint8)).convert("RGB")
    return depth_color


def draw_bbox_on_image(image: Image.Image,
                       bboxes: npt.NDArray,
                       extrin: npt.NDArray,
                       intrin: npt.NDArray,
                       labels: npt.NDArray=None,
                       label2cat: dict=None,
                       bbox_orientations: Optional[Union[npt.NDArray, torch.Tensor]]=None,
                       bbox_positions: Optional[Union[npt.NDArray, torch.Tensor]]=None,
                       voxelized: bool=False) -> Image.Image:

    corners_3d = get_corner_bbox_np(bboxes)
    if bbox_orientations is not None and bbox_positions is not None:
        if isinstance(bbox_orientations, torch.Tensor):
            bbox_orientations = bbox_orientations.cpu().numpy()
        if isinstance(bbox_positions, torch.Tensor):
            bbox_positions = bbox_positions.cpu().numpy()

        corners_3d = (bbox_orientations[:, None] @ corners_3d[..., None])[..., 0] + bbox_positions[:, None]

    corners_2d = project_3d_to_2d_np(corners_3d, extrin, intrin)

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    font = ImageFont.load_default()

    # only draw centroids
    if voxelized:
        centroids_3d = bboxes[:, :3]
        centroids_2d = project_3d_to_2d_np(centroids_3d, extrin, intrin)[:, :2].astype(np.int32)
        for centroid in centroids_2d:
            draw.point(tuple(centroid), fill='red')
        return img_draw

    for i, box_corners in enumerate(corners_2d):
        if np.any(box_corners[..., 2] < 0):
            continue
        box_corners = box_corners[..., :2]
        edges = [(0, 1), (0, 2), (0, 3), (1, 6), (1, 7), (2, 5), (2, 7), (3, 5), (3, 6), (4, 5), (4, 6), (4, 7)]
        for edge in edges:
            draw.line([tuple(box_corners[edge[0]]), tuple(box_corners[edge[1]])], fill=(0, 255, 0), width=5)

        if labels is not None:
            label = labels[i]
            cat = label2cat[label]
            draw.text(box_corners[0], cat, font=font, fill=(255, 0, 0))

    return img_draw


def draw_text_on_image(image: Image.Image, texts: Union[str, List[str]], locs: Union[tuple, List[tuple]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if not isinstance(texts, list):
        texts = [texts]
    assert isinstance(texts[0], str), f"Got type of `texts` {type(texts[0])}."
    if not isinstance(locs, list):
        locs = [locs]
    for text, loc in zip(texts, locs):
        draw.text(loc, text, fill="black", font=font)
    return image


def transform_points_to_voxels(points: Union[npt.NDArray, torch.Tensor], voxel_size: list = [0.2, 0.2, 0.2],
                               labels: Union[npt.NDArray, torch.Tensor] = None, max_num_points: int = -1,
                               determinstic: bool = True):
    try:
        from scenecraft.ops.voxelize.voxelization import voxelization
    except:
        raise ImportError

    if isinstance(points, npt.NDArray):
        points = torch.tensor(points, device='cuda', dtype=torch.float32)

    if labels is not None:
        if isinstance(labels, npt.NDArray):
            labels = torch.tensor(labels.astype(np.int32), device='cuda', dtype=torch.float32)
        points = torch.cat([points[:, :3], labels[..., None].float()], dim=1)

    # only use x y z label
    points = points[:, :4]
    # will add zero points in hard voxelization
    points[:, -1] = points[:, -1] + 1
    max_voxels = 1e5
    max_num_points = 1e3 / voxel_size[0]
    # remove nan
    points = points[~(torch.isnan(points[:, 0]) | torch.isnan(points[:, 1]) | torch.isnan(points[:, 2]))]
    # NOTE: need to add min range to transform voxel to original position
    point_cloud_range = [points[:, 0].min(), points[:, 1].min(), points[:, 2].min(),
                         points[:, 0].max(), points[:, 1].max(), points[:, 2].max()]

    voxels = voxelization(points, voxel_size, point_cloud_range, int(max_num_points), int(max_voxels), determinstic)
    voxels = [e.cpu() for e in voxels] if not isinstance(voxels, torch.Tensor) else voxels.cpu()

    # hard voxelization
    if max_num_points != -1 and max_voxels != -1:
        voxels, coors, _ = voxels
        labels = voxels[..., -1] # [M, N]
        unique_labels, mapped_labels = torch.unique(labels, sorted=True, return_inverse=True)
        label_counts = torch.zeros((len(voxels), len(unique_labels))).to(labels.device).long()
        label_counts.scatter_add_(1, mapped_labels.long(), torch.ones_like(mapped_labels).long())

        indices = torch.argsort(label_counts, dim=-1, descending=True)
        top1_labels = unique_labels[indices[:, 0]]
        if indices.shape[-1] > 1:
            top2_labels = unique_labels[indices[:, 1]]
            top1_labels = torch.where(top1_labels == 0, top2_labels, top1_labels)
        top1_labels = top1_labels - 1

    # TODO: add dynamic voxelization
    else:
        pass

    # note the sequence of coors
    voxels = np.concatenate([coors.numpy()[:, [2, 1, 0]], top1_labels.numpy()[..., np.newaxis]], axis=-1) # [M, 4]

    return voxels


def visualize_voxels(colors: npt.NDArray, voxels: Union[npt.NDArray, List[npt.NDArray]]=None, voxel_paths: List[str]=None,
                        voxel_dir: str=None):

    def _visualize_voxels(colors: npt.NDArray, voxel: npt.NDArray):
        from mayavi import mlab

        voxel = voxel.reshape(-1, 4)
        figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
        plot = mlab.points3d(voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3],
                            colormap="viridis", scale_factor=1., mode="cube", opacity=1.0)

        plot.glyph.scale_mode = "scale_by_vector"
        plot.module_manager.scalar_lut_manager.lut.table = colors
        scene = figure.scene
        scene.camera.compute_view_plane_normal()
        scene.render()

        mlab.show()

    if voxel_dir is not None:
        voxel_paths = [os.path.join(voxel_dir, f) for f in os.listdir(voxel_dir)]

    if voxel_paths is not None:
        for path in voxel_paths:
            _visualize_voxels(colors, np.load(path))
        return

    assert voxels is not None, f"Should give one of voxels, voxel_paths, voxel_dir."
    if not isinstance(voxels, list):
        voxels = [voxels]
    for voxel in voxels:
        _visualize_voxels(colors, voxel)


# Adapted from nerfstudio/exporter/exporter_utils.py
def generate_point_cloud(
    pipeline, # SceneCraftPipeline
    num_points: int = 10000000,
    remove_outliers: bool = True,
    estimate_normals: bool = False,
    reorient_normals: bool = False,
    rgb_output_name: str = "rgb",
    depth_output_name: str = "depth",
    normal_output_name: Optional[str] = None,
    use_bounding_box: bool = True,
    bounding_box_min: Optional[Tuple[float, float, float]] = None,
    bounding_box_max: Optional[Tuple[float, float, float]] = None,
    crop_obb: Optional[OrientedBox] = None,
    std_ratio: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Generate a point cloud from a nerf.

    Args:
        pipeline: Pipeline to evaluate with.
        num_points: Number of points to generate. May result in less if outlier removal is used.
        remove_outliers: Whether to remove outliers.
        reorient_normals: Whether to re-orient the normals based on the view direction.
        estimate_normals: Whether to estimate normals.
        rgb_output_name: Name of the RGB output.
        depth_output_name: Name of the depth output.
        normal_output_name: Name of the normal output.
        use_bounding_box: Whether to use a bounding box to sample points.
        bounding_box_min: Minimum of the bounding box.
        bounding_box_max: Maximum of the bounding box.
        std_ratio: Threshold based on STD of the average distances across the point cloud to remove outliers.

    Returns:
        Point cloud.
    """

    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
    points = []
    rgbs = []
    normals = []
    view_directions = []
    if use_bounding_box and (crop_obb is not None and bounding_box_max is not None):
        CONSOLE.print("Provided aabb and crop_obb at the same time, using only the obb", style="bold yellow")
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=num_points)
        while not progress_bar.finished:
            normal = None

            # SceneCraftPipeline.get_inference()
            with torch.no_grad():
                ray_bundle, _ = pipeline.datamanager.next_eval()
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            if rgb_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rgb_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rgb_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            if depth_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {depth_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --depth_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            rgba = pipeline.model.get_rgba_image(outputs, rgb_output_name)
            depth = outputs[depth_output_name]
            if normal_output_name is not None:
                if normal_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {normal_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --normal_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                normal = outputs[normal_output_name]
                assert (
                    torch.min(normal) >= 0.0 and torch.max(normal) <= 1.0
                ), "Normal values from method output must be in [0, 1]"
                normal = (normal * 2.0) - 1.0
            point = ray_bundle.origins + ray_bundle.directions * depth
            view_direction = ray_bundle.directions

            # Filter points with opacity lower than 0.5
            mask = rgba[..., -1] > 0.5
            point = point[mask]
            view_direction = view_direction[mask]
            rgb = rgba[mask][..., :3]
            if normal is not None:
                normal = normal[mask]

            if use_bounding_box:
                if crop_obb is None:
                    comp_l = torch.tensor(bounding_box_min, device=point.device)
                    comp_m = torch.tensor(bounding_box_max, device=point.device)
                    assert torch.all(
                        comp_l < comp_m
                    ), f"Bounding box min {bounding_box_min} must be smaller than max {bounding_box_max}"
                    mask = torch.all(torch.concat([point > comp_l, point < comp_m], dim=-1), dim=-1)
                else:
                    mask = crop_obb.within(point)
                point = point[mask]
                rgb = rgb[mask]
                view_direction = view_direction[mask]
                if normal is not None:
                    normal = normal[mask]

            points.append(point)
            rgbs.append(rgb)
            view_directions.append(view_direction)
            if normal is not None:
                normals.append(normal)
            progress.advance(task, point.shape[0])
    points = torch.cat(points, dim=0)
    rgbs = torch.cat(rgbs, dim=0)
    view_directions = torch.cat(view_directions, dim=0).cpu()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.double().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(rgbs.double().cpu().numpy())

    ind = None
    if remove_outliers:
        CONSOLE.print("Cleaning Point Cloud")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=std_ratio)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Cleaning Point Cloud")
        if ind is not None:
            view_directions = view_directions[ind]

    # either estimate_normals or normal_output_name, not both
    if estimate_normals:
        if normal_output_name is not None:
            CONSOLE.rule("Error", style="red")
            CONSOLE.print("Cannot estimate normals and use normal_output_name at the same time", justify="center")
            sys.exit(1)
        CONSOLE.print("Estimating Point Cloud Normals")
        pcd.estimate_normals()
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Estimating Point Cloud Normals")
    elif normal_output_name is not None:
        normals = torch.cat(normals, dim=0)
        if ind is not None:
            # mask out normals for points that were removed with remove_outliers
            normals = normals[ind]
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    # re-orient the normals
    if reorient_normals:
        normals = torch.from_numpy(np.array(pcd.normals)).float()
        mask = torch.sum(view_directions * normals, dim=-1) > 0
        normals[mask] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals.double().cpu().numpy())

    return pcd


def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image], image_type: str) -> npt.NDArray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    Adapted from `diffusers.image_processor.VaeImageProcessor.pil_to_numpy`.
    """
    if not isinstance(images, list):
        images = [images]
    if image_type == "indice":
        scale_factor = 1.
    elif image_type == "depth":
        scale_factor = 20.
    else:
        scale_factor = 255.
    images = [np.array(image).astype(np.float32) / scale_factor for image in images]
    images = np.stack(images, axis=0)

    return images


def numpy_to_pt(images: npt.NDArray, image_type: str) -> torch.FloatTensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    Adapted from `diffusers.image_processor.VaeImageProcessor.pil_to_numpy`.
    """
    if images.ndim == 3 and image_type != "indice":
        images = images[..., None]

    if image_type != "indice":
        images = images.transpose(0, 3, 1, 2) # BHWC to BCHW
    images = torch.from_numpy(images) # BCHW or BHW
    # if image_type is rgb, then c = 3; if image_type is depth, then c = 1
    return images


def format_poses(*args) -> Union[Tensor, npt.NDArray]:
    """
    Args:
        case1: given rots and trans matrix
        case2: given yaw, roll, pitch

    Returns:
        poses: camera to world matrix
    """
    if len(args) == 2:
        rots: Union[Tensor, npt.NDArray] = args[0]
        trans: Union[Tensor, npt.NDArray] = args[1]

        shape = rots.shape[:-2]
        if isinstance(rots, Tensor):
            poses = torch.eye(4).broadcast_to([*shape, 4, 4]).clone().to(rots.device)
        elif isinstance(rots, np.ndarray):
            poses = np.broadcast_to(np.eye(4), [*shape, 4, 4]).copy()
        poses[..., :3, :3] = rots
        poses[..., :3, 3] = trans

        return poses


def deformat_poses(poses, return_angles: Optional[bool]=False) -> Union[Tensor, npt.NDArray]:
    """
    Args:
        poses: camera to world matrix
        
    Returns:
        case1: rots and trans
        case2: yaw, roll, pitch
    """
    if return_angles:
        pass
    
    rots = poses[..., :3, :3]
    trans = poses[..., :3, 3]
    
    return rots, trans


def sample_trajectory(all_valid_ids: list, all_extrins: npt.NDArray, all_depths: List[str],
                      intrins: npt.NDArray, image_size: List[int]) -> list:
    """
    Args:
        all_extrins: list of extrinsic matrices, shape (N, 4, 4)
        intrins: intrinsic matrix, shape (3, 3)
    """
    height, width = image_size
    # NOTE: there is a hyper-parameter to set the reference point
    ref_points = np.array(
        [[width / 2, 0, 6], [0, height / 2, 6],
         [width / 2, height, 6], [width, height / 2, 6]])
    ref_points[:, :2] *= ref_points[:, 2:3]
    if intrins.shape[0] == 4:
        ref_points = np.concatenate([ref_points, [[1], [1], [1], [1]]], axis=-1)
    last_id = 0

    all_ids, ids = [], []
    for i in all_valid_ids:
        ref_extrin = all_extrins[last_id]
        extrin = all_extrins[i]
        # avoid nan or inf values in extrinsics
        if (np.any(np.any(np.isnan(extrin)) or np.any(np.isinf(extrin)))
            or Image.open(all_depths[i]).mode != 'I'):
            continue
        all_ids.append(i)
        if i == 0:
            last_id = 0
            ids.append(i)
        else:
            # transfrom matrix from current image coordinate to reference image coordinate
            transform = np.matmul(np.matmul(intrins, np.linalg.inv(extrin)),
                                  np.linalg.inv(np.matmul(intrins, np.linalg.inv(ref_extrin))))
            points = np.array([np.matmul(transform, ref_point) for ref_point in ref_points])
            points[:, :2] /= points[:, 2:3]
            xs, ys = points[:, 0], points[:, 1]
            dxs, dys = np.min([width - xs, xs], axis=0), np.min([height - ys, ys], axis=0)
            # NOTE: hyper-parameter to set the threshold
            # TODO: fix the gap
            if not (np.all(dxs < width // 8) and np.all(dys < height // 8)):
                continue
            else:
                last_id = i
                ids.append(i)
    return all_ids, ids


def get_correspondence_np(depth, transform, K, **kwargs) -> npt.NDArray:
    h, w = depth.shape # [H, W]
    n = transform.shape[0]
    # backprojection
    xy_coords = kwargs.get("xy_coords", None)
    if xy_coords is None:
        i, j = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='xy')
        xy_coords = np.stack([i, j], axis=-1).reshape(-1, 2) # [H, W, 2]
    points_2d = np.concatenate([xy_coords, np.ones_like(xy_coords[..., :1])], axis=-1) @ np.linalg.inv(K[:3, :3]).T # [H*W, 3]
    points_2d = points_2d.reshape(h, w, 3)
    points_2d[..., 2] = 1
    points_2d *= depth[..., np.newaxis] # [H, W, 3]
    points_2d = points_2d.reshape(-1, 3)[np.newaxis].repeat(n, axis=0) # [N, H*W, 3]
    R, T = transform[:, :3, :3], transform[:, :3, 3]
    points_2d = points_2d @ R.transpose(0, 2, 1) + T[:, np.newaxis] # [N, H*W, 3]
    # projection
    points_2d = points_2d @ K[:3, :3][np.newaxis].repeat(n, axis=0).transpose(0, 2, 1)
    points_2d = points_2d.reshape(n, h, w, 3)
    points_2d = points_2d[..., :2] / (points_2d[..., -1:] + 1e-7) # [N, H, W, 2]

    points_2d[:, depth == 0] = -1e10
    return points_2d


def get_correspondence(depth, transform, K, **kwargs) -> torch.Tensor:
    n, h, w = depth.shape
    # backprojection
    xy_coords = kwargs.get("coords", None)
    if xy_coords is None:
        i, j = torch.meshgrid(torch.linspace(0, w-1, w), torch.linspace(0, h-1, h), indexing='xy')
        xy_coords = torch.stack([i, j], dim=-1).reshape(-1, 2) # [H*W, 2]
    points_2d = torch.concat([xy_coords, torch.ones_like(xy_coords[..., :1])], dim=-1).to(K.device) @ \
                                            torch.linalg.inv(K[:3, :3].float()).to(xy_coords.dtype).T
    points_2d = points_2d.reshape(h, w, 3).to(depth.dtype)
    points_2d = points_2d[None].repeat(n, 1, 1, 1)
    points_2d[..., -1] = 1
    points_2d *= depth[..., None]
    R, T = transform[:, :3, :3], transform[:, :3, 3]
    points_2d = rearrange(points_2d, 'n h w c -> n c (h w)')
    points_2d = R @ points_2d + T[..., None]
    # projection
    points_2d = K[None].repeat(n, 1, 1)[..., :3, :3] @ points_2d
    points_2d = rearrange(points_2d, 'n c (h w) -> n h w c', h=h, w=w)
    points_2d[..., :2] = points_2d[..., :2] / (points_2d[..., -1:] + 1e-7) # [..., H*W, 2]

    points_2d[depth == 0] = -1e4 # fp16
    return points_2d


def depth_inv_norm_transforms(depth: torch.Tensor):
    h, w = depth.shape
    depth_valid_mask = depth > 0
    depth_inv = 1. / (depth + 1e-6)
    depth_max = depth_inv[depth_valid_mask].max()
    depth_min = depth_inv[depth_valid_mask].min()
    depth_inv_norm_full = (depth_inv - depth_min) / (depth_max - depth_min + 1e-6) * 2 - 1 # in range [-1, 1]
    depth_inv_norm_full[~depth_valid_mask] = -2
    depth_inv_norm_small = F.interpolate(depth_inv_norm_full[None, None], size=(h // 8, w // 8), mode="bicubic", align_corners=False)[0][0]
    return depth_inv_norm_small


def setup_logger(name: str, level: str = None, path: str = None) -> MultiProcessAdapter:
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing with additional FileHandler.
    """
    if PartialState._shared_state == {}: PartialState()
    if level is None:
        level = "DEBUG" if os.getenv("DEBUG", False) else "INFO"
    name = name.split('.')[0]
    if path is None:
        os.makedirs("logs", exist_ok=True)
        path = f"logs/{name}"

    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f"{path}.log")
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(file_handler)

    return MultiProcessAdapter(logger, {})


def patchize(images: Tensor, patch_size: int = 224) -> Tensor:
    if images.ndim == 3:
        images = images[None]

    # resize short
    height, width = images.shape[-2:]
    new_width = int(224 / height * width)
    images = F.interpolate(images, (224, new_width), mode="bilinear")
    images = images.view(-1, 3, *images.shape[-2:])

    # center crop
    width = images.shape[-1]
    crop_size = int((width - 224) / 2)
    images = images[..., crop_size : crop_size + 224]
    return images


def module_wrapper(ddp_or_model):
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        cast_module = type(ddp_or_model.module)
        return cast(cast_module, ddp_or_model.module)
    return ddp_or_model


""" Adapted profiler from nerfstudio to support debugging and measure time in cuda """

class TimeFunction(profiler._TimeFunction):

    def __exit__(self, *args, **kwargs):
        while self._profiler_contexts:
            context = self._profiler_contexts.pop()
            context.__exit__(*args, **kwargs)
        if profiler.PROFILER:
            torch.cuda.synchronize()
            profiler.PROFILER[0].update_time(self.name, self.start, profiler.time.time())

        # print profiler at each step in debug mode
        if profiler.PROFILER:
            profiler.PROFILER[0].print_profile()


@profiler.decorate_all([profiler.check_profiler_enabled, check_local_main_thread])
class Profiler(profiler.Profiler):
    """Profiler class enabled with `is_local_main_process`"""

    def __init__(self, config: profiler.cfg.LoggingConfig):
        self.config = config
        self.profiler_dict = {}

    def update_time(self, func_name: str, start_time: float, end_time: float):
        """update the profiler dictionary with running averages of durations

        Args:
            func_name: the function name that is being profiled
            start_time: the start time when function is called
            end_time: the end time when function terminated
        """
        val = end_time - start_time
        func_dict = self.profiler_dict.get(func_name, {"val": 0, "step": 0})
        prev_val = func_dict["val"]
        prev_step = func_dict["step"]
        self.profiler_dict[func_name] = {"val": (prev_val * prev_step + val) / (prev_step + 1), "step": prev_step + 1}

    def print_profile(self):
        """helper to print out the profiler stats"""
        profiler.CONSOLE.print("Printing profiling stats, from longest to shortest duration in seconds")
        sorted_keys = sorted(
            self.profiler_dict.keys(),
            key=lambda k: self.profiler_dict[k]["val"],
            reverse=True,
        )
        for k in sorted_keys:
            val = f"{self.profiler_dict[k]['val']:0.4f}"
            profiler.CONSOLE.print(f"{k:<20}: {val:<20}")


def time_function(name_or_func: Union[profiler.CallableT, str]) -> Union[profiler.CallableT, profiler.ContextManager[Any]]:
    """Profile a function or block of code. Can be used either to create a context or to wrap a function.

    Args:
        name_or_func: Either the name of a context or function to profile.

    Returns:
        A wrapped function or context to use in a `with` statement.
    """
    return TimeFunction(name_or_func)


def setup_profiler(config: profiler.cfg.LoggingConfig, log_dir: Path):
    """Initialization of profilers"""
    global PYTORCH_PROFILER
    if is_local_main_process() and os.getenv("MEASURE_TIME", False):
        profiler.PROFILER.append(Profiler(config))
        if config.profiler == "pytorch":
            PYTORCH_PROFILER = profiler.PytorchProfiler(log_dir)
