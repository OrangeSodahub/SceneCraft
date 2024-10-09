import os
import json
import numpy as np
import argparse
import sys
sys.path.append(".")
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scenecraft.data.dataset import ScannetDataset, HypersimDataset
from scenecraft.utils import load_from_json


def extract_hypersim(args):
    os.makedirs(args.output_path, exist_ok=True)
    dataset = HypersimDataset(scene_ids=args.scene_id)
    for scene in tqdm(dataset):
        # layouts
        images, extrins, intrins, metas = scene
        scene_id = metas["scene_id"]
        save_path = os.path.join(args.output_path, f"{scene_id}.json")
        bbox_extents = metas["bounding_box_object_aligned_2d_extents"]
        bbox_orientation = metas["bounding_box_object_aligned_2d_orientation"]
        bbox_positions = metas["bounding_box_object_aligned_2d_positions"]
        thetas = np.degrees(np.arctan2(bbox_orientation[..., 1, 0], bbox_orientation[..., 0, 0])).tolist()
        labels = list(map(lambda x: str(x), metas["obj_class"].tolist()))
        bboxes = np.concatenate([bbox_positions, bbox_extents], axis=-1).tolist()
        data = dict(bboxes=bboxes, labels=labels, thetas=thetas)
        with open(save_path, 'w') as f:
            json.dump(data, f)
        # cameras
        save_path = os.path.join(args.output_path, f"{scene_id}_cameras.json")
        camera_paths = [
            {
                "camera_to_world": extrin.reshape(-1).tolist(),
                "fov": 60,
            }
            for extrin in extrins
        ]
        data = dict(key_frames=[], camera_paths=camera_paths)
        with open(save_path, 'w') as f:
            json.dump(data, f)


def extract_scannetpp(args):
    os.makedirs(args.output_path, exist_ok=True)
    data_dir = os.path.join(args.data_root, 'scannet_instance_data')
    image_dir = os.path.join(args.data_root, 'posed_images')
    all_scenes = list(sorted(os.listdir(image_dir)))
    print(f"======> Load {len(all_scenes)} scenes.")
    cat_ids2class = {nyu40id: i for i, nyu40id in enumerate(ScannetDataset.CAT_IDS)}
    for scene in tqdm(all_scenes, leave=False):
        save_path = os.path.join(args.output_path, f"{scene}.json")
        label_path = os.path.join(data_dir, f'{scene}_aligned_bbox.npy')
        bbox_label = np.load(label_path)
        bbox_label = np.array(list(filter(lambda x: x[-1] in cat_ids2class, bbox_label)))
        bboxes, labels = bbox_label[:, :6], bbox_label[:, -1]
        bboxes = bboxes.tolist()
        labels = list(map(lambda x: str(cat_ids2class[int(x)]), labels))
        data = dict(bboxes=bboxes, labels=labels)
        with open(save_path, 'w') as f:
            json.dump(data, f)


def draw_open3d(args):
    import open3d as o3d

    layout_file = Path(args.layout_file)
    layouts = load_from_json(layout_file)

    if "thetas" not in layouts:
        layouts["thetas"] = [0.] * len(layouts["bboxes"])

    assets = []

    for bbox, label, theta in zip(layouts["bboxes"], layouts["labels"], layouts["thetas"]):
        if int(label) == 22:
            continue
        trans = bbox[:3]
        dx, dy, dz = bbox[3:]
        bbox = o3d.geometry.TriangleMesh.create_box(width=dx, height=dy, depth=dz)
        theta = -np.deg2rad(theta)
        theta -= np.pi / 2
        rotation = R.from_rotvec(theta * np.array([0, 0, 1]))
        rotation_matrix = rotation.as_matrix()
        bbox.rotate(rotation_matrix, center=bbox.get_center())
        trans[0] -= dx / 2
        trans[1] -= dy / 2
        trans[2] -= dz / 2
        bbox.translate(trans)
        bbox.compute_vertex_normals()
        color = list(map(lambda x: x / 255., HypersimDataset.COLORS[int(label)]))
        bbox.paint_uniform_color(color)
        assets.append(bbox)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for asset in assets:
        vis.add_geometry(asset)

    ctr = vis.get_view_control()
    parameters = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=640, height=480, fx=640, fy=640, cx=320, cy=240)
    parameters.intrinsic = intrinsic

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
    extrinsic[2, 3] = -5
    parameters.extrinsic = extrinsic
    
    ctr.convert_from_pinhole_camera_parameters(parameters)

    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)

    image_path = layout_file.parent / "layout.png"
    o3d.io.write_image(str(image_path), o3d.geometry.Image((np.asarray(image) * 255).astype(np.uint8)))

    combined_mesh = o3d.geometry.TriangleMesh()
    for asset in assets:
        combined_mesh += asset
    o3d.io.write_triangle_mesh(str(layout_file.parent / f"{str(layout_file.parent.name)}.obj"), combined_mesh, write_ascii=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_layout", action="store_true")
    parser.add_argument("--convert_to_open3d", action="store_true")
    parser.add_argument("--convert_to_mayavi", action="store_true")
    parser.add_argument("--layout_file", type=str, default=None, required=False)
    parser.add_argument("--dataset", type=str, default="hypersim", required=False)
    parser.add_argument("--data_root", type=str, default="./data/scannet", required=False, help="Path to local dataset.")
    parser.add_argument("--scene_id", type=str, default=None, required=False)
    parser.add_argument("--output_path", type=str, default="./data/scannet/nerstudio_aabb", required=False)
    args = parser.parse_args()

    if args.extract_layout:
        if args.dataset == "hypersim":
            extract_hypersim(args)
        elif args.dataset == "scannetpp":
            extract_scannetpp(args)

    elif args.convert_to_open3d:
        draw_open3d(args)
