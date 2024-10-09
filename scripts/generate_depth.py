import os
import json
import torch
import argparse
import trimesh
import numpy as np
from tqdm import tqdm
from PIL import Image
from functools import partial
from concurrent import futures
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scenecraft.renderer import Model
from scenecraft.utils import qvec2rotmat, format_poses, colorize_depth


def process_scene(height, width, scene, idx):
    tqdm.write(f"Processing scene {idx}, {scene}.")
    os.makedirs(os.path.join(args.output_path, scene), exist_ok=True)
    raw_scene_path = os.path.join(args.raw_data_root, scene)
    scene_path = os.path.join(args.data_root, scene)
    transform = json.load(open(os.path.join(scene_path, 'dslr', 'nerfstudio', f'transforms_2.json'), 'r'))
    fl_x, fl_y, cx, cy = transform['fl_x'], transform['fl_y'], transform['cx'], transform['cy']
    intrin = (np.array([[fl_x, 0, cx, 0], [0, fl_y, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
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
    image_ids = colmap[:, 9].tolist()
    all_extrins = np.linalg.inv(format_poses(rots, trans))
    raw_image_ids = os.listdir(os.path.join(scene_path, 'dslr', 'resized_images_2'))
    if len(raw_image_ids) != len(image_ids):
        tqdm.write(f"Scene {scene} has wrong number of images.")

    mesh_path = os.path.join(raw_scene_path, "scans", "mesh_aligned_0.05.ply")
    mesh = trimesh.load(mesh_path)

    for id, extrin in tqdm(zip(image_ids, all_extrins)):
        if os.path.exists(os.path.join(args.output_path, scene, f"{id[:-4]}.bin")):
            if np.fromfile(os.path.join(args.output_path, scene, f"{id[:-4]}.bin")).size == height * width:
                tqdm.write(f"Skipped id {id}.")
                continue
        # dense depth from mesh
        # cpp version will be faster:
        # https://github.com/apple/ml-hypersim/blob/main/code/cpp/tools/generate_ray_intersections/main.cpp
        # other options:
        # https://pyrender.readthedocs.io/en/latest/examples/quickstart.html#minimal-example-for-offscreen-rendering
        try:
            rays_o, rays_d = Model._get_rays(height, width, torch.Tensor(intrin), torch.Tensor(extrin), normalize_dir=True)
            rays_o = rays_o.reshape(-1, 3).numpy()
            rays_d = rays_d.reshape(-1, 3).numpy()
            coords = np.array(list(np.ndindex(height, width))).reshape(height, width, -1).transpose(1, 0, 2).reshape(-1, 2)
            w2c = np.linalg.inv(extrin)
            points, index_ray, _ = RayMeshIntersector(mesh).intersects_location(rays_o, rays_d, multiple_hits=False)
            depth = (points @ w2c[:3, :3].T + w2c[:3, -1])[:,-1]
            pixel_ray = coords[index_ray]
            depth_image = np.full([height, width], np.nan)
            depth_image[pixel_ray[:, 0], pixel_ray[:, 1]] = depth
            depth_image.tofile(os.path.join(args.output_path, scene, f"{id[:-4]}.bin"))
            tqdm.write(f"Processing id {id}, scene no.{idx}.")

            depth_colormap = colorize_depth(depth_image)
            depth_colormap = Image.fromarray(depth_colormap.astype(np.uint8))
            depth_colormap.save(os.path.join(args.output_path, scene, f"{id[:-4]}.png"))
        except Exception as e:
            print(e)


def main(args):

    height, width = 584, 876
    all_scenes = os.listdir(args.raw_data_root)
    scene_idx = [i for i in range(len(all_scenes))]

    process_scene_partial = partial(process_scene, height, width)
    with futures.ProcessPoolExecutor(max_workers=128) as executor:
        executor.map(process_scene_partial, all_scenes, scene_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_root", type=str, default="data/scannetpp/data", required=False)
    parser.add_argument("--data_root", type=str, default="data/scannetpp_processed/data", required=False)
    parser.add_argument("--output_path", type=str, default="data/scannetpp_processed/depths", required=False)
    args = parser.parse_args()
    main(args)
