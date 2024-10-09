# https://github.com/cobanov/depth2normal

import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from concurrent import futures


class DepthToNormalMap:

    def __init__(self, h, w, depth_map_path: str, max_depth: int = 255) -> None:
        # self.depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
        self.depth_map = np.fromfile(depth_map_path).reshape(h, w)
        self.depth_map = (self.depth_map * 1000).astype(np.uint16)

        if self.depth_map is None:
            raise ValueError(
                f"Could not read the depth map image file at {depth_map_path}"
            )
        self.max_depth = max_depth

    def convert(self, output_path: str) -> None:
        rows, cols = self.depth_map.shape

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Calculate the partial derivatives of depth with respect to x and y
        dx = cv2.Sobel(self.depth_map, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(self.depth_map, cv2.CV_32F, 0, 1)

        # Compute the normal vector for each pixel
        normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        # the normal has values in range [-1, 1]
        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal = (normal + 1) * 127.5
        normal = normal.clip(0, 255).astype(np.uint8)

        # Save the normal map to a file
        normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, normal_bgr)


def process_scene(height, width, scene, idx):
    tqdm.write(f"Processing scene {idx}, {scene}.")
    os.makedirs(os.path.join(args.output_path, scene), exist_ok=True)
    scene_path = os.path.join(args.depth_path, scene)
    depth_files = list(filter(lambda fname: fname.endswith(".bin"), os.listdir(scene_path)))

    for depth_file in tqdm(depth_files):
        try:
            converter = DepthToNormalMap(height, width, os.path.join(scene_path, depth_file))
            converter.convert(os.path.join(args.output_path, scene, depth_file.replace(".bin", ".jpg")))
        except Exception as e:
            print(e)


def main():

    height, width = 584, 876
    all_scenes = os.listdir(args.depth_path)
    scene_idx = [i for i in range(len(all_scenes))]

    process_scene_partial = partial(process_scene, height, width)
    with futures.ProcessPoolExecutor(max_workers=128) as executor:
        executor.map(process_scene_partial, all_scenes, scene_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_path", type=str, default="data/scannetpp_processed/depths", required=False)
    parser.add_argument("--output_path", type=str, default="data/scannetpp_processed/normals", required=False)
    args = parser.parse_args()
    main()