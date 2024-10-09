import os
import argparse
import torch
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image


generate_diffusion_2d_dir = ""
nerf_model_dir = ""


def process_2d_generation():
    diffusion_2d_generations = list(filter(lambda fname: fname.endswith(".png"), os.listdir(generate_diffusion_2d_dir)))
    os.makedirs(os.path.join(generate_diffusion_2d_dir, "seperated"), exist_ok=True)
    for file in tqdm(diffusion_2d_generations):
        image = Image.open(os.path.join(generate_diffusion_2d_dir, file)).convert("RGB")
        width, height = image.size
        new_width = width // 3
        layout_image, generate_image, depth_image = (
            image.crop((0, 0, new_width, height)),
            image.crop((new_width, 0, new_width * 2, height)),
            image.crop((new_width * 2, 0, width, height))
        )
        image_id = file.removesuffix(".png")
        layout_image.save(os.path.join(generate_diffusion_2d_dir, "seperated", f"{image_id}_layout.png"))
        generate_image.save(os.path.join(generate_diffusion_2d_dir, "seperated", f"{image_id}_generate.png"))
        depth_image.save(os.path.join(generate_diffusion_2d_dir, "seperated", f"{image_id}_depth.png"))


def extract_guide_image():
    nerf_models = list(filter(lambda fname: fname.endswith(".ckpt"), os.listdir(nerf_model_dir)))

    guide_images = defaultdict(defaultdict)
    for nerf_model in tqdm(nerf_models):
        state_dict = torch.load(os.path.join(nerf_model_dir, nerf_model), map_location="cpu")
        meta_infos = state_dict["meta_infos"]
        guide_buffers = meta_infos["nerf_guide_buffers"]
        for k, v in tqdm(guide_buffers.items(), leave=False):
            image_idx, step = k.split("_")
            image = v["rgb"][0].permute(1, 2, 0).numpy() * 255.
            guide_images[image_idx][step] = Image.fromarray(image.astype(np.uint8))

    output_path = "guide_buffers"
    os.makedirs(os.path.join(nerf_model_dir, output_path), exist_ok=True)
    for image_idx, buffer in tqdm(guide_images.items()):
        os.makedirs(os.path.join(nerf_model_dir, output_path, image_idx), exist_ok=True)
        for step, image in tqdm(buffer.items(), leave=False):
            image.save(os.path.join(nerf_model_dir, output_path, image_idx, f"{step}.png"))

    with open(os.path.join(nerf_model_dir, "guide_buffer.pkl"), "wb") as pickle_file:
        pickle.dump(guide_images, pickle_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_2d_generation", action="store_true")
    parser.add_argument("--extract_guide_image", action="store_true")
    args = parser.parse_args()

    if args.process_2d_generation:
        process_2d_generation()

    elif args.extract_guide_image:
        extract_guide_image()