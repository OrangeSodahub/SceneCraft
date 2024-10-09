from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import imageio
import random
import datetime
import json
import yaml
from tqdm import tqdm
from typing import List
from PIL import Image
from pathlib import Path
from torchvision import transforms
from diffusers import UniPCMultistepScheduler, DDIMScheduler
from scenecraft.data.dataset import SCALE_H, RAW_DATASETS
from scenecraft.model import StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline, \
                                UNet2DConditionModel, ControlNetModel, MultiControlNetModel
from scenecraft.finetune.train_controlnet_sd import make_class_embeddings
from scenecraft.renderer import DatasetRender, LayoutRender, ExportTSDFMesh, ExportPointCloud
from scenecraft.utils import colorize_depth, depth_inv_norm_transforms
from scenecraft.trainer import SceneCraftTrainerConfig


def find_save_id(output_path: str, prefix: str = "") -> int:
    paths = list(sorted(filter(lambda fname:
                    fname.startswith(f"{prefix}_") and
                    os.path.isdir(os.path.join(output_path, fname)) and
                    os.listdir(os.path.join(output_path, fname)),
                os.listdir(output_path))))
    id = 0 if not paths else int(paths[-1].split("_")[1]) + 1
    return id


def generate_guidance(output_path: str, semantics_path: str, scene_id: str, **kwargs) -> None:
    seed = kwargs.pop("seed", 0)
    batch_size = kwargs.pop("batch_size", 1)
    sequential_input = kwargs.pop("sequential_input", False)
    os.makedirs(output_path, exist_ok=True)

    # unet
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.base_model_path, subfolder="unet", torch_dtype=torch.float16
    )

    # controlnet
    if args.checkpoint_path is None or args.no_cond:
        controlnet = None
    elif "multi_control" not in args.checkpoint_path:
        controlnet = ControlNetModel.from_pretrained(
            args.checkpoint_path, subfolder=f"{args.checkpoint_subfolder.strip('/')}/controlnetmodel", torch_dtype=torch.float16)
    else:
        controlnet: MultiControlNetModel = MultiControlNetModel.from_pretrained(
            args.checkpoint_path, subfolder=f"{args.checkpoint_subfolder.strip('/')}", torch_dtype=torch.float16
        )
        assert isinstance(args.controlnet_conditioning_scale, (list, tuple))

    # pipeline
    if controlnet is not None:
        pipe_method = StableDiffusionControlNetInpaintPipeline if args.inpaint else StableDiffusionControlNetPipeline
        pipe = pipe_method.from_pretrained(
            args.base_model_path, unet=unet, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model_path, unet=unet, torch_dtype=torch.float16, safety_checker=None,
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if args.lora_layer_path:
        pipe.unet.load_attn_procs(args.lora_layer_path)

    pipe.to(torch_device=torch.device('cuda'))
    pipe.enable_xformers_memory_efficient_attention()
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # load_dataset
    files = list(sorted(os.listdir(semantics_path)))
    source_image_files = list(filter(lambda fname: fname.endswith(".png"), files))
    source_files = list(filter(lambda fname: fname.endswith(".npz"), files))
    if controlnet is not None:
        indice_to_embedding = make_class_embeddings(args.dataset, ("random" if args.condition_type == "embedding" else "clip"),
                                                    pipe.tokenizer, pipe.text_encoder, torch.device('cuda')).to(torch.float16) \
                                if args.condition_type == "embedding" or args.condition_type == "clip_embedding" else None

    # indices
    batch_len = len(source_files) // batch_size * batch_size
    batch_indices = list(range(batch_len))
    if not sequential_input:
        random.shuffle(batch_indices)
    batch_indices = [batch_indices[i : i + batch_size] for i in range(0, len(source_files), batch_size)]

    latents = None
    height, width = None, None
    # get preparation for epipolar attention module
    if args.enable_epi:
        args.fix_noise = True
        # get src model
        midas = DPT_BEiT_L_512().to(pipe.device)
        # get src image for the first generation
        height = SCALE_H[args.dataset]
        width = int(height / RAW_DATASETS[args.dataset].ori_h * RAW_DATASETS[args.dataset].ori_w)
        midas_transforms = get_midas_transforms(height, keep_aspect_ratio=True)
        src_image: Image.Image = pipe(args.prompt,
                                      height=height, width=width,
                                      num_inference_steps=20,
                                      latents=None,
                                      generator=generator,
                                      guidance_scale=args.guidance_scale).images[0]

    if args.fix_noise:
        if height is None or width is None:
            height, width = np.load(os.path.join(semantics_path, source_files[0]))["labels"].shape
        latents = pipe.prepare_latents(batch_size,
                                        pipe.unet.config.in_channels,
                                        height, width,
                                        torch.float16, pipe.device, generator)

    prompts = [args.prompt] * batch_size
    for i, indices in tqdm(enumerate(batch_indices), leave=False):
        source_images = [source_image_files[i] for i in indices]
        sources = [source_files[i] for i in indices]
        sample_ids = [source_image.removesuffix(".png") for source_image in source_images]
        source_paths = [os.path.join(semantics_path, source) for source in sources]
        sources = [np.load(source_path) for source_path in source_paths]
        rgb_source_image_paths = [os.path.join(semantics_path, source_image) for source_image in source_images]
        rgb_source_images = [Image.open(rgb_source_image_path).convert("RGB") for rgb_source_image_path in rgb_source_image_paths]
        rgb_real_images = [None] * len(rgb_source_images)
        if args.inpaint:
            rgb_real_image_paths = [RAW_DATASETS[dataset].get_rgb_image_path(scene_id, sample_id) for sample_id in sample_ids]
            rgb_real_images = [Image.open(rgb_real_path).convert("RGB") for rgb_real_path in rgb_real_image_paths]

        image_width, image_height = rgb_source_images[0].size
        rgb_source_images = [rgb_source_image.crop((0, 0, image_width, image_height // 2))
                             for rgb_source_image in rgb_source_images]
        source_images = rgb_source_images if args.condition_type == "rgb" else \
                            [Image.fromarray(source["labels"].astype(np.uint8)) for source in sources]
        depth_images = [Image.fromarray(source["depths"]) for source in sources]
        control_images = [source_images, depth_images] if isinstance(controlnet, MultiControlNetModel) else [source_images]

        extra_kwargs = dict()

        # depth kwargs
        if pipe.unet.config.in_channels == 5:
            if "depths" not in extra_kwargs:
                scale_h = SCALE_H[args.dataset]
                depths = [transforms.Resize(scale_h, interpolation=transforms.InterpolationMode.NEAREST_EXACT)(depth) for depth in depth_images]
                extra_kwargs.update(depths=depths)

            depth_inv_norm_smalls = [depth_inv_norm_transforms(transforms.ToTensor()(depth)[0])
                                        for depth in extra_kwargs["depths"]]
            height, width = extra_kwargs["depths"][0].shape
            extra_kwargs.update(depth_inv_norm_smalls=depth_inv_norm_smalls, height=height, width=width)

        # control kwargs
        if isinstance(pipe, StableDiffusionControlNetPipeline):
            extra_kwargs.update(image=control_images, condition_type=args.condition_type, 
                                indice_to_embedding=indice_to_embedding,
                                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                                no_depth_cond=args.no_depth_cond)

        if isinstance(pipe, StableDiffusionControlNetInpaintPipeline):
            width, height = rgb_source_images[0].size
            mask_images = np.zeros((len(rgb_source_images), 1, height, width))
            mask_images[:, :, :, : width // 2] = 1
            extra_kwargs.update(init_image=rgb_real_images,
                                mask_image=mask_images, strength=0.5)

        guide_images: List[Image.Image] = pipe(prompts,
                                               num_inference_steps=20,
                                               latents=latents,
                                               generator=generator,
                                               guidance_scale=args.guidance_scale,
                                               **extra_kwargs).images

        for guide_image, rgb_source_image, rgb_real_image, depth_image, sample_id in zip(
            guide_images, rgb_source_images, rgb_real_images, depth_images, sample_ids
        ):

            if guide_image.size != rgb_source_image.size:
                guide_image = guide_image.resize(rgb_source_image.size, resample=Image.Resampling.BILINEAR)

            if args.enable_epi and args.no_cond:
                refer_rgb_image = Image.open(
                    os.path.join("data/scannetpp_processed", RAW_DATASETS[args.dataset].get_relative_rgb_image_path(args.scene_id, sample_id)))

                if refer_rgb_image.size != rgb_source_image.size:
                    refer_rgb_image = refer_rgb_image.resize(rgb_source_image.size, resample=Image.Resampling.BILINEAR)

                image = Image.new("RGB", (image_width, image_height))
                image.paste(refer_rgb_image, (0, 0))
                image.paste(guide_image, (0, image_height // 2))

            else:
                image = Image.new("RGB", (image_width * 3, image_height // 2))
                image.paste(rgb_source_image, (0, 0))
                image.paste(guide_image, (image_width, 0))
                if args.inpaint:
                    image.paste(rgb_real_image, (image_width * 2, 0))
                else:
                    image.paste(colorize_depth(depth_image, 0.1, 3, output_type="pil"), (image_width * 2, 0))

            image.save(f"{output_path}/{sample_id}.png")

            if kwargs.get("save_real", False):
                dataset = kwargs["dataset"]
                scene_id = kwargs["scene_id"]
                real_image_path = RAW_DATASETS[dataset].get_rgb_image_path(scene_id, sample_id)
                real_image = Image.open(real_image_path)
                real_image.save(f"{output_path}/{sample_id}_real.png")

        # update src image for next generation
        if args.enable_epi:
            src_image = guide_images[-1]
            src_extrin = extrins[-1:]
            src_intrin = intrins[-1:]


def generate_layout(output_path):
    semantic_images_path = os.path.join(output_path, "semantic_images")
    os.makedirs(semantic_images_path, exist_ok=True)

    print("Generating semantics begins.")
    layout_render = LayoutRender(dataset_type=args.dataset)
    layout_render.render_scene(output_path, scene_id=args.scene_id, voxel_size=args.voxel_size)


def main():
    seed = args.seed or 0
    if args.nerf or args.mesh or args.point:
        nerf_config: SceneCraftTrainerConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader)
        prompt = nerf_config.pipeline.prompt.strip(".")

        config_name = Path(args.load_config).parent.name
        *_, dataset_type, scene_id = config_name.split("-")
        args.dataset = dataset_type
        args.scene_id = scene_id

    assert args.dataset is not None and args.scene_id is not None, "dataset and scene_id need to be specified."
    output_path = os.path.join(args.output_dir, args.dataset, args.scene_id)
    os.makedirs(output_path, exist_ok=True)

    if not (args.layout or args.diffusion or args.nerf or args.mesh or args.point):
        raise RuntimeError

    if args.layout:
        generate_layout(output_path)

    # diffusion outputs
    if args.diffusion:
        assert args.prompt is not None, "Need a prompt!"
        if args.prompt == "This is one view of a room.":
            diffusion_path = os.path.join(output_path, "diffusion2d_real")
        else:
            generate_id = find_save_id(output_path, prefix="diffusion2d")
            diffusion_path = os.path.join(output_path, f"diffusion2d_{str(generate_id).zfill(2)}_{'_'.join(args.prompt.lower().strip('.').split(' '))}")
        print(f"Results will be saved to {diffusion_path}")

        # prepare the semantic layout and depth
        semantic_images_path = os.path.join(output_path, "semantic_images")
        if not os.path.exists(semantic_images_path) or len(os.listdir(semantic_images_path)) == 0:
            generate_layout(output_path)

        # generate diffusion outputs
        print("Generating guidance begins.")
        generate_guidance(diffusion_path, semantic_images_path, seed=seed,
                          dataset=args.dataset,
                          scene_id=args.scene_id,
                          batch_size=args.batch_size,
                          sequential_input=args.sequential_input,
                          save_real=args.save_real)

        # generate videos
        image_files = sorted(os.listdir(diffusion_path))
        with imageio.get_writer(f"{diffusion_path.strip('.')}.mp4", fps=10) as video:
            for image_file in tqdm(image_files):
                if not image_file.endswith(".png") or "real" in image_file: continue
                frame = Image.open(os.path.join(diffusion_path, image_file)).convert("RGB")
                video.append_data(np.array(frame))

        # save diffusion version
        meta_info = dict(
            dataset=args.dataset,
            scene_id=args.scene_id,
            base_model=args.base_model_path,
            prompt=args.prompt,
            frames=len(image_files),
            checkpoint_path=args.checkpoint_path,
            checkpoint_subfolder=args.checkpoint_subfolder,
            condition_type=args.condition_type,
            guidance_scale=args.guidance_scale,
            control_scale=args.controlnet_conditioning_scale,
            output_file=f"{diffusion_path.strip('.')}.mp4",
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        with open(f"{diffusion_path}/meta.json", 'w') as f:
            json.dump(meta_info, f)

        # print meta info
        for k, v in meta_info.items():
            print(f"{k} : {v}")

    # nerf outputs
    if args.nerf:
        render_id = find_save_id(output_path, prefix="render3d")
        nerf_path = os.path.join(output_path, f"render3d_{str(render_id).zfill(2)}_{'_'.join(prompt.lower().split(' '))}")
        print(f"Results will be saved to {nerf_path}")

        # generate nerf outputs
        print("Rendering begins.")
        os.makedirs(nerf_path, exist_ok=True)
        dataset_render = DatasetRender(load_config=Path(args.load_config), scene_id=scene_id, output_path=Path(nerf_path))
        dataset_render.render_scene()

        layout_files = filter(lambda fname: fname.endswith(".png"), sorted(os.listdir(os.path.join(output_path, "semantic_images"))))
        rgb_files = sorted(os.listdir(os.path.join(nerf_path, "rgb")))
        depth_files = sorted(os.listdir(os.path.join(nerf_path, "depth")))
        with imageio.get_writer(f"{nerf_path.strip('.')}.mp4", fps=10) as video:
            for layout_file, rgb_file, depth_file in tqdm(zip(layout_files, rgb_files, depth_files)):
                layout = Image.open(os.path.join(output_path, "semantic_images", layout_file)).convert("RGB")
                image_width, image_height = layout.size
                layout = layout.crop((0, 0, image_width, image_height // 2))
                rgb = Image.open(os.path.join(nerf_path, "rgb", rgb_file)).convert("RGB")
                depth = Image.open(os.path.join(nerf_path, "depth", depth_file)).convert("RGB")
                layout = layout.resize(rgb.size)
                frame = Image.new("RGB", (rgb.size[0] * 3, rgb.size[1]))
                frame.paste(layout, (0, 0))
                frame.paste(rgb, (rgb.size[0], 0))
                frame.paste(depth, (rgb.size[0] * 2, 0))
                video.append_data(np.array(frame))

    # mesh
    if args.mesh:
        mesh_id = find_save_id(output_path, prefix="mesh")
        mesh_path = os.path.join(output_path, f"mesh_{str(mesh_id).zfill(2)}_{'_'.join(prompt.lower().split(' '))}")
        exporter = ExportTSDFMesh(load_config=Path(args.load_config), output_dir=Path(mesh_path))
        exporter.main()

    # pointcloud
    if args.point:
        point_id = find_save_id(output_path, prefix="point")
        point_path = os.path.join(output_path, f"point_{str(point_id).zfill(2)}_{'_'.join(prompt.lower().split(' '))}")
        exporter = ExportPointCloud(load_config=Path(args.load_config), output_dir=Path(point_path))
        exporter.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # type
    parser.add_argument("--layout", action="store_true", required=False)
    parser.add_argument("--diffusion", action="store_true", required=False)
    parser.add_argument("--nerf", action="store_true", required=False)
    parser.add_argument("--mesh", action="store_true", required=False)
    parser.add_argument("--point", action="store_true", required=False)
    # runtime
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-2-1", required=False)
    parser.add_argument("--dataset", type=str, default=None, required=False)
    parser.add_argument("--output_dir", type=str, default="outputs", required=False)
    parser.add_argument("--scene_id", type=str, default=None, required=False)
    parser.add_argument("--voxel_size", type=float, default=-1., required=False)
    parser.add_argument("--inpaint", action="store_true")
    parser.add_argument("--seed", type=int, default=None, required=False)
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--sequential_input", action="store_true")
    # generation control
    parser.add_argument("--save_real", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="gzzyyxy/layout_diffusion_hypersim_prompt_one_hot_multi_control_bs32_epoch24", required=False)
    parser.add_argument("--checkpoint_subfolder", type=str, default="checkpoint-10900", required=False)
    parser.add_argument("--no_cond", action="store_true")
    parser.add_argument("--no_depth_cond", action="store_true")
    parser.add_argument("--fix_noise", action="store_true")
    parser.add_argument("--prompt", type=str, default="This is one view of a room.", required=False)
    parser.add_argument("--condition_type", type=str, default="one_hot", help="Type of condition image, choose from `rgb`, `embedding`, `clip_embedding`, `one_hot`")
    parser.add_argument("--guidance_scale", type=float, default=7.5, required=False)
    parser.add_argument("--controlnet_conditioning_scale", metavar='N', type=float, nargs='+', default=[3.5, 1.5], required=False)
    parser.add_argument("--control_guidance_start", type=float, default=0.0, required=False)
    parser.add_argument("--control_guidance_end", type=float, default=1.0, required=False)
    parser.add_argument("--lora_layer_path", type=str, default=None, required=False,
                        help="Pretrained lora checkpoint path, should be like 'sd-lora/checkpoint-xxxx'.")
    # nerf
    parser.add_argument("--load_config", type=str, default=None, required=False)
    parser.add_argument("--load_checkpoint", type=str, default=None, required=False)
    args = parser.parse_args()
    main()
