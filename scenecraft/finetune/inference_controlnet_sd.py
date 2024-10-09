import argparse
import functools
import os

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from diffusers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import DataLoader

from scenecraft.utils import setup_logger
from scenecraft.data.dataset import Dataset
from scenecraft.guidance import StableDiffusionGuidance
from scenecraft.model import StableDiffusionControlNetPipeline, ControlNetModel, MultiControlNetModel, MappingNetwork
from scenecraft.prompt_processor import StableDiffusionPromptProcessor
from scenecraft.finetune.train_controlnet_sd import make_class_embeddings


def save_outputs(output_dir, image: Image.Image, control_image: Image.Image, scene_id: str, id: str, i: int):
    save_path = os.path.join(output_dir, f"{scene_id}_{str(i)}")
    os.makedirs(save_path, exist_ok=True)

    width, height = image.size
    merged_image = Image.new('RGB', (width, height * 2))
    merged_image.paste(control_image, (0, 0))
    merged_image.paste(image, (0, height))
    merged_image.save(f'{save_path}/{id}.png')


def collate_fn(examples):
    scene_ids = [example["scene_id"] for example in examples]
    ids = [example["id"] for example in examples]
    images = [example["images"] for example in examples]
    prompts = [example["prompts"] for example in examples]
    embeddings = [example["embeddings"] for example in examples]

    depth_dict = {}
    if "depths" in examples[0]:
        depths = [example["depths"] for example in examples]
        depth_dict["depths"] = depths

    return {
        "scene_ids": scene_ids,
        "ids": ids,
        "images": images,
        "prompts": prompts,
        "embeddings": embeddings,
        **depth_dict,
    }


def main(args):
    seed = args.seed or 0
    save_outputs_partial = functools.partial(save_outputs, args.output_dir)
    logger = setup_logger("inference_controlnet")
    # dataset
    dataset = load_dataset(args.test_data_dir, args.dataset, split=args.split)
    dataset = Dataset(dataset, logger)
    if args.scene_id is not None:
        scene_ids = dataset.get_unique_scene_ids()
        if args.scene_id not in scene_ids:
            raise ValueError(f"Input scene {args.scene_id} was not found in dataset.")
        dataset = dataset.filter(lambda x: x["scene_id"] == args.scene_id)

    column_names = dataset.column_names
    logger.info(f"Load {column_names}")

    meta_data_column = "source"
    conditioning_image_column = "source_image"
    prompt_column = "prompt"
    embedding_column = "embedding"

    def preprocess_val(examples):
        if args.condition_type == "rgb":
            conditioning_images = [image.convert("RGB").crop((0, image.size[1] // 3, image.size[0], image.size[1] // 3 * 2))
                                                                for image in examples[conditioning_image_column]]
        else:
            conditioning_images = [Image.fromarray(np.load(image)["labels"].astype(np.uint8))
                                            for image in examples[meta_data_column]]
        examples["images"] = conditioning_images
        examples["prompts"] = examples[prompt_column]
        # Avoid error when embedding column was removed temporarily
        if embedding_column in examples:
            examples["embeddings"] = examples[embedding_column]
        if args.enable_depth_cond or "multi_control" in args.checkpoint_path:
            conditioning_images2 = [Image.fromarray(np.load(data)["depths"]) for data in examples[meta_data_column]]
            examples["depths"] = conditioning_images2

        return examples

    val_dataset = dataset.with_transform(preprocess_val)
    batch_size = args.batch_size

    # model
    if "multi_control" not in args.checkpoint_path:
        controlnet = ControlNetModel.from_pretrained(
            args.checkpoint_path, subfolder=f"{args.checkpoint_subfolder.strip('/')}/controlnetmodel", torch_dtype=torch.float16)
    else:
        controlnet: MultiControlNetModel = MultiControlNetModel.from_pretrained(
            args.checkpoint_path, subfolder=f"{args.checkpoint_subfolder.strip('/')}", torch_dtype=torch.float16
        )
        assert isinstance(args.controlnet_conditioning_scale, (list, tuple))
        if batch_size != 1:
            batch_size = 1
            logger.warning("A single batch of multiple conditioning inputs are supported at the moment, thus force `batch_size` to be 1.")
    mapping_network = MappingNetwork.from_pretrained(
        args.checkpoint_path, subfolder=f"{args.checkpoint_subfolder.strip('/')}/mappingnetwork", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model_path, controlnet=controlnet, mapping_network=mapping_network, torch_dtype=torch.float16
    )
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # load lora if available
    if args.lora_layer_path:
        # pipe.load_lora_weights(args.lora_layer_path)
        pipe.unet.load_attn_procs(args.lora_layer_path)

    pipe.to(torch_device=torch.device('cuda'))
    # remove following line if xformers is not installed or when using Torch 2.0.
    pipe.enable_xformers_memory_efficient_attention()

    # check the guidance pipeline in debug mode
    guidance = None
    if os.environ.get("DEBUG", False):
        assert args.batch_size == 1, f"Use `batch_size` = 1 in debug mode."
        guidance = StableDiffusionGuidance(
            pipe.device, args.dataset,
            guidance_use_full_precision=False,
            checkpoint_path=args.checkpoint_path,
            checkpoint_subfolder=args.checkpoint_subfolder,
            condition_type=args.condition_type,
            pretrained_model_name_or_path=args.base_model_path)
        prompt_processor = StableDiffusionPromptProcessor(
            pretrained_model_name_or_path=args.base_model_path,
            tokenizer=guidance.pipe.tokenizer,
            text_encoder=guidance.pipe.text_encoder.to(torch.float32),
            prompts=dataset.get_column_data(column_name="prompt"),
            do_classifier_free_guidance=(args.guidance_scale > 1.),
            spawn=False,
        )

    # prepare `indice_to_embedding`
    indice_to_embedding = make_class_embeddings(args.dataset, pipe.tokenizer, pipe.text_encoder,
                            torch.device('cuda')).to(torch.float16) if args.condition_type == "embedding" else None

    # generate image
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    repeat_times = args.repeat_times if args.scene_id is not None else 1
    for i in range(repeat_times):
        # reset scene embedding
        if args.scene_id is not None and args.random_embedding and i > 0:
            np.random.seed(seed)
            new_embeddings = np.random.randn(val_dataset.num_samples_per_scene, 77, 768).tolist()
            val_dataset = val_dataset.remove_columns("embedding")
            val_dataset = val_dataset.add_column(name="embedding", column=new_embeddings)
            val_dataset = Dataset(val_dataset, logger)

        val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)
        # TODO: progress bar
        for step, batch in enumerate(val_dataloader):
            scene_ids, ids, control_images, prompts, embeddings, *_ = batch.values()
            if "depths" in batch.keys():
                # A single batch of multiple conditioning inputs are supported at the moment
                control_images = [control_images[0], batch["depths"][0]]
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)

            latents = None
            if os.environ.get("DEBUG", False):
                batch_size = len(control_images)
                width, height = control_images[0].size
                shape = (batch_size, 4, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
                latents = randn_tensor(shape, generator, device=pipe.device).to(pipe.text_encoder.dtype)
                text_embeddings = prompt_processor(prompts)
                guidance_outputs = guidance.denoise(
                    control_images, embeddings,
                    text_embeddings=text_embeddings.to(pipe.text_encoder.dtype),
                    T = pipe.scheduler.timesteps.shape[0], latents_noisy=latents,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    control_guidance_start=args.control_guidance_start,
                    control_guidance_end=args.control_guidance_end,
                    return_denoised_image=True,
                )
                guidance_images = guidance_outputs.pop("denoised_images_pil")

            images = pipe(
                prompts, control_images, embeddings,
                indice_to_embedding=indice_to_embedding,
                condition_type=args.condition_type,
                num_inference_steps=20, latents=latents,
                generator=generator, guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                control_guidance_start=args.control_guidance_start,
                control_guidance_end=args.control_guidance_end,
            ).images

            if not os.environ.get("DEBUG", False):
                for image, control_image, scene_id, id in zip(images, control_images, scene_ids, ids):
                    save_outputs_partial(image, control_image, scene_id, id, i)
            else:
                os.makedirs("test_guidance", exist_ok=True)
                for j, image, guidance_image, control_image in zip(range(len(images)), images, guidance_images, control_images):
                    width, height = image.size
                    new_image = Image.new("RGB", (width, height * 3))
                    new_image.paste(image, (0, 0))
                    new_image.paste(guidance_image, (0, height))
                    new_image.paste(control_image, (0, height * 2))
                    new_image.save(f"./test_guidance/{step}-{j}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # runtime
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, default="./proj/data/dataset.py", required=False)
    parser.add_argument("--output_dir", type=str, default="./outputs/scannetpp", required=False)
    parser.add_argument("--split", type=str, default="test", required=False)
    parser.add_argument("--batch_size", type=int, default=24, required=False)
    parser.add_argument("--scene_id", type=str, default=None, required=False)
    parser.add_argument("--random_embedding", action="store_true")
    parser.add_argument("--repeat_times", type=int, default=4, required=False,
                        help="Repeat times for generation of specified scene, only used when scene_id is set.")
    parser.add_argument("--seed", type=int, default=None, required=False)
    # generation control
    parser.add_argument("--checkpoint_path", type=str, default="gzzyyxy/layout_diffusion_scannetpp_all_labels_multi_control_bs48", required=False)
    parser.add_argument("--checkpoint_subfolder", type=str, default="checkpoint-17150", required=False)
    parser.add_argument("--condition_type", type=str, default="embedding", help="Type of condition image, choose from `rgb`, `embedding`, `one_hot`")
    parser.add_argument("--guidance_scale", type=float, default=7.5, required=False)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=3.5, required=False)
    parser.add_argument("--control_guidance_start", type=float, default=0.0, required=False)
    parser.add_argument("--control_guidance_end", type=float, default=1.0, required=False)
    parser.add_argument("--enable_depth_cond", action="store_true", required=False)
    # lora layers
    parser.add_argument("--lora_layer_path", type=str, default=None, required=False,
                        help="Pretrained lora checkpoint path, should be like 'sd-lora/checkpoint-xxxx'.")
    args = parser.parse_args()
    main(args)
