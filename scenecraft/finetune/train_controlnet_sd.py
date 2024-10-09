#!/usr/bin/env python
# coding=utf-8

# Modified from https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py


import argparse
import logging
import math
import os
import random
import shutil
import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from itertools import chain
from pathlib import Path
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAXFormersAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from scenecraft.utils import setup_logger, module_wrapper, load_from_jsonl, opengl_to_opencv
from scenecraft.data import dataset
from scenecraft.data.dataset import CustomSampler
from scenecraft.model import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel, MultiControlNetModel

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = setup_logger(os.path.basename(__file__))

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


VAL_SCENE_IDS = {
    "scannetpp": ['0a7cc12c0e', '0a7cc12c0e', '0a7cc12c0e', '0a7cc12c0e',
                  '0a7cc12c0e', '0a7cc12c0e', '0a7cc12c0e', '0a7cc12c0e',],
    "hypersim": ['ai_010_005', 'ai_010_005', 'ai_010_005', 'ai_010_005',
                 'ai_010_005', 'ai_010_005', 'ai_010_005', 'ai_010_005',],
}
VAL_IDS = {
    "scannetpp": ['DSC05831', 'DSC05836', 'DSC05877', 'DSC05892',
                  'DSC05892', 'DSC05898', 'DSC05903', 'DSC05914',
                  'DSC05892', 'DSC05898', 'DSC05903', 'DSC05914'],
    "hypersim": ['frame.0000', 'frame.0001', 'frame.0002', 'frame.0009',
                 'frame.0018', 'frame.0019', 'frame.0020', 'frame.0021',
                 'frame.0018', 'frame.0019', 'frame.0020', 'frame.0021'],
}


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, indice_to_embedding, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)
    unet = accelerator.unwrap_model(unet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    image_logs = []
    custom_ids = ['000', '025', '050', '075']
    data_info = load_from_jsonl(Path(dataset.FinetuneDataset.BUILDER_PARAMS[args.dataset]["JSONL_PATH"]) / "all.jsonl")
    custom_info = load_from_jsonl(Path(dataset.FinetuneDataset.BUILDER_PARAMS["custom"]["JSONL_PATH"]) / "exp0.jsonl")
    try:
        data_infos = [list(filter(lambda x: x["scene_id"] == VAL_SCENE_IDS[args.dataset][i] and x["id"] == VAL_IDS[args.dataset][i], data_info))[0]
                                                                                                        for i in range(len(VAL_IDS[args.dataset]))]
        custom_infos = [list(filter(lambda x: x["id"] == custom_ids[i], custom_info))[0] for i in range(len(custom_ids))]
        infos = data_infos + custom_infos
    except IndexError as e:
        print(f"{e}! Check whether the image ids exist in the jsonl file. Validation will be skipped!")
        return
    validation_labels = [np.load(info["source"])["labels"].astype(np.int32) for info in infos]
    rgb_condition_images = [Image.open(info["source"].replace(".npz", ".png")) for info in infos]
    w, h = rgb_condition_images[0].size
    rgb_condition_images = [image.crop((0, h // 3, w, h // 3 * 2)) for image in rgb_condition_images]
    validation_metas = [{"extrins": np.load(info["source"])["extrin"].reshape(4, 4),
                         "intrins": np.load(info["source"])["intrin"].reshape(4, 4),
                         "depths": Image.fromarray(np.load(info["source"])["depths"])} for info in infos]
    validation_prompts = ["This is one view of a bedroom in icy winter style."] * 4 + ["This is one view of a bedroom in pink princess style."] * 4 + \
                         ["This is one view of a vampire's bloody bedroom."]
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    keep_aspect_ratio = True
    scale_h = dataset.SCALE_H[args.dataset]
    rgb_condition_images = [
                transforms.Compose([
                    transforms.Resize(scale_h, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(scale_h) if not keep_aspect_ratio else transforms.Lambda(lambda x: x)
                ])(image) for image in rgb_condition_images]

    for i in range(3):

        validation_label = validation_labels[i * 4 : (i + 1) * 4]
        validation_prompt = validation_prompts[i * 4 : (i + 1) * 4]
        validation_meta = validation_metas[i * 4 : (i + 1) * 4]
        rgb_condition_image = rgb_condition_images[i * 4 : (i + 1) * 4]

        validation_meta = {k: [meta[k] for meta in validation_meta] for k in validation_meta[0].keys()}

        if args.condition_type == "rgb":
            if args.from_lllyasviel:
                max_label = max([label_image.max() for label_image in validation_label])
                colors = np.array([dataset.RAW_DATASETS[args.dataset].COLORS[j] for j in range(max_label + 1)])
                colors = np.array([dataset.ada_palette[dataset.nyu40_to_ade20k[i] + 1] for i in range(len(colors))])
                validation_image = [Image.fromarray(colors[label_image.astype(np.int32)].astype(np.uint8)) for label_image in validation_label]
                validation_image = [
                    transforms.Compose([
                        transforms.Resize(scale_h, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(scale_h) if not keep_aspect_ratio else transforms.Lambda(lambda x: x)
                    ])(image) for image in validation_image]
            else:
                validation_image = rgb_condition_image
        else:
            validation_image = [Image.fromarray(label.astype(np.uint8)) for label in validation_label]
            validation_image = [
                transforms.Compose([
                    transforms.Resize(scale_h, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                    transforms.CenterCrop(scale_h) if not keep_aspect_ratio else transforms.Lambda(lambda x: x)
                ])(image) for image in validation_image]
        width, height = validation_image[0].size

        validation_meta["depths"] = [
            transforms.Compose([
                transforms.Resize(scale_h, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                transforms.CenterCrop(scale_h) if not keep_aspect_ratio else transforms.Lambda(lambda x: x)
            ])(depth) for depth in validation_meta["depths"]]
        

        ori_w, ori_h = validation_meta["intrins"][0][0][2] * 2, validation_meta["intrins"][0][1][2] * 2
        scale_factor = scale_h / ori_h
        scale_w = ori_w * scale_factor
        offset_w = (scale_w - scale_h) / 2
        def post_process_K(intrin):
            intrin[:2, :3] *= scale_factor
            if not keep_aspect_ratio:
                intrin[0][2] -= int(offset_w)
            return intrin
        validation_meta["extrins"] = np.array(validation_meta["extrins"])
        validation_meta["intrins"] = np.array([post_process_K(intrin) for intrin in validation_meta["intrins"]])
        if args.dataset == "hypersim": # c2w stored in hypersim is in opengl format
            validation_meta["extrins"] = opengl_to_opencv(validation_meta["extrins"])

        # multiple controlnet inputs: [[rgb1, rgb2, ..., rgbN], [depth1, depth2, ..., depthN]] in PIL.Image type
        if args.enable_depth_cond:
            assert isinstance(controlnet, MultiControlNetModel)
            validation_depth = [Image.fromarray(np.concatenate([np.array(depth)[:, :, None]] * 3, axis=2).astype(np.uint8)) for depth in validation_meta["depths"]] \
                                if not args.single_depth_channels else validation_meta["depths"]
            validation_image = [validation_image, validation_depth]

        # batch generation
        with torch.autocast("cuda"):
            images = pipeline(
                validation_prompt, validation_image,
                indice_to_embedding=indice_to_embedding,
                condition_type=args.condition_type,
                num_inference_steps=20, generator=generator,
                guidance_scale=args.guidance_scale,
                control_guidance_scale=args.control_guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                control_guidance_start=args.control_guidance_start,
                control_guidance_end=args.control_guidance_end,
                **validation_meta,
            ).images

        merged_images = []
        for k in range(len(images)):
            merged_image = Image.new("RGB", (width, height * 2))
            merged_image.paste(rgb_condition_image[k], (0, 0))
            merged_image.paste(images[k], (0, height))
            merged_images.append(merged_image)

        image_logs.append(
            {"validation_image": merged_images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        # tracker.name == "wandb":
        formatted_images = []
        for log in image_logs:
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            for validation_image_, validation_prompt_ in zip(validation_image, validation_prompt):
                formatted_images.append(wandb.Image(validation_image_, caption=validation_prompt_))

        tracker.log({"validation": formatted_images}, step=step)
        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
            ---
            license: creativeml-openrail-m
            base_model: {base_model}
            tags:
            - stable-diffusion
            - stable-diffusion-diffusers
            - text-to-image
            - diffusers
            - controlnet
            inference: true
            ---
            """
    model_card = f"""
            # controlnet-{repo_id}

            These are controlnet weights trained on {base_model} with new type of conditioning.
            {img_str}
            """
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="Which type of dataset.",
    )
    parser.add_argument(
        "--enable_lora_layers",
        action="store_true",
        help="Whether to add lora layers simultaneously."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use for controlnet and others.",
    )
    parser.add_argument(
        "--learning_rate_embedding",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use for controlnet and others.",
    )
    parser.add_argument(
        "--learning_rate_lora",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for lora.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token", type=str, default=None, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ', `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./scenecraft/data/dataset.py",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="A higher guidance scale value encourages the model to generate images closely linked to the text `prompt` at the expense of lower image quality.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        metavar='N',
        type=float,
        nargs='+',
        default=[7.5, 3.5],
        help="The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added to the residual in the original `unet`.",
    )
    parser.add_argument(
        "--control_guidance_start",
        type=float,
        default=0.,
        help="The percentage of total steps at which the ControlNet starts applying.",
    )
    parser.add_argument(
        "--control_guidance_end",
        type=float,
        default=1.,
        help="The percentage of total steps at which the ControlNet stops applying.",
    )
    parser.add_argument(
        "--condition_type",
        type=str,
        default="one_hot",
        help="Type of condition image, choose from `rgb`, `clip_embedding`, `one_hot`, `embedding`"
    )
    parser.add_argument(
        "--conditioning_channels",
        type=int,
        default=8,
        help="Channels of embedding image only when `--condition_type` is `embedding`."
    )
    parser.add_argument(
        "--sequential_input", action="store_true", help="Whether force the inputs of the same scene/batch to be random"
    )
    parser.add_argument(
        "--enable_depth_cond", action="store_true", help="Whether to use depth condition."
    )
    parser.add_argument(
        "--load_pretrained_unet", action="store_true")
    parser.add_argument(
        "--load_pretrained_controlnet", action="store_true")
    parser.add_argument(
        "--from_lllyasviel", action="store_true")
    parser.add_argument(
        "--single_depth_channels", action="store_true")
    parser.add_argument(
        "--control_guidance_scale", type=float, default=1., required=False,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    args.resolution = 512
    logger.info(f"Resolution was set to {args.resolution} automatically of {args.dataset} dataset.")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def make_class_embeddings(dataset_type, type="random", tokenizer=None, text_encoder=None, device=torch.device('cpu')):
    class_embeddings = []

    if type == "clip":
        for c in dataset.RAW_DATASETS[dataset_type].CLASSES:
            token_ids = tokenizer(c, return_tensors="pt").input_ids
            embedding = text_encoder(token_ids.to(device)).last_hidden_state[0][1] # [768]
            class_embeddings.append(embedding)
        class_embeddings = torch.stack(class_embeddings)
    elif type == "random":
        embedding_path = dataset.FinetuneDataset.BUILDER_PARAMS[dataset_type]["COND_EMBED_PATH"]
        if not os.path.exists(embedding_path):
            for c in dataset.RAW_DATASETS[dataset_type].CLASSES:
                class_embeddings.append(torch.rand(8) * 2 - 1)
            np.save(embedding_path, torch.stack(class_embeddings).numpy())
        class_embeddings = torch.tensor(np.load(embedding_path), device=device)

    return class_embeddings


def make_train_dataset(args, tokenizer, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Here the dataset is defined in `scenecraft.dataset:FinetuneDataset`.
    raw_dataset = load_dataset(args.train_data_dir, args.dataset, cache_dir=args.cache_dir, split="train")
    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = raw_dataset.column_names
    logger.info(f"Loaded column names: {column_names}")

    # 6. Get the column names for input/target.
    # column_names = "scene_id", "voxel", "id", "target", "source", "source_image", "item_id", "embedding", "prompt"
    is_online = "online" in args.dataset
    if not is_online:
        label_column, cond_depth_column, extrin_column, intrin_column = None, None, None, None
        _, _, image_column, meta_data_column, conditioning_image_column, _, _, prompt_column = column_names
    else:
        _, _, image_column, meta_data_column, conditioning_image_column, label_column, cond_depth_column, extrin_column, intrin_column, _, _, prompt_column = column_names

    def tokenize_prompts(examples, is_train=True):
        prompts = []
        for prompt in examples[prompt_column]:
            if random.random() < args.proportion_empty_prompts:
                prompts.append("")
            elif isinstance(prompt, str):
                prompts.append(prompt)
            elif isinstance(prompt, (list, np.ndarray)):
                # take a random caption if there are multiple
                prompts.append(random.choice(prompt) if is_train else prompts[0])
            else:
                raise ValueError(
                    f"Prompt column `{prompt_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(), # values in range [0, 1], channels in CHW order
            transforms.Normalize([0.5], [0.5]), # values in range [-1, 1]
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST_EXACT)
            if args.condition_type != "rgb" else
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor() if args.condition_type == "rgb" else
            transforms.Lambda(lambda x: torch.tensor(np.asarray(x), dtype=torch.long)),
        ]
    )

    depth_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
            transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: torch.cat([torch.tensor(np.asarray(x), dtype=torch.float32)[None]] *
                                                  (3 if not args.single_depth_channels else 1))), # [3/1, h, w]
        ]
    )

    default_transforms = lambda x: torch.tensor(np.array(x), dtype=torch.float32)

    def preprocess_train(examples):
        if examples is None:
            return None

        # load rgb images and condition images
        images = [image.convert("RGB") for image in examples[image_column]]

        if not is_online:
            conditioning_labels = [np.load(data)["labels"].astype(np.int32) for data in examples[meta_data_column]]
        else:
            conditioning_labels = [np.array(labels) for labels in examples[label_column]]
        if args.condition_type == "rgb":
            max_label = max([label_image.max() for label_image in conditioning_labels])
            colors = np.array([dataset.RAW_DATASETS[args.dataset].COLORS[j] for j in range(max_label + 1)])
            if args.from_lllyasviel:
                colors = np.array([dataset.ada_palette[dataset.nyu40_to_ade20k[i] + 1] for i in range(len(colors))])
            conditioning_images = [Image.fromarray(colors[label_image.astype(np.int32)].astype(np.uint8)) for label_image in conditioning_labels]
        else:
            conditioning_images = [Image.fromarray(label_image.astype(np.uint8)) for label_image in conditioning_labels]

        # transform rgb images and condition images
        images = [image_transforms(image) for image in images]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        # depth
        if args.enable_depth_cond:
            if not is_online:
                depths = [Image.fromarray(np.load(data)["depths"]) for data in examples[meta_data_column]]
            else:
                depths = [Image.fromarray(np.array(cond_depth)) for cond_depth in examples[cond_depth_column]]
            depths = [torch.nan_to_num(depth_transforms(depth), 0) for depth in depths]
            examples["depths"] = depths

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_prompts(examples)

        return examples

    with accelerator.main_process_first():
        raw_dataset = raw_dataset.with_transform(preprocess_train)
        train_dataset = dataset.Dataset(raw_dataset, logger)

    return train_dataset


def collate_fn(examples):
    if examples is None:
        return None

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    depths_dict = {}
    if "depths" in examples[0]:
        depths = torch.stack([example["depths"] for example in examples])
        depths_dict["depths"] = depths

    cameras_dict = {}
    if "extrins" in examples[0] and "intrins" in examples[0]:
        extrins = torch.stack([example["extrins"] for example in examples])
        intrins = torch.stack([example["intrins"] for example in examples])
        cameras_dict["extrins"] = extrins
        cameras_dict["intrins"] = intrins

    return {
        "scene_id": [example["scene_id"] for example in examples],
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        **depths_dict,
        **cameras_dict,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    logger.info("DDPM noise scheduler loaded")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    logger.info("Text encoder loaded")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    logger.info("AutoencoderKL loaded")

    # load unet model
    if not args.load_pretrained_unet:
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision,
        )
        logger.info(f"Unet2DConditionModel loaded from {args.pretrained_model_name_or_path}.")
    else:
        checkpoint_path = ...
        checkpoint_subfolder = ...
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            checkpoint_path, subfolder=f"{checkpoint_subfolder}/unet2dconditionmodel", revision=args.revision
        )
        logger.info(f"Unet2DConditionModel loaded from {checkpoint_path}.")

    # load controlnet model
    if not args.load_pretrained_controlnet:
        logger.info("Initializing controlnet weights from unet")
        if not args.from_lllyasviel:
            controlnet: ControlNetModel = ControlNetModel.from_unet(
                unet, make_embedding_reduce_layer=(args.condition_type=="clip_embedding"),
                make_one_hot_embedding_layer=(args.condition_type=="one_hot"),
                num_embeddings=len(dataset.RAW_DATASETS[args.dataset].CLASSES),
                embedding_dim=text_encoder.config.hidden_size,
                conditioning_channels=args.conditioning_channels)
        else:
            assert args.condition_type == "rgb" and args.conditioning_channels == 3, "Wrong condition type or conditioning channels"
            assert "v1-5" in args.pretrained_model_name_or_path, "Should use sd v1.5"
            controlnet: ControlNetModel = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg")
        embedding_reduce_layer_prompt = "" if controlnet.embedding_reduce is None else "with clip embedding layer"
        logger.info(f"Controlnet model loaded {embedding_reduce_layer_prompt}")
        if args.enable_depth_cond:
            if not args.from_lllyasviel:
                controlnet2: ControlNetModel = ControlNetModel.from_unet(unet, conditioning_channels=3 if not args.single_depth_channels else 1)
            else:
                controlnet2: ControlNetModel = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth")
            logger.info(f"Controlnet2 model loaded.")
            controlnet: MultiControlNetModel = MultiControlNetModel([controlnet, controlnet2])
    else:
        checkpoint_path = {
            "scannetpp": "gzzyyxy/layout_diffusion_scannetpp_blipv2_one_hot_multi_control_bs16_epoch24",
            "hypersim": "gzzyyxy/layout_diffusion_hypersim_blipv2_one_hot_multi_control_bs16_epoch24",
            }
        checkpoint_subfolder = {
            "scannetpp": "checkpoint-8250",
            "hypersim": "checkpoint-8400",
            }
        if "multi_control" in checkpoint_path[args.dataset]:
            if not args.enable_depth_cond:
                args.enable_depth_cond = True
                logger.warning(f"Load pretrained controlnet model {checkpoint_path[args.dataset]} but depth condition is not enabled, it will be enabled automatically.")
            controlnet: MultiControlNetModel = MultiControlNetModel.from_pretrained(
                checkpoint_path[args.dataset], subfolder=f"{checkpoint_subfolder[args.dataset].strip('/')}")
        else:
            controlnet: ControlNetModel = ControlNetModel.from_pretrained(
                checkpoint_path[args.dataset], subfolder=f"{checkpoint_subfolder[args.dataset].strip('/')}/controlnetmodel",
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    # TODO: More elegant way.
    if version.parse(accelerate.__version__) >= version.parse("0.16.0") and not args.enable_lora_layers:
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    # could be "controlnet" or "mappingnetwork"
                    sub_dir = model.__class__.__name__.lower()
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    logger.info(f"Saved model {model.__class__.__name__} to {os.path.join(output_dir, sub_dir)}")

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnetmodel")
                elif isinstance(model, MultiControlNetModel):
                    load_model = MultiControlNetModel.from_pretrained(input_dir, subfolder=".")
                elif isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet2dconditionmodel")
                else:
                    logger.warning(f"Not found pretrained model {model.__class__.__name__} and will be skipped.")
                    continue
                logger.info(f"Loaded pretrained model {model.__class__.__name__} from {os.path.join(input_dir, model.__class__.__name__.lower())}")

                if isinstance(model, MultiControlNetModel):
                    model.nets[0].register_to_config(**load_model.nets[0].config)
                    model.nets[1].register_to_config(**load_model.nets[1].config)
                else:
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # LoRA Layers
    if args.enable_lora_layers:
        # now we will add new LoRA weights to the attention layers
        # It's important to realize here how many attention weights will be added and of which sizes
        # The sizes of the attention layers consist only of two different variables:
        # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
        # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

        # Let's first see how many attention processors we will have to set.
        # For Stable Diffusion, it should be equal to:
        # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
        # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
        # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
        # => 32 layers

        # choose attn processor
        AttnProcessor = LoRAXFormersAttnProcessor if args.enable_xformers_memory_efficient_attention \
                                                                                else LoRAAttnProcessor
        # Set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = AttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=args.rank,
            )

        # This operation will add new layers into unet model
        unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            if isinstance(controlnet, MultiControlNetModel):
                for subcontrolnet in controlnet.nets:
                    subcontrolnet.enable_xformers_memory_efficient_attention()
            else:
                controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors) if args.enable_lora_layers else None
    if lora_layers is not None:
        logger.info("LoRA layers loaded")

    if args.gradient_checkpointing:
            if isinstance(controlnet, MultiControlNetModel):
                for subcontrolnet in controlnet.nets:
                    subcontrolnet.enable_gradient_checkpointing()
            else:
                controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    controlnet_dtype = accelerator.unwrap_model(controlnet).dtype if not isinstance(controlnet, MultiControlNetModel) else \
                        accelerator.unwrap_model(controlnet).nets[0].dtype
    if controlnet_dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {controlnet_dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = []
    params_to_optimize.append(
        {'params': controlnet.parameters()})
    if lora_layers is not None:
        params_to_optimize.append(
            {'params': lora_layers.parameters(), 'lr': args.learning_rate_lora})
    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # shape [num_classes, embed_dim=768]
    class_embeddings = make_class_embeddings(args.dataset, "random" if args.condition_type == "embedding" else "clip",
                                             tokenizer, text_encoder, accelerator.device) \
                            if args.condition_type == "embedding" or args.condition_type == "clip_embedding" else None

    train_dataset = make_train_dataset(args, tokenizer, accelerator)

    batch_size = args.train_batch_size
    train_sampler = CustomSampler(train_dataset, batch_size, sequential_input=args.sequential_input)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    controlnet = accelerator.prepare(controlnet)
    if lora_layers is not None:
        lora_layers = accelerator.prepare(lora_layers)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Control parameters *****")
    logger.info(f"  Guidance scale = {args.guidance_scale}")
    logger.info(f"  Controlnet conditioning scale = {args.controlnet_conditioning_scale}")
    logger.info(f"  Control guidance start = {args.control_guidance_start}")
    logger.info(f"  Control guidance end = {args.control_guidance_end}")
    logger.info(f"  Condition type = {args.condition_type}")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.
        for step, batch in enumerate(train_dataloader):
            if batch is None:
                logger.error(f"Step {step} skipped.")
                continue

            assert isinstance(batch, dict), f"Unexpected type of batch data: {type(batch)}."
            assert len(set(batch["scene_id"])) == 1, f"Batch data should be from the same scene, instead of {len(set(batch['scene_id']))}."
            with accelerator.accumulate(controlnet):

                # Convert images to latent space
                latents: torch.FloatTensor = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"]).last_hidden_state # [B, 77, 768]

                # Prepare multiple conditions
                if isinstance(module_wrapper(controlnet), MultiControlNetModel):
                    controlnet_image = [
                        batch["conditioning_pixel_values"].to(dtype=weight_dtype),
                        batch["depths"].to(dtype=weight_dtype) / 20.
                    ]
                    controlnet_image[1][torch.isinf(controlnet_image[1])] = 0.
                else:
                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # Calculate residuals for unet
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    indice_to_embedding=class_embeddings,
                    # guidance_start=0., guidance_end=1.
                    conditioning_scale=args.controlnet_conditioning_scale,
                    return_dict=False,
                )

                # Prepare extra kwargs
                extra_kwargs = dict()

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    **extra_kwargs
                ).sample

                # control classifier free guidance TODO: to be optimized
                if args.control_guidance_scale > 1.:
                    model_pred_uncond = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        **extra_kwargs
                    ).sample

                    model_pred = model_pred_uncond + args.control_guidance_scale * (model_pred - model_pred_uncond)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                extra_loss = dict()
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    if lora_layers is not None:
                        params_to_clip = chain(params_to_clip, lora_layers.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae, text_encoder, tokenizer, unet, controlnet, class_embeddings, args, accelerator, weight_dtype, global_step)

            logs = {"loss": loss.detach().item(), "train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0], **extra_loss}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            train_loss = 0.

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)
        if lora_layers is not None:
            unet = unet.to(torch.float32)
            unet.save_attn_procs(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)