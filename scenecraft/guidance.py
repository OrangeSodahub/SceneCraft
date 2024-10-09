import PIL
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List
from torch import nn
from jaxtyping import Float
from torch import Tensor

from diffusers import UniPCMultistepScheduler, DDIMScheduler, DDPMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from nerfstudio.utils.rich_utils import CONSOLE

from scenecraft.utils import cleanup
from scenecraft.model import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel, MultiControlNetModel
from scenecraft.prompt_processor import PromptProcessorOutput
from scenecraft.finetune.train_controlnet_sd import make_class_embeddings


@dataclass
class StableDiffusionGuidanceOutput(BaseOutput):
    """
    loss_grad: (`torch.FloatTensor`)
    images (`List[PIL.Image.Image]` or `np.ndarray`)
        List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
        num_channels)`.
    """
    # time schedule settings
    T: int
    timesteps: list

    # denoise outputs
    images_pil: List[PIL.Image.Image]
    images_pt: torch.FloatTensor
    depths_pt: torch.FloatTensor


class StableDiffusionGuidance(nn.Module):
    """Guidance implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self,
                 device: Union[torch.device, str],
                 dataset: str,
                 num_train_timesteps: int = 1000,
                 num_inference_steps: int = 50,
                 guidance_use_full_precision: bool = False,
                 checkpoint_path: Union[str, Path] = None,
                 checkpoint_subfolder: str="",
                 unet_checkpoint_path: Union[str, Path] = None,
                 unet_checkpoint_subfolder: str="",
                 pretrained_model_name_or_path: Union[str, Path] = None,
                 condition_type: str = "rgb",
                 rgb_as_latents: bool = False,
                 guidance_scale: float = 7.5, 
                 guidance_depth: bool = False,
                 controlnet_conditioning_scale: float = 3.5,
                 control_guidance_start: Union[float, List[float]] = 0.,
                 control_guidance_end: Union[float, List[float]] = 1.,
                 scheduler_type: str = "ddim",
                 fix_init_noise: bool = False,
                 loss_type: str = "rgb_loss") -> None:
        super().__init__()

        self.device = device
        self.dataset = dataset
        self.min_timestep = -1
        self.max_timestep = -1
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.condition_type = condition_type
        self.rgb_as_latents = rgb_as_latents
        self.guidance_scale = guidance_scale
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.control_guidance_start = control_guidance_start
        self.control_guidance_end = control_guidance_end
        self.guidance_use_full_precision = guidance_use_full_precision
        self.fix_init_noise = fix_init_noise
        self.loss_type = loss_type

        # model, no lora layers.
        if True:
            unet_checkpoint_path = pretrained_model_name_or_path
            unet_checkpoint_subfolder = "unet"
        else:
            unet_checkpoint_subfolder = f"{unet_checkpoint_subfolder.strip('/')}/unet2dconditionmodel"
        if "multi_control" not in checkpoint_path:
            controlnet: ControlNetModel = ControlNetModel.from_pretrained(
                checkpoint_path, subfolder=f"{checkpoint_subfolder.strip('/')}/controlnetmodel", torch_dtype=torch.float16)
        else:
            controlnet: MultiControlNetModel = MultiControlNetModel.from_pretrained(
                checkpoint_path, subfolder=f"{checkpoint_subfolder.strip('/')}", torch_dtype=torch.float16)
            assert isinstance(self.controlnet_conditioning_scale, (list, tuple))
        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            unet_checkpoint_path, subfolder=unet_checkpoint_subfolder, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path, unet=unet, controlnet=controlnet, safety_checker=None)
        if scheduler_type == "ddpm":
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        elif scheduler_type == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif scheduler_type == "unipc":
            # speed up diffusion process with faster scheduler and memory optimization
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)

        # prepare indice to embedding
        self.indice_to_embedding = make_class_embeddings(dataset,
                                                         "random" if self.condition_type == "embedding" else "clip",
                                                         pipe.tokenizer, pipe.text_encoder,
                                                         self.device) if self.condition_type == "embedding" or self.condition_type == "clip_embedding" else None

        # enable memory efficient attention by default
        if not is_xformers_available():
            CONSOLE.print("xformers is not available, memory efficient attention is not enabled.")
        else:
            pipe.enable_xformers_memory_efficient_attention()

        # improve memory performance
        pipe.enable_attention_slicing()
        cleanup()

        # use for improved quality at cost of higher memory
        if self.guidance_use_full_precision:
            self.weights_dtype = torch.float32
        else:
            self.weights_dtype = torch.float16

        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.controlnet.eval()
        pipe.unet.to(dtype=self.weights_dtype)
        pipe.vae.to(dtype=self.weights_dtype)
        pipe.text_encoder.to(dtype=self.weights_dtype)
        pipe.controlnet.to(dtype=self.weights_dtype)
        if self.indice_to_embedding is not None:
            self.indice_to_embedding = self.indice_to_embedding.to(dtype=self.weights_dtype)

        self.pipe = pipe
        # cast members to `self` object in order to move them to DDP
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.controlnet = pipe.controlnet

        self.weighting_strategy = "uniform"
        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        self.alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(self.device)

        # depth estimator
        self.depth_estimator = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to(self.device) \
                                if guidance_depth else None

        CONSOLE.print("StableDiffusionGuidance loaded!")

    def prepare_latents(self, b, h, w, generator=None, dtype=torch.float32):
        shape = (b, 4, h // self.vae_scale_factor, w // self.vae_scale_factor)
        latents = randn_tensor(shape, generator, device=self.device, dtype=dtype)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

    def forward_unet(self,
                     latent_model_input: Float[Tensor, "B 4 H W"],
                     t: Float[Tensor, "B"],
                     encoder_hidden_states: Float[Tensor, ""],
                     down_block_res_samples: List[Float[Tensor, "B C fH fW"]],
                     mid_block_res_sample: Float[Tensor, ""],
                     ) -> Float[Tensor, ""]:
        input_dtype = latent_model_input.dtype
        unet_output: UNet2DConditionOutput = self.pipe.unet(
            latent_model_input.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )
        return unet_output.sample.to(input_dtype)

    def encode_images(self, imgs: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B 4 H W"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0 # range [0, 1] -> [-1, 1]
        posterior = self.pipe.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor
        return latents.to(input_dtype)

    def decode_latents(self, latents: Float[Tensor, "B 4 H W"], latent_height: int=64, latent_width: int=64
                       ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.to(self.weights_dtype)).sample
        return image.to(input_dtype)

    def denoise_loop(self,
                     latents: torch.Tensor,
                     cond_image: Union[torch.Tensor, List[torch.Tensor]],
                     text_embeddings: torch.Tensor,
                     do_classifier_free_guidance: bool,
                     cross_attention_kwargs: dict,
                     **kwargs):
        # time schedule
        timesteps = self.pipe.scheduler.timesteps

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            # only infer conditional batch if guress mode
            control_model_input = latent_model_input.to(self.pipe.controlnet.dtype)
            controlnet_text_embeddings = text_embeddings.to(self.pipe.controlnet.dtype)

            cond_scale = self.controlnet_conditioning_scale
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                sample=control_model_input, timestep=t,
                encoder_hidden_states=controlnet_text_embeddings,
                controlnet_cond=cond_image, conditioning_scale=cond_scale,
                return_dict=False,
            )

            # predict the noise residual
            latent_model_input = latent_model_input.to(self.pipe.unet.dtype)
            text_embeddings = text_embeddings.to(self.pipe.unet.dtype)
            noise_pred = self.pipe.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
                **kwargs,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            step_outputs = self.pipe.scheduler.step(noise_pred, t, latents)
            latents = step_outputs.prev_sample

            # denoise single step: x_t -> x_0
            if self.num_inference_steps == 1:
                # assert isinstance(self.pipe.scheduler, DDIMScheduler), "Wrong scheduler type."
                alpha_prod_t = self.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                latents = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                break

        return latents

    def denoise(self,
                cond_image: Union[List[PIL.Image.Image], Float[Tensor, "B 3 H W"], Float[Tensor, "B H W"]],
                text_embeddings: Union[Float[Tensor, "N max_len embed_dim"], PromptProcessorOutput],
                image: Float[Tensor, "B 3 H W"] = None,
                latents_noisy: Float[Tensor, "B 4 H W"] = None,
                T: int = 1000,
                generator: torch.Generator = None,
                return_denoised_image: bool = False, **kwargs) -> dict:

        """
            Refer to `scenecraft.model.StableDiffusionControlNetPipeline.__call__`

        Args:
            cond_image: (`List[PIL.Image.Image]` or `torch.FloatTensor`) Condition image.
            text_embeddings: (`torch.FloatTensor`) Text embeddings encoded by `CLIPTextModel`.
            image: (`torch.FloatTensor`) Rendered image to edit, conflict with `latents`.
            T: (`int`) Noise level, only set in debug mode.
            latents: (`torch.FloatTensor`) Samples to be denoised, only set in debug mode.
            generator: (`torch.Generator`) To generate random noise.
            rgb_as_latents: (`bool`) If rendering outputs are RGB or latents.
            guidance_scale: (`float`) Text-guidance scale
            diffusion_steps: (`int`) Number of diffusion inference steps.
            min_timestep_percent: (`float`) Lower bound for diffusion timesteps to use for image editing.
            max_timestep_percent: (`float`) Upper bound for diffusion timesteps to use for image editing.
            return_denoised_image: (`bool`) If decode latents to RGB image and return.
        """

        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        output_type = kwargs.pop("output_type", "pil")
        cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", None)
        cond_depth = kwargs.pop("cond_depth", None)
        extrins = kwargs.pop("extrins", None)
        intrins = kwargs.pop("intrins", None)
        depths = kwargs.pop("depths", None)

        if isinstance(cond_image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(cond_image, list):
            batch_size = len(cond_image)
        elif isinstance(cond_image, Tensor):
            batch_size = cond_image.shape[0]
        else:
            raise TypeError(f"Unsupported type of input condition image {type(cond_image)}.")
        device = self.pipe._execution_device

        # time schedule
        self.pipe.scheduler.config.num_train_timesteps = int(T if T > 0 else self.num_train_timesteps)
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.pipe._execution_device)

        # prepare latents
        # in case of image is None (no rendered results for current selected
        # idx) and T <= 0 (in the stage of early distilling), use pure noise.
        if image is not None and T > 0:
            latents = self.encode_images(image) if not self.rgb_as_latents else image
            # add noise: x_t = √α_bar_t * x_0 + √1 - α_bar_t * N(0, I)
            noise = torch.randn_like(latents)
            latents_noisy = self.pipe.scheduler.add_noise(latents, noise, self.pipe.scheduler.timesteps[0].view(1))  # type: ignore
        else:
            if self.fix_init_noise:
                assert latents_noisy is not None, "Set fix init noise but not found input noise."
            else:
                # refer to `StableDiffusionControlNetPipeline.prepare_latents()`
                assert isinstance(cond_image, Tensor), f"Expected `torch.Tensor`, got `{type(cond_image)}`."
                height, width = cond_image.shape[-2:]
                latents_noisy = self.prepare_latents(batch_size, height, width, generator, dtype=cond_image.dtype)
        latents_noisy = latents_noisy.to(self.device)

        do_classifier_free_guidance = self.guidance_scale > 1.0
        text_embeddings = text_embeddings.to(device=self.device, dtype=self.weights_dtype)
        latents = latents_noisy.clone()

        with torch.no_grad():
            # prepare text embeddings
            if isinstance(text_embeddings, torch.Tensor) and do_classifier_free_guidance:
                text_embeddings, negative_text_embeddings = self.pipe.encode_prompt(
                    None, device, do_classifier_free_guidance=do_classifier_free_guidance,
                    prompt_embeds=text_embeddings, num_images_per_prompt=1,
                )
                text_embeddings = torch.cat([negative_text_embeddings, text_embeddings])
            elif isinstance(text_embeddings, PromptProcessorOutput):
                text_embeddings = text_embeddings.get_text_embeddings()
            else:
                raise TypeError(f"Unknown type of `text_embeddings` {type(text_embeddings)}, should "
                                f"be one of [`torch.Tensor`, `PromptProcessorOutput`].")

            # prepare condition image
            cond_image = self.pipe.prepare_image(
                image=cond_image, width=width, height=height, batch_size=batch_size,
                num_images_per_prompt=1, device=device, dtype=self.pipe.controlnet.dtype,
                image_type=("indice" if self.condition_type != "rgb" else self.condition_type),
                indice_to_embedding=self.indice_to_embedding,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            height, width = cond_image.shape[-2:]
            if cond_depth is not None:
                assert isinstance(self.pipe.controlnet, MultiControlNetModel)
                cond_depth = self.pipe.prepare_image(
                    image=cond_depth, width=width, height=height, batch_size=batch_size,
                    num_images_per_prompt=1, device=device, dtype=self.pipe.controlnet.dtype,
                    image_type="depth", do_classifier_free_guidance=do_classifier_free_guidance,
                )
                cond_depth = cond_depth / 20. # [B, 1, H, W]
                cond_image = [cond_image, cond_depth]

            # denoise!
            latents = self.denoise_loop(latents,
                                        cond_image,
                                        text_embeddings,
                                        do_classifier_free_guidance,
                                        cross_attention_kwargs,
                                )

            # If we do sequential model offloading, let's offload unet and controlnet
            # manually for max memory savings
            if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
                self.pipe.unet.to("cpu")
                self.pipe.controlnet.to("cpu")
                torch.cuda.empty_cache()

            denoised_images = None
            if not output_type == "latent" or return_denoised_image:
                denoised_images = self.pipe.vae.decode(latents.to(self.pipe.vae.dtype) / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
                denoised_images_pt = self.pipe.image_processor.postprocess(denoised_images, output_type="pt")
                denoised_images_pil = VaeImageProcessor.numpy_to_pil(VaeImageProcessor.pt_to_numpy(denoised_images_pt))

            # Offload all models
            self.pipe.maybe_free_model_hooks()

        return_dict = {"T": int(T), "timesteps": self.pipe.scheduler.timesteps.tolist(),
                       "samples": latents_noisy, "denoised_samples": latents, "image": image}

        if denoised_images is not None:
            denoised_images_pt = torch.nan_to_num(denoised_images_pt)
            return_dict.update(denoised_images_pt=denoised_images_pt,
                               denoised_images_pil=denoised_images_pil)
            if self.depth_estimator is not None:
                depth_images_pt = self.depth_estimator.infer(denoised_images_pt.float())
                return_dict.update(depth_images_pt=depth_images_pt)

        return return_dict

    def __call__(self, *args, **kwargs) -> StableDiffusionGuidanceOutput:

        denoise_outputs = self.denoise(*args, **kwargs)
        T = denoise_outputs.pop("T")
        timesteps = denoise_outputs.pop("timesteps")
        samples = denoise_outputs.pop("samples")
        denoised_samples = denoise_outputs.pop("denoised_samples")
        denoised_images_pt = denoise_outputs.pop("denoised_images_pt", None)
        denoised_images_pil = denoise_outputs.pop("denoised_images_pil", None)
        depth_images_pt = denoise_outputs.pop("depth_images_pt", None)

        return StableDiffusionGuidanceOutput(
            images_pil=denoised_images_pil, images_pt=denoised_images_pt, T=T, timesteps=timesteps,
            depths_pt=depth_images_pt
        )

    def enable_xformers_memory_efficient_attention(self):
        self.pipe.enable_xformers_memory_efficient_attention()

    def disable_xformers_memory_efficient_attention(self):
        self.pipe.disable_xformers_memory_efficient_attention()
