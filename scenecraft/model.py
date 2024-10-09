from __future__ import annotations

import importlib
import inspect
import re
import os
import torch
import numpy as np
import torch.nn.functional as F
import PIL
import accelerate
from torchvision import models
from torch import Tensor, nn
from dataclasses import dataclass, field
from typing import Callable, Type, Literal, Tuple, Dict, List, Union, Optional, Sequence, Any

from diffusers import StableDiffusionControlNetPipeline as _StableDiffusionControlNetPipeline, \
                      UNet2DConditionModel as _UNet2DConditionModel, UniPCMultistepScheduler, DDIMScheduler, AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.configuration_utils import register_to_config, logger
from diffusers.models.controlnet import ControlNetOutput
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    ControlNetModel as _ControlNetModel, MultiControlNetModel as _MultiControlNetModel,
    PipelineImageInput, replace_example_docstring, is_compiled_module, EXAMPLE_DOC_STRING)
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils import DIFFUSERS_CACHE, HF_HUB_OFFLINE, SAFETENSORS_WEIGHTS_NAME, is_accelerate_available, __version__
from diffusers.models.modeling_utils import _get_model_file, _add_variant, load_state_dict, load_model_dict_into_meta
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.field_heads import PredNormalsFieldHead
from nerfstudio.fields.nerfacto_field import NerfactoField as _NerfactoField
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler as _ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.losses import L1Loss, MSELoss, interlevel_loss, distortion_loss, \
                                                                scale_gradients_by_distance_squared
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

from scenecraft.utils import LoggableConfig, markvar, patchize, pil_to_numpy, numpy_to_pt, get_correspondence, to_matrix


class OverwriteMixin:
    """Overwrite some methods."""

    ignore_for_config = []

    @staticmethod
    def _get_init_keys(cls):
        keys = set(dict(inspect.signature(cls.__init__).parameters).keys())
        keys.remove("self")
        # remove general kwargs if present in dict
        if "kwargs" in keys:
            keys.remove("kwargs")
        return keys

    @classmethod
    # avoid the newly added attributes been filtered
    def extract_init_dict(cls, config_dict, **kwargs):
        # Skip keys that were not present in the original config, so default __init__ values were used
        used_defaults = config_dict.get("_use_default_values", [])
        config_dict = {k: v for k, v in config_dict.items() if k not in used_defaults and k != "_use_default_values"}

        # 0. Copy origin config dict
        original_dict = dict(config_dict.items())

        # 1. Retrieve expected config attributes from __init__ signature
        expected_keys = cls._get_init_keys(cls)

        # 2. Remove attributes that cannot be expected from expected config attributes
        # remove keys to be ignored
        if len(cls.ignore_for_config) > 0:
            expected_keys = expected_keys - set(cls.ignore_for_config)

        # NOTE: load diffusers library to import original module
        # hard code here, we force the inherited class has the same name!
        diffusers_library = importlib.import_module("diffusers")
        orig_cls_name = cls.__name__
        if hasattr(diffusers_library, orig_cls_name):
            orig_cls = getattr(diffusers_library, orig_cls_name)
            orig_expected_keys = cls._get_init_keys(orig_cls)
            unexpected_keys_from_orig = expected_keys - orig_expected_keys
            if len(unexpected_keys_from_orig) > 0:
                logger.warning(
                    f"The expected attributes of {cls} have more {unexpected_keys_from_orig} than original {orig_cls}.")
            # NOTE: merge the keys to avoid the missing attributes when loading from configs
            # hard code here, we only consider the first parent class.
            expected_keys = expected_keys | orig_expected_keys

        # remove private attributes
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

        # 3. Create keyword arguments that will be passed to __init__ from expected keyword arguments
        init_dict = {}
        for key in expected_keys:
            # if config param is passed to kwarg and is present in config dict
            # it should overwrite existing config dict key
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)

            if key in kwargs:
                # overwrite key
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                # use value from config dict
                init_dict[key] = config_dict.pop(key)

        # 4. Give nice warning if unexpected values have been passed
        if len(config_dict) > 0:
            logger.warning(
                f"The config attributes {config_dict} were passed to {cls.__name__}, "
                "but are not expected and will be ignored. Please verify your "
                f"{cls.config_name} configuration file."
            )

        # 5. Give nice info if config attributes are initialized to default because they have not been passed
        passed_keys = set(init_dict.keys())
        if len(expected_keys - passed_keys) > 0:
            logger.warning(
                f"{expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
            )

        # 6. Define unused keyword arguments
        unused_kwargs = {**config_dict, **kwargs}

        # 7. Define "hidden" config parameters that were saved for compatible classes
        hidden_config_dict = {k: v for k, v in original_dict.items() if k not in init_dict}

        return init_dict, unused_kwargs, hidden_config_dict


import torch.utils.checkpoint
from diffusers.utils import is_torch_version
from diffusers.utils.torch_utils import apply_freeu
class UNet2DConditionModel(OverwriteMixin, _UNet2DConditionModel):

    @register_to_config
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # additional operations
        pass

    def forward(self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        timestep_cond: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True, **kwargs
    ) -> Union[UNet2DConditionOutput, Tuple]:

        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        if cross_attention_kwargs is None: cross_attention_kwargs = dict()
        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            assert not downsample_block.gradient_checkpointing, f"Checkpointing not allowed."

            # -------------------------------------------- [start] forward `CrossAttnDownBlock2D` --------------------------------------------
            res_samples = ()
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:

                blocks = list(zip(downsample_block.resnets, downsample_block.attentions))
                for k, (resnet, attn) in enumerate(blocks):
                    sample = resnet(sample, emb, scale=lora_scale)
                    sample = attn(
                        sample,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=cross_attention_kwargs.get("attention_mask", None),
                        encoder_attention_mask=cross_attention_kwargs.get("encoder_attention_mask", None),
                        return_dict=False,
                    )[0]

                    # apply additional residuals to the output of the last pair of resnet and attention blocks
                    additional_residuals = cross_attention_kwargs.get("additional_residuals", None)
                    if k == len(blocks) - 1 and additional_residuals is not None:
                        sample = sample + additional_residuals

                    res_samples = res_samples + (sample,)

            else:
                for resnet in downsample_block.resnets:
                    sample = resnet(sample, emb, scale=lora_scale)

                    res_samples = res_samples + (sample,)

            # forward `CrossAttnDownBlock2D`
            if downsample_block.downsamplers is not None:
                for downsampler in downsample_block.downsamplers:
                    sample = downsampler(sample, scale=lora_scale)

                res_samples = res_samples + (sample,)

            # -------------------------------------------- [end] forward `CrossAttnDownBlock2D` --------------------------------------------

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            assert not self.mid_block.gradient_checkpointing, f"Checkpointing not allowed."
            sample = self.mid_block.resnets[0](sample, emb, scale=lora_scale)

            for attn, resnet in zip(self.mid_block.attentions, self.mid_block.resnets[1:]):
                sample = attn(
                    sample,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=cross_attention_kwargs.get("attention_mask", None),
                    encoder_attention_mask=cross_attention_kwargs.get("encoder_attention_mask", None),
                    return_dict=False,
                )[0]
                sample = resnet(sample, emb, scale=lora_scale)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        # -------------------------------------------- [start] forward `CrossAttnUpBlock2D` --------------------------------------------
        epi_weights_indice = [3, 2, 1, 0]
        for i, upsample_block in enumerate(self.up_blocks):
            assert not upsample_block.gradient_checkpointing, f"Checkpointing not allowed."
            assert not (
                        getattr(upsample_block, "s1", None)
                        and getattr(upsample_block, "s2", None)
                        and getattr(upsample_block, "b1", None)
                        and getattr(upsample_block, "b2", None)
                    ), f"FreeU not allowed."

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            # forward `CrossAttnUpBlock2D`
            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:

                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):

                    # pop res hidden states
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    sample = torch.cat([sample, res_hidden_states], dim=1)

                    sample = resnet(sample, emb, scale=lora_scale)
                    sample = attn(
                        sample,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=cross_attention_kwargs.get("attention_mask", None),
                        encoder_attention_mask=cross_attention_kwargs.get("encoder_attention_mask", None),
                        return_dict=False,
                    )[0]

            else:

                for resnet in upsample_block.resnets:

                    # pop res hidden states
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    sample = torch.cat([sample, res_hidden_states], dim=1)

                    sample = resnet(sample, emb, scale=lora_scale)

            if upsample_block.upsamplers is not None:
                for upsampler in upsample_block.upsamplers:
                    sample = upsampler(sample, upsample_size, scale=lora_scale)

        # -------------------------------------------- [end] forward `CrossAttnUpBlock2D` --------------------------------------------

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
                        subfolder: Optional[Union[str, os.PathLike]], **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)
        variant = kwargs.pop("variant", None)

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            raise RuntimeError(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment.")

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            local_files_only=local_files_only,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            user_agent=user_agent,
            **kwargs,
        )

        # load model
        model_file = _get_model_file(
            pretrained_model_name_or_path,
            weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
            cache_dir=cache_dir,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=local_files_only,
            use_auth_token=None,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            commit_hash=commit_hash,
        )

        # Instantiate model with empty weights
        with accelerate.init_empty_weights():
            model: UNet2DConditionModel = cls.from_config(config, **unused_kwargs)

        # if device_map is None, load the state dict and move the params from meta device to the cpu
        # if this model is from pretrained runwayml/stable_diffusion_1.5, then the caa parameters are initialized
        # if this model is from finetuned one, then the caa parameters are loaded
        param_device = "cpu"
        state_dict = load_state_dict(model_file, variant=variant)
        model._convert_deprecated_attention_blocks(state_dict)
        # move the params from meta device to cpu
        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"
                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                " those weights or else make sure your checkpoint file is correct."
            )

        unexpected_keys = load_model_dict_into_meta(
            model,
            state_dict,
            device=param_device,
            dtype=torch_dtype,
            model_name_or_path=pretrained_model_name_or_path,
        )

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warn(
                f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
            )

        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model.to(torch_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model


class ControlNetModel(OverwriteMixin, _ControlNetModel):
    r"""
    A ControlNet model.

    Args:
        make_embedding_reduce_layer: (`bool`, defaults to False).
        embedding_dim: (`int`, defaults to 768).
        conditioning_channels: (`int`, defaults to 3) if using rgb image as the condition
            image, it will be 3; Otherwise use different values e.g. 8 when using embedding
            image as the condition image.
    """
    @register_to_config
    def __init__(self, make_embedding_reduce_layer: bool = False, make_one_hot_embedding_layer: bool = False,
                 num_embeddings: int = 21, embedding_dim: int = 768, conditioning_channels: int = 3, **kwargs):
        super().__init__(conditioning_channels=conditioning_channels, **kwargs)
        # Force the channel order so that the channel won't be changed
        if self.config.get("controlnet_conditioning_channel_order") != "rgb":
            raise ValueError("`controlnet_conditioning_channel_order` must be `rgb`.")

        self.embedding_reduce = nn.Linear(embedding_dim, conditioning_channels) \
                                        if make_embedding_reduce_layer else None
        self.one_hot_embedding = nn.Embedding(num_embeddings, conditioning_channels) \
                                        if make_one_hot_embedding_layer else None

    @classmethod
    def from_unet(
        cls, unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        load_weights_from_unet: bool = True,
        **kwargs
    ):
        # parameters of embedding projection layer
        make_embedding_reduce_layer = kwargs.pop("make_embedding_reduce_layer", False)
        make_one_hot_embedding_layer = kwargs.pop("make_one_hot_embedding_layer", False)
        num_embeddings = kwargs.pop("num_embeddings", 21)
        embedding_dim = kwargs.pop("embedding_dim", None)
        conditioning_channels = kwargs.pop("conditioning_channels", 3)

        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            make_embedding_reduce_layer=make_embedding_reduce_layer,
            make_one_hot_embedding_layer=make_one_hot_embedding_layer,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            conditioning_channels=conditioning_channels,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    def forward(self, controlnet_cond: Union[torch.FloatTensor, torch.LongTensor],
                indice_to_embedding: torch.FloatTensor = None, **kwargs) -> Union[ControlNetOutput, Tuple]:

        kwargs.pop("no_depth_cond", None)
        if not isinstance(controlnet_cond, torch.Tensor):
            raise TypeError(f"Couldn't handle type of input controlent condition image {type(controlnet_cond)}.")

        # label embedding projection if called controlnet outside `pipeline`
        # refer to `StableDiffusionControlNetPipeline.prepare_image`.
        if indice_to_embedding is not None:
            assert self.one_hot_embedding is None, f"Too much embedding layers."
            # convert label ids to embeddings
            if not (controlnet_cond.ndim == 3 or controlnet_cond.ndim == 2):
                raise ValueError(f"Wrong shape of input condition image, expected 3D or 2D array consists "
                                 f"of label ids, got shape {controlnet_cond.shape}.")
            if controlnet_cond.ndim == 2: controlnet_cond = controlnet_cond[None, ...] # add batch_size dimension
            controlnet_cond[controlnet_cond == 255] = -1.

            if self.embedding_reduce is not None:
                if indice_to_embedding.shape[-1] != self.config.embedding_dim:
                    raise ValueError(f"Wrong class embedding dimension, expected {self.config.embedding_dim}, "
                                    f"got {indice_to_embedding.shape[-1]}.")
                indice_to_embedding = self.embedding_reduce(indice_to_embedding)
            if indice_to_embedding.shape[-1] != self.controlnet_cond_embedding.conv_in.in_channels:
                raise ValueError(f"Wrong indice_to_embedding dimension, expected {self.controlnet_cond_embedding.conv_in.in_channels}, "
                                f"got {indice_to_embedding.shape[-1]}.")
            controlnet_cond = indice_to_embedding[controlnet_cond.long()] \
                                .permute(0, 3, 1, 2).to(self.dtype)

        # check the initial indice image input with dims equals to 3 (BHW)
        if self.one_hot_embedding is not None and controlnet_cond.ndim == 3:
            assert indice_to_embedding is None, f"Too much embedding layers."
            controlnet_cond[controlnet_cond == 255] = self.one_hot_embedding.num_embeddings - 1 # background label
            controlnet_cond = self.one_hot_embedding(controlnet_cond.long()) \
                                .permute(0, 3, 1, 2).to(self.dtype)

        if controlnet_cond.shape[1] != self.config.conditioning_channels:
            raise ValueError(f"Wrong channels of input condition, expected {self.config.conditioning_channels}, "
                             f"got {controlnet_cond.shape[1]}.")
        return super().forward(controlnet_cond=controlnet_cond, **kwargs)


class MultiControlNetModel(OverwriteMixin, _MultiControlNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        model_path_to_load = pretrained_model_path
        subfolder = kwargs.get("subfolder")

        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        controlnets = [
            ControlNetModel.from_pretrained(model_path_to_load, subfolder=f"{subfolder}/multicontrolnetmodel", torch_dtype=torch_dtype),
            ControlNetModel.from_pretrained(model_path_to_load, subfolder=f"{subfolder}/multicontrolnetmodel_1", torch_dtype=torch_dtype)
        ]

        logger.info(f"{len(controlnets)} controlnets loaded from {pretrained_model_path}.")

        return cls(controlnets)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.Tensor],
        conditioning_scale: List[float],
        **kwargs
    ) -> Union[ControlNetOutput, Tuple]:
        assert isinstance(controlnet_cond, (list, tuple))
        assert isinstance(conditioning_scale, (list, tuple))
        # semantic controlnet
        down_samples, mid_sample = self.nets[0](
            sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond[0], conditioning_scale=conditioning_scale[0], **kwargs
        )
        if not kwargs.get("no_depth_cond", False):
            # depth controlnet
            kwargs.pop("indice_to_embedding", None)
            down_samples2, mid_sample2 = self.nets[1](
                sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond[1], conditioning_scale=conditioning_scale[1], **kwargs
            )
            down_block_res_samples = [
                samples + samples2 for samples, samples2 in zip(down_samples, down_samples2)
            ]
            mid_block_res_sample = mid_sample + mid_sample2
        else:
            down_block_res_samples = down_samples
            mid_block_res_sample = mid_sample

        return down_block_res_samples, mid_block_res_sample


class StableDiffusionControlNetPipeline(OverwriteMixin, _StableDiffusionControlNetPipeline):

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    controlnet: ControlNetModel
    scheduler: Union[UniPCMultistepScheduler, DDIMScheduler]

    # TODO: remove `check_inputs` when pipeline is finalized.
    # Now modify this func to accept multiple batches for mutliple controlnets
    def check_inputs(
        self,
        prompt, image,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
            self.controlnet, torch._dynamo.eval_frame.OptimizedModule
        )
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            self.check_image(image, prompt, prompt_embeds)
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if not isinstance(image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in image):
                # raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                pass
            elif len(image) != len(self.controlnet.nets):
                raise ValueError(
                    f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                )

            for image_ in image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if (
            isinstance(self.controlnet, ControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, ControlNetModel)
        ):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif (
            isinstance(self.controlnet, MultiControlNetModel)
            or is_compiled
            and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
        ):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        if isinstance(self.controlnet, MultiControlNetModel):
            if len(control_guidance_start) != len(self.controlnet.nets):
                raise ValueError(
                    f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    def prepare_image(self, image, image_type: str = "rgb", indice_to_embedding: torch.FloatTensor = None, **kwargs):

        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        if isinstance(image, supported_formats): image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        # before feed "indice" type condition image to `control_image_process`, we will convert them to "embedding"
        # type image, so that `VaeImageProcessor` could take them as the normal RGB image.
        if image_type == "indice" or image_type == "depth":
            assert isinstance(image[0], (PIL.Image.Image, np.ndarray, torch.Tensor)), f"Unsupported type of indice condition image {type(image[0])}."
            if isinstance(image[0], PIL.Image.Image): image = pil_to_numpy(image, image_type)
            if isinstance(image[0], np.ndarray): image = numpy_to_pt(image, image_type)
            if isinstance(image[0], torch.Tensor) and isinstance(image, (tuple, list)): image = torch.cat(image)

            if image_type == "indice":
                controlnet = self.controlnet if not isinstance(self.controlnet, MultiControlNetModel) else self.controlnet.nets[0]

                if controlnet.embedding_reduce is not None:
                    image[image == 255] = -1 # background label
                    assert indice_to_embedding is not None, "Missing `indice_to_embedding` for given indice condition image, " \
                                                            "check your `condition_type` should be `embedding`."
                    if indice_to_embedding.shape[-1] != controlnet.config.embedding_dim:
                        raise ValueError(f"Wrong class embedding dimension, expected {controlnet.config.embedding_dim}, "
                                        f"got {indice_to_embedding.shape[-1]}.")
                    indice_to_embedding = controlnet.embedding_reduce(indice_to_embedding)
                    image = indice_to_embedding[image.long()].permute(0, 3, 1, 2) # BCHW

                elif indice_to_embedding is not None:
                    if indice_to_embedding.shape[-1] != controlnet.controlnet_cond_embedding.conv_in.in_channels:
                        raise ValueError(f"Wrong indice_to_embedding dimension, expected {controlnet.controlnet_cond_embedding.conv_in.in_channels}, "
                                        f"got {indice_to_embedding.shape[-1]}.")
                    image = indice_to_embedding[image.long()].permute(0, 3, 1, 2) # BCHW

                elif controlnet.one_hot_embedding is not None:
                    image[image == 255] = controlnet.one_hot_embedding.num_embeddings - 1
                    image = image.to(controlnet.device)
                    image = controlnet.one_hot_embedding(image.long()).permute(0, 3, 1, 2) # BCHW

            image = torch.nan_to_num(image)
            image[torch.isinf(image)] = 0.

        return super().prepare_image(image, **kwargs)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        condition_type: str = "rgb",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        return_dict: bool = True,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        **kwargs,
    ):
        r"""
        The call function inherited from parent class.

        Args:
            (other args could refer to the docstring under parent class)
            embedding (`torch.FloatTensor): the embedding of current scene.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        negative_prompt = kwargs.pop("negative_prompt", None)
        num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)
        eta = kwargs.pop("eta", 0.0)
        generator = kwargs.pop("generator", None)
        latents = kwargs.pop("latents", None)
        prompt_embeds = kwargs.pop("prompt_embeds", None)
        negative_prompt_embeds = kwargs.pop("negative_prompt_embeds", None)
        indice_to_embedding = kwargs.pop("indice_to_embedding", None)
        output_type = kwargs.pop("output_type", "pil")
        return_dict = kwargs.pop("return_dict", True)
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", 1)
        cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", None)
        no_depth_cond = kwargs.pop("no_depth_cond", False)

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [control_guidance_end]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                image_type=("indice" if condition_type != "rgb" else condition_type),
                indice_to_embedding=indice_to_embedding,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            # Length of image: len(image) = len(controlnet.nets) = 2 => two controlnets
            # For single batch, image => [rgb, depth] => batch_size = 1
            # For multiple batch, image => [[rgb1, rgb2, ...], [depth1, depth2, ...]] => batch_size > 1
            # Length of prompts should be equal to batch_size
            for i, image_ in enumerate(image):
                if i == 0:
                    image_type = "indice" if condition_type != "rgb" else condition_type
                else:
                    image_type = "depth"
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    image_type=image_type,
                    indice_to_embedding=indice_to_embedding,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=False,
                )

                images.append(image_)

            assert len(images) == len(controlnet.nets), f"Length of images {len(images)} isn't equal to length of `MultiControlNetModel` {len(controlnet.nets)}."
            assert images[0].shape[0] == batch_size * (1 if not do_classifier_free_guidance else 2), \
                f"Batch size of output image tensors isn't equal to batch size of input images."

            image = images
            height, width = image[0].shape[-2:]

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    sample=control_model_input,
                    timestep=t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    no_depth_cond=no_depth_cond,
                    guess_mode=False,
                    return_dict=False,
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image, has_nsfw_concept

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)


class StableDiffusionControlNetInpaintPipeline(StableDiffusionControlNetPipeline):

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    controlnet: ControlNetModel
    scheduler: Union[UniPCMultistepScheduler, DDIMScheduler]

    def __init__(self, **kwargs):
        from diffusers.image_processor import VaeImageProcessor
        super().__init__(**kwargs)
        self.init_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    # copied from StableDiffusionInpaintPipeline
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            from diffusers.utils.torch_utils import randn_tensor
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        init_image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.Tensor = None,
        strength: float = 1.0,
        condition_type: str = "rgb",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        return_dict: bool = True,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        **kwargs,
    ):
        r"""
        The call function inherited from parent class.

        Args:
            (other args could refer to the docstring under parent class)
            embedding (`torch.FloatTensor): the embedding of current scene.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        negative_prompt = kwargs.pop("negative_prompt", None)
        num_images_per_prompt = kwargs.pop("num_images_per_prompt", 1)
        eta = kwargs.pop("eta", 0.0)
        generator = kwargs.pop("generator", None)
        latents = kwargs.pop("latents", None)
        prompt_embeds = kwargs.pop("prompt_embeds", None)
        negative_prompt_embeds = kwargs.pop("negative_prompt_embeds", None)
        indice_to_embedding = kwargs.pop("indice_to_embedding", None)
        output_type = kwargs.pop("output_type", "pil")
        return_dict = kwargs.pop("return_dict", True)
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", 1)
        cross_attention_kwargs = kwargs.pop("cross_attention_kwargs", None)
        no_depth_cond = kwargs.pop("no_depth_cond", False)

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [control_guidance_end]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            image,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                image_type=("indice" if condition_type != "rgb" else condition_type),
                indice_to_embedding=indice_to_embedding,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=False,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            # Length of image: len(image) = len(controlnet.nets) = 2 => two controlnets
            # For single batch, image => [rgb, depth] => batch_size = 1
            # For multiple batch, image => [[rgb1, rgb2, ...], [depth1, depth2, ...]] => batch_size > 1
            # Length of prompts should be equal to batch_size
            for i, image_ in enumerate(image):
                if i == 0:
                    image_type = "indice" if condition_type != "rgb" else condition_type
                else:
                    image_type = "depth"
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    image_type=image_type,
                    indice_to_embedding=indice_to_embedding,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=False,
                )

                images.append(image_)

            assert len(images) == len(controlnet.nets), f"Length of images {len(images)} isn't equal to length of `MultiControlNetModel` {len(controlnet.nets)}."
            assert images[0].shape[0] == batch_size * (1 if not do_classifier_free_guidance else 2), \
                f"Batch size of output image tensors isn't equal to batch size of input images."

            image = images
            height, width = image[0].shape[-2:]

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4
        if num_channels_unet != 4:
            raise RuntimeError("Error!")

        original_image = init_image
        init_image = self.init_image_processor.preprocess(
            init_image, height=height, width=width).to(dtype=torch.float32)

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 6.1 Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width)

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        # mask image latents only be used to concat with the input image latents
        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    sample=control_model_input,
                    timestep=t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    no_depth_cond=no_depth_cond,
                    guess_mode=False,
                    return_dict=False,
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if num_channels_latents == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image, has_nsfw_concept

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)

class PerceptualLoss(torch.nn.Module):

    def __init__(self, model='vgg16', device: torch.device=None, weigths=None) -> None:
        super(PerceptualLoss, self).__init__()

        assert model.startswith('vgg'), f"Use vgg model."
        self._vgg = getattr(models, model)(pretrained=True)
        if device is not None:
            self._vgg = self._vgg.to(device)
        self.layers = [4, 9, 16, 23]
        self.weights = weigths if weigths is not None else [1] * len(self.layers)
        self.loss_network = self._vgg.features[:self.layers[-1] + 1].eval()
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = torch.nn.L1Loss()

    def _extract_features(self, x: Tensor) -> List[Tensor]:
        x_vgg = []
        for layer in self.loss_network:
            x = layer(x)
            x_vgg.append(x)
        return x_vgg

    def _gram_mat(self, x: Tensor):
        n, c, h, w = x.shape
        features = x.reshape(n, c, h * w)
        features = features / torch.norm(features, dim=1, keepdim=True) / (h * w) ** 0.5
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        return gram

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        if device is not None:
            assert isinstance(device, torch.device)
            self._vgg.to(device)
            self.loss_network.to(device)
        if dtype is not None:
            assert isinstance(dtype, torch.dtype)
            self._vgg.to(dtype)
            self.loss_network.to(dtype)
        return self

    @property
    def dtype(self):
        return list(self._vgg.parameters())[0].dtype

    def forward(self, out_images: Tensor, target_images: Tensor,
                patch: bool = False, patch_size: int = 224) -> Tensor:
        if patch:
            out_images = patchize(out_images.float(), patch_size=patch_size)
            target_images = patchize(target_images.float(), patch_size=patch_size)
        out_images = out_images.to(self.dtype)
        target_images = target_images.to(self.dtype)

        input_features, target_features = self._extract_features(out_images), \
                                          self._extract_features(target_images)
        percep_loss = 0
        for weight, layer in zip(self.weights, self.layers):
            percep_loss += weight * self.criterion(input_features[layer].float(), target_features[layer].float())

        style_loss = 0
        for weight, layer in zip(self.weights, self.layers):
            loss = weight * self.criterion(self._gram_mat(input_features[layer]).float(), self._gram_mat(target_features[layer]).float())
            if not (torch.isnan(loss) or torch.isinf(loss)):
                style_loss += loss

        return percep_loss, style_loss


def zvar_loss(weights_list, ray_samples_list, depth, accumulation):
    ray_samples = ray_samples_list[-1]
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2 
    weights = weights_list[-1]
    mask = (accumulation[:, None] > 0.5).float()
    loss_zvar = (mask * (steps - depth[:, None]) ** 2 * weights / (accumulation[:, None] + 1e-10)).mean() # (N_rays, 1)
    return loss_zvar


@dataclass
class SceneCraftModelConfig(NerfactoModelConfig, LoggableConfig):
    """Configuration for the SceneCraftNeRFModel."""
    _target: Type = field(default_factory=lambda: SceneCraftNeRFModel)

    proposal_weights_anneal_max_num_iters: int = markvar(2500)
    """Max num iterations for the annealing function."""
    near_plane: float = markvar(0.05)
    """How far along the ray to start sampling."""
    far_plane: float = markvar(1000.)
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""
    rgb_as_latents: bool = False
    """Whether to use rgb as latents."""
    train_num_rays_per_chunk: int = markvar(1 << 16)
    """Number of rays per chunk for training."""
    eval_num_rays_per_chunk: int = markvar(4096)
    """specifies number of rays per chunk during eval"""

    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = markvar(2)
    """Sample every n steps after the warmup"""
    proposal_warmup: int = markvar(500)
    """Scales n from 1 to proposal_update_every over this many steps"""

    use_lpips: bool = markvar(True)
    """Whether to use perceptual loss"""
    use_l1: bool = markvar(True)
    """Whether to use L1 loss for rgb loss"""
    use_latent_loss: bool = markvar(False)
    """Whether to use latent loss"""
    random_crop: bool = markvar(False)
    """Whether to crop image randomly before loss"""
    patch_size: int = markvar(32)
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = markvar(0.1)
    """Multiplier for LPIPS loss."""
    style_loss_mult: float = markvar(30.)
    """Multiplier for style loss."""
    latent_loss_mult: float = markvar(0.8)
    """Muliplier for latent loss."""
    rgb_loss_mult: float = markvar(5.)
    """Multiplier for RGB loss."""
    interlevel_loss_mult: float = markvar(.5)
    """Proposal loss multiplier."""
    distortion_loss_mult: float = markvar(0.002)
    """Distortion loss multiplier."""
    depth_loss_mult: float = markvar(0.5)
    """Depth loss multiplier."""
    guide_depth_loss_mult: float = markvar(0.2)
    """Guide depth loss multiplier."""
    depth_consistency_loss_mult: float = markvar(5.)
    """Depth loss multiplier."""
    zvar_loss_mult: float = markvar(1.)
    """zvar loss multiplier."""
    use_appearance_embedding: bool = markvar(False)
    """Whether to use appearance embedding."""
    use_direction_encoding: bool = markvar(False)
    """Whether to use direction embedding."""
    compute_background_color: bool = markvar(False)
    """Whether to compute background color."""

    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""


class NerfactoField(_NerfactoField):

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        pass_semantic_gradients: bool = False,
        appearance_embedding_dim: int = 32,
        use_pred_normals: bool = False,
        use_direction_encoding: bool = False,
        use_appearance_embedding: bool = False,
        use_average_appearance_embedding: bool = False,
        compute_background_color: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ):
        # Parent of `_NerfactoField`
        Field.__init__(self)

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.use_pred_normals = use_pred_normals
        self.use_direction_encoding = use_direction_encoding
        self.appearance_embedding_dim = appearance_embedding_dim
        self.use_appearance_embedding = use_appearance_embedding
        self.use_average_appearance_embedding = use_average_appearance_embedding
        if self.use_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.compute_background_color =  compute_background_color
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res

        if self.compute_background_color or self.use_direction_encoding:
            self.direction_encoding = SHEncoding(
                levels=4,
                implementation=implementation,
            )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

        # predicted normals
        if self.use_pred_normals:
            self.mlp_pred_normals = MLP(
                in_dim=self.geo_feat_dim + self.position_encoding.get_out_dim(),
                num_layers=3,
                layer_width=64,
                out_dim=hidden_dim_transient,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.field_head_pred_normals = PredNormalsFieldHead(in_dim=self.mlp_pred_normals.get_out_dim())

        if self.compute_background_color:
            self.mlp_background_color = MLP(
                in_dim=self.direction_encoding.get_out_dim(),
                num_layers=2,
                layer_width=32,
                out_dim=3,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )

        self.mlp_head = MLP(
            in_dim=self.geo_feat_dim
                   + (self.direction_encoding.get_out_dim() if self.use_direction_encoding else 0)
                   + (self.appearance_embedding_dim if self.use_appearance_embedding else 0),
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_background_rgb(self, ray_bundle: RayBundle) -> Tensor:
        """Predicts background colors at infinity."""
        directions = get_normalized_directions(ray_bundle.directions)

        outputs_shape = ray_bundle.directions.shape[:-1]
        directions_flat = self.direction_encoding(directions.view(-1, 3))
        background_rgb = self.mlp_background_color(directions_flat).view(*outputs_shape, -1).to(directions)

        return background_rgb

    # The checkpointing doesn't support return type to be `dict`, refer to
    # https://stackoverflow.com/questions/63102642/gradient-checkpointing-returning-values
    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tensor = None, *args, **kwargs) -> List[Tensor]:
        assert density_embedding is not None
        outputs = []
        directions = get_normalized_directions(ray_samples.frustums.directions)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        embedded_appearance = None
        if self.use_appearance_embedding:
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        # direction
        direction_encoding = None
        if self.use_direction_encoding:
            directions_flat = directions.view(-1, 3)
            direction_encoding = self.direction_encoding(directions_flat)
        

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs.append(self.field_head_pred_normals(x))

        h = torch.cat(
            (
                [direction_encoding] if self.use_direction_encoding else [] 
            )
            + [
                density_embedding.view(-1, self.geo_feat_dim)
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if self.use_appearance_embedding else []
            ),
            dim= -1
        )

        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.append(rgb)

        return outputs

    def nerf_field_get_density_repl(self, ray_samples):
        from nerfstudio.data.scene_box import SceneBox
        from nerfstudio.field_components.activations import trunc_exp

        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)

        with torch.no_grad():
            depth = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2.0
            selector_pre = (depth > 0.1).all(dim=-1)
            # print(f"render_depth_min: {FLAGS['render_depth_min']} selector_pre: {selector_pre.float().mean()}")
            selector &= selector_pre

        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]

        return density, base_mlp_out

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False, compute_depth: bool = False, **kwargs) -> List[Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if self.training:
            density, density_embedding = self.get_density(ray_samples)
        else:
            density, density_embedding = self.nerf_field_get_density_repl(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding, **kwargs)
        field_outputs += [density]  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs += [normals]  # type: ignore
        return field_outputs


class ProposalNetworkSampler(_ProposalNetworkSampler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.updated_ray_samples = False

    # Since we need to call `ProposalNetworkSampler` multiple itmes for each train/eval step,
    # rather than only once for the original nerfstudio so need to change the policy of `updated`.
    def step_cb(self, step):
        """Callback to register a training step has passed. This is used to keep track of the sampling schedule"""
        self._step = step
        if self.updated_ray_samples:
            self._steps_since_update = 0
        self._steps_since_update += 1

    # Since `step_cb` is a training callback term so the backward keeps the same behavior as the forward.
    def generate_ray_samples(self, ray_bundle: RayBundle, density_fns: List[Callable],
                             ) -> Tuple[RaySamples, List, List]:

        weights_list = []
        ray_samples_list = []

        n = self.num_proposal_network_iterations
        self.updated_ray_samples = self._steps_since_update > self.update_sched(self._step) or self._step < 10

        weights = None
        ray_samples: RaySamples
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                # Uniform sampling because we need to start with some samples
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                # PDF sampling based on the last samples and their weights
                # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
                assert weights is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                if self.updated_ray_samples:
                    # always update on the first step or the inf check in grad scaling crashes
                    density = density_fns[i_level](ray_samples.frustums.get_positions())
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.frustums.get_positions())
                weights = ray_samples.get_weights(density)
                weights_list.append(weights)  # (num_rays, num_samples)
                ray_samples_list.append(ray_samples)

        return ray_samples, weights_list, ray_samples_list


class SceneCraftNeRFModel(NerfactoModel):
    """Model for scenecraft."""

    config: SceneCraftModelConfig
    field: NerfactoField
    # `self.device` and `self._device` are equal to solve the problem
    # of call `self.device` before it is assigned valid value.
    _device: torch.device = None

    def populate_modules(self):
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        self.enable_depth_loss = True

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_appearance_embedding=self.config.use_appearance_embedding,
            use_direction_encoding=self.config.use_direction_encoding,
            compute_background_color=self.config.compute_background_color,
            implementation=self.config.implementation,
        )

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # loss
        self.rgb_loss = L1Loss() if self.config.use_l1 else MSELoss()
        if self.config.use_latent_loss:
            from diffusers import AutoencoderKL
            self.vae: AutoencoderKL = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
            self.latent_loss = L1Loss() if self.config.use_l1 else MSELoss()
        if self.config.use_lpips:
            self.ploss = PerceptualLoss(device=self._device)
        self.depth_loss = MSELoss()

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        # Expect `field_outputs` to be [`rgb`, `density`, `normals`]
        field_outputs = self.field.forward(ray_samples,
                                           compute_normals=self.config.predict_normals,
                                           compute_depth=self.enable_depth_loss)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(densities=field_outputs[1])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[0], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        if self.config.compute_background_color:
            background_rgb = self.field.get_background_rgb(ray_bundle)
            accum_mask = torch.clamp((torch.nan_to_num(accumulation, nan=0.0)), min=0.0, max=1.0)
            rgb = accum_mask * rgb + (1.0 - accum_mask) * background_rgb

        normals = None
        if len(field_outputs) > 2:
            normals = self.renderer_normals(normals=field_outputs[2], weights=weights)

        return rgb, weights_list, ray_samples_list, depth, expected_depth, accumulation, normals

    def get_metrics_dict(self, outputs, batch):
        raise NotImplementedError

    def get_loss_dict(self, outputs: dict, batch: dict = None) -> Dict[str, torch.Tensor]:
        loss_dict = {}

        if "weights_list" in outputs and "ray_samples_list" in outputs:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            loss_dict["zvar_loss"] = self.config.zvar_loss_mult * zvar_loss(
                outputs["weights_list"], outputs["ray_samples_list"], outputs["depth"], outputs["accumulation"]
            )

        if batch is None:
            return loss_dict

        # calculate (train pixels) guide loss for nerf model
        if "image" in batch:
            pred_image = outputs["rgb"] # [N, 3]
            image = batch["image"].to(self.device)
            loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(image, pred_image)            

        # calculate (full image) guide loss for nerf model
        # key `guide_image` must be aligned with `scenecraft.dataset.InputDataset.__getitem__()`
        if "guide_image" in batch:
            image = outputs.get("real_rgb", outputs["rgb"]).permute(0, 3, 1, 2)
            guide_image = batch["guide_image"].to(self.device)
            if image.shape[1] != guide_image.shape[1]:
                raise RuntimeError(f"Different channels between rendered image and guidance image, got ",
                                   f"{image.shape[1]} and {guide_image.shape[1]}.")
            if image.min() < -1e-4 or image.max() > 1 + 1e-4:
                CONSOLE.print(f"Expected output image to be in range [0, 1] while got [{image.min()}, {image.max()}].", style="bold yellow")
            if guide_image.min() < -1e-4 or guide_image.max() > 1+1e-4:
                CONSOLE.print(f"Expected guide image to be in range [0, 1] while got [{guide_image.min()}, {guide_image.max()}].", style="bold yellow")

            # randomly crop
            if self.config.random_crop:
                _, _, height, width = image.shape
                p = torch.randn(1) > 0
                ratio = ((torch.rand(1) + 1) / 2).item()
                if p:
                    new_height = int(height * ratio)
                    height_offset = int(height - new_height)
                    image = image[..., height_offset : height_offset + new_height, :]
                    guide_image = guide_image[..., height_offset : height_offset + new_height, :]
                else:
                    new_width = max(int(width * ratio), height)
                    width_offset = int(width - new_width)
                    image = image[..., width_offset : width_offset + new_width]
                    guide_image = guide_image[..., width_offset : width_offset + new_width]

            # rgb img loss
            grad = image - guide_image
            grad = grad.clamp(-0.5, 0.5)
            target = (image - grad).detach()
            loss_dict["rgb_loss"] = self.config.rgb_loss_mult * self.rgb_loss(image, target)

            # z img loss
            if self.config.use_latent_loss:
                latents: torch.FloatTensor = self.vae.encode(
                    F.interpolate(image, (512, 512), mode="bilinear")).latent_dist.sample()
                latents *= self.vae.config.scaling_factor
                denoised_latents = self.vae.encode(
                    F.interpolate(guide_image, (512, 512), mode="bilinear")).latent_dist.sample()
                denoised_latents *= self.vae.config.scaling_factor
                grad = latents - denoised_latents
                grad = grad.clamp(-0.5, 0.5)
                target = (latents - grad).detach()
                loss_dict["latent_loss"] = self.config.latent_loss_mult * self.latent_loss(latents, target)

            # lpips loss
            if self.config.use_lpips:
                lpips_loss, style_loss = self.ploss(image, guide_image, patch=True)
                loss_dict["lpips_loss"] = self.config.lpips_loss_mult * lpips_loss
                loss_dict["style_loss"] = self.config.style_loss_mult * style_loss

        # guidance depth loss
        if "guide_depth" in batch and batch["guide_depth"][0] is not None:
            mask = outputs["weights_mask"]
            target = batch["guide_depth"].to(self.device).reshape(-1)[mask]
            target_normalized = (target - target.mean()) / target.std()
            pred_depth = outputs["depth"].reshape(-1)
            disparity = 1 / pred_depth[mask]
            pred = disparity - disparity.mean()
            pred_rescaled = torch.sum(pred * target_normalized) / torch.sum(pred ** 2) * pred
            error = pred_rescaled - target_normalized
            loss_dict["depth_reg"] = self.config.guide_depth_loss_mult * torch.mean(error ** 2) / 2 * (256 * 256)

        # reference depth loss
        if self.enable_depth_loss and "reference_depth" in batch and isinstance(outputs.get("depth", None), torch.Tensor):
            reference_depth = batch["reference_depth"].to(self.device)[..., None] # [H, W, 1]
            depth_mask = (torch.abs(outputs["depth"] - reference_depth * batch["scale_factor"]) > 0.2
                          ) & (reference_depth > 0) & (not (outputs["depth"] > 10.).any()) & (not (reference_depth > 10.).any())
            if torch.any(depth_mask):
                loss_dict["depth_loss"] = self.config.depth_loss_mult * self.depth_loss(outputs["depth"][depth_mask],
                                                                                        reference_depth[depth_mask] * batch["scale_factor"])

        # neightbor dpeth loss
        if self.enable_depth_loss and "neighbor_depth" in batch and isinstance(outputs.get("depth", None), torch.Tensor):
            height, width = outputs["depth"].shape[:2]
            if batch["neighbor_depth"].shape != outputs["depth"].shape:
                batch["neighbor_depth"] = F.interpolate(batch["neighbor_depth"], outputs["depth"].shape, mode='nearest')
            transform = torch.linalg.inv(to_matrix(batch["neighbor_extrin"])) @ to_matrix(batch["extrin"])
            points_2d = get_correspondence(depth=outputs["depth"] / batch["scale_factor"], transform=transform,
                                           K=batch["intrin"], coords=batch["coords"])
            depth_mask = (points_2d[..., 0] >= 0) & (points_2d[..., 0] < width) * (points_2d[..., 1] > 0) & (points_2d[..., 1] < height)
            if torch.any(depth_mask):
                loss_dict["depth_consistency_loss"] = self.config.depth_consistency_loss_mult * self.depth_loss(outputs["depth"][depth_mask],
                                                                            batch["neighbor_depth"].to(self.device)[depth_mask])

        return loss_dict

    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, keep_image_shape: bool = True, *args,
                                          ) -> Union[Sequence[torch.Tensor], dict]:
        """Takes in camera parameters and computes the output of the model.
            `outputs`: (`dict`) contains `rgb`, `accumulation`, `depth`, `expected_depth`,
                       (`normals`, `pred_normals` if predict_normals), (`weights_list`, `ray_samples_list` if training),
                       (`rendered_orientation_loss`, `rendered_pred_normal_loss` if predict_normals and training),
                       `prop_depth_1`, `prop_depth_2`. Not all that we need.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.train_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        output_rgb, output_regs, output_depth, output_expected_depth, output_acc, output_normal, output_weights_mask = [], [], [], [], [], [], []
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            rgb, weights, ray_samples, depth, expected_depth, accumulation, normal = self(ray_bundle)
            output_rgb.append(rgb)
            if self.training:
                output_regs.append(sum(self.get_loss_dict(
                    dict(weights_list=weights, ray_samples_list=ray_samples, depth=expected_depth, accumulation=accumulation)).values()))
                output_weights_mask.append(weights[-1][..., 0])
            if depth is not None: output_depth.append(depth)
            if expected_depth is not None: output_expected_depth.append(expected_depth)
            if accumulation is not None: output_acc.append(accumulation)
            if normal is not None: output_normal.append(normal)
        output_rgb = torch.cat(output_rgb)
        if output_regs: output_regs = torch.stack(output_regs)
        if output_depth: output_depth = torch.cat(output_depth)
        if output_expected_depth: output_expected_depth = torch.cat(output_expected_depth)
        if output_acc: output_acc = torch.cat(output_acc)
        if output_normal: output_normal = torch.cat(output_normal)
        if output_weights_mask:
            output_weights_mask = torch.cat(output_weights_mask)
            output_weights_mask = output_weights_mask.sum(-1) > 0.5 # bool [N]
        if keep_image_shape:
            output_rgb = output_rgb.view(image_height, image_width, -1)  # type: ignore
            if isinstance(output_depth, torch.Tensor): output_depth = output_depth.view(image_height, image_width, -1)
            if isinstance(output_expected_depth, torch.Tensor): output_expected_depth = output_expected_depth.view(image_height, image_width, -1)
            if isinstance(output_acc, torch.Tensor): output_acc = output_acc.view(image_height, image_width, -1)
            if isinstance(output_normal, torch.Tensor): output_normal = output_normal.view(image_height, image_width, -1)

        if self.training:
            outputs = (output_rgb, output_regs, output_depth, output_expected_depth, output_acc, output_normal, output_weights_mask)
        else:
            outputs = dict(rgb=output_rgb, depth=output_depth, accumulation=output_acc, normal=output_normal)

        return outputs

    def update_to_step(self, step: int) -> None:
        return super().update_to_step(step)