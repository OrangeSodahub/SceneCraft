"""Pipeline and trainer"""
# Forward remote server ports to local ports:
# ssh -L 7007:localhost:7007 yxy@infinity-hn.cs.illinois.edu -N

import os
import torch
import numpy as np
import wandb
import torch.nn.functional as F
import math
import functools
from torch.nn import Parameter
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from typing import Optional, Type, Union, Literal, Tuple, Sequence, List, Any, Dict, cast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.checkpoint import CheckpointFunction as _CheckpointFunction
from torch.utils.checkpoint import check_backward_validity, detach_variable

from nerfstudio.utils import profiler, comms
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig, Pipeline
from nerfstudio.cameras.rays import RayBundle

from scenecraft.utils import LoggableConfig, setup_logger, draw_text_on_image, markvar, module_wrapper, \
                       time_function, check_local_main_thread, colorize_depth
from scenecraft.data.datamanager import *
from scenecraft.guidance import StableDiffusionGuidance
from scenecraft.model import SceneCraftModelConfig, SceneCraftNeRFModel
from scenecraft.prompt_processor import StableDiffusionPromptProcessor

logger = setup_logger(os.path.basename(__file__))


""" Customed checkpointing exclusively for Nerfacto Model """

# Adapted from 'torch.utils.checkpoint.py'
class CheckpointFunction(_CheckpointFunction):

    @staticmethod
    # Here keep a copy of `CheckpointFunctionForward` from torch2.0.1
    # to avoid the possible imcompatibility with the Backward function
    # due to the change of version of torch.  
    def forward(ctx, run_function, backward_chunk_size, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.backward_chunk_size = backward_chunk_size
        ctx.had_autocast_in_fwd = torch.is_autocast_enabled()

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # `args` should have the lenght=4:
        # args[0]: ∂loss/∂rgb = torch.autograd.grad(loss, rgb);
        # args[1]: ∂loss/∂reg; args[2]: ∂loss/∂depth or None; args[3]: None.
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # NOTE: hard code here, force the last argument to be the dummy tensor with `requires_grad=True`
        # And the first argument must be the initial input with `requires_grad=False`
        detached_inputs = detach_variable(tuple(inputs))
        ray_bundle: RayBundle = detached_inputs[0]
        num_rays = len(ray_bundle)
        partial_loss_partial_image_flattened = args[0].view(-1, 3)
        partial_loss_partial_reg = args[1]
        partial_loss_partial_expected_depth_flattened = args[3]
        partial_loss_partial_mask_flattened = args[6].view(-1)
        if partial_loss_partial_expected_depth_flattened is not None:
            partial_loss_partial_expected_depth_flattened = partial_loss_partial_expected_depth_flattened.view(-1, 1)

        if len(partial_loss_partial_reg) != math.ceil(num_rays / ctx.backward_chunk_size):
            raise RuntimeError(f"Should make sure that ray chunk sizes of both forward and backward process to be the same.")

        # Conduct backward process chunk by chuck to save the memory
        for i in range(0, num_rays, ctx.backward_chunk_size):
            start_idx = i
            end_idx = i + ctx.backward_chunk_size
            ray_chunk = ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            partial_loss_partial_image_chunk = partial_loss_partial_image_flattened[start_idx : end_idx]
            partial_loss_partial_reg_chunk = partial_loss_partial_reg[i // ctx.backward_chunk_size]
            partial_loss_partial_mask_chunk = partial_loss_partial_mask_flattened[start_idx : end_idx]
            if partial_loss_partial_expected_depth_flattened is not None:
                partial_loss_partial_expected_depth_chunk = partial_loss_partial_expected_depth_flattened[start_idx : end_idx]
            # Keep the output in shape of 1D array
            detached_inputs_chunk = (ray_chunk, False, detached_inputs[-1])

            with torch.enable_grad(), torch.cuda.amp.autocast(ctx.had_autocast_in_fwd):
                outputs_chunk: Sequence[torch.Tensor] = ctx.run_function(*detached_inputs_chunk)
                output_rgb_chunk, output_reg_chunk, _, output_expected_depth_chunk, _, _, output_mask_chunk = outputs_chunk

                assert isinstance(output_rgb_chunk, torch.Tensor) and output_rgb_chunk.shape == \
                            partial_loss_partial_image_chunk.shape, f"Please make sure that the `outputs` is " \
                            f"`torch.Tensor` and the shape equals to {partial_loss_partial_image_chunk.shape}, " \
                            f"got {type(output_rgb_chunk)}, {output_rgb_chunk.shape}."
                if not output_rgb_chunk.requires_grad:
                    raise RuntimeError("None of outputs has requires_grad=True, this checkpoint() is not necessary.")

                loss_chunk = (output_rgb_chunk * partial_loss_partial_image_chunk).sum() + \
                             (output_mask_chunk * partial_loss_partial_mask_chunk).sum() + \
                              output_reg_chunk * partial_loss_partial_reg_chunk

                if partial_loss_partial_expected_depth_flattened is not None:
                    loss_chunk = loss_chunk + (output_expected_depth_chunk * partial_loss_partial_expected_depth_chunk).sum()

            loss_chunk.backward()

        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)

        return (None, None) + grads


def checkpoint(function, *args, **kwargs):
    r"""Checkpoint the `nerfacto` model
    """
    backward_chunk_size = kwargs.pop('backward_chunk_size')
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return CheckpointFunction.apply(function, backward_chunk_size, *args)


def update_train_settings(func):
    @functools.wraps(func)
    def wraps(*args, **kwargs):
        self: SceneCraftPipeline = args[0]
        diffusion_step = self.step_buffers["diffusion_step"].value
        nerf_step = self.step_buffers["nerf_step"].value

        extra_kwargs = dict()

        if diffusion_step > self.time_schedule[0][0] / 3:
            self.datamanager._data_mode = "half_standard"
        if diffusion_step > self.time_schedule[0][0]:
            self.datamanager._data_mode = "standard"

        # update diffusion settings
        if self.guidance is not None:
            # loaded pretrained checkpoint
            if self.step_buffers["diffusion_step"].value > kwargs["step"]:
                kwargs["step"] = self.step_buffers["diffusion_step"].value + 1
                diffusion_step = kwargs["step"]
            # anneal the controlnet scale
            if diffusion_step > self.time_schedule[0][0]:
                decay = (diffusion_step - self.time_schedule[0][0]) / (self.time_schedule[3][0] - self.time_schedule[0][0]) * 1.5
                self.guidance.controlnet_conditioning_scale = list(map(lambda x: x - decay, self.init_controlnet_conditioning_scale))

            # update nosie schedule
            schedules = list(filter(lambda x: x[0] >= diffusion_step, self.time_schedule))
            curr_schedule = schedules[0] if len(schedules) > 0 else self.time_schedule[-1]
            is_last = curr_schedule == self.time_schedule[-1]

            if not is_last or diffusion_step > curr_schedule[0]:
                min_timestep_percent = curr_schedule[1]
                max_timestep_percent = curr_schedule[2]
            else:
                # linear descending
                penult_schedule = self.time_schedule[-2]
                distance = (diffusion_step - curr_schedule[0]) / (curr_schedule[0] - penult_schedule[0])
                min_timestep_percent = curr_schedule[1] + distance * (curr_schedule[1] - penult_schedule[1])
                max_timestep_percent = curr_schedule[2] + distance * (curr_schedule[2] - penult_schedule[2])

            self.guidance.min_timestep = int(self.guidance.num_train_timesteps * min_timestep_percent)
            self.guidance.max_timestep = int(self.guidance.num_train_timesteps * max_timestep_percent)
            T = torch.randint(self.min_timestep, self.max_timestep + 1, [1], dtype=torch.long).item()
            extra_kwargs.update(T=T)

        if nerf_step % self.config.full_image_every == 0 or not self.config.async_mode:
            extra_kwargs.update(full_image=True)

        # update nerf settings
        if self.modelB is not None:
            if diffusion_step > self.time_schedule[0][0] or not self.config.async_mode:
                self.modelB.enable_depth_loss = False
            if diffusion_step > self.time_schedule[0][0]:
                decay_mult = max(self.time_schedule[2][0] - diffusion_step, 0.) / (self.time_schedule[2][0] - self.time_schedule[0][0]) * 0.9 + 0.1
                self.modelB.config.latent_loss_mult = self.init_latent_loss_mult * decay_mult

        # update modelA modelB
        if self.modelB is not None and self.modelA is not None:
            if nerf_step % 500 == 0 and nerf_step % 10000 > 1000 and nerf_step != 0:
                state_dictB = self.modelB.state_dict()
                self.modelA.load_state_dict(state_dictB, strict=True)
                for p in self.modelA.parameters():
                    p.requires_grad_(False)
                print("ModelA updated!")
            if nerf_step % 10000 == 0 and nerf_step != 0:
                device = self.device
                self.modelB.load_state_dict(self.init_state, strict=True)
                self.modelB.to(device)
                print("ModelB reinitialized!")

        return func(*args, **kwargs, **extra_kwargs)

    return wraps


@dataclass
class SceneCraftPipelineConfig(LoggableConfig, VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SceneCraftPipeline)
    """target class to instantiate"""

    prompt: str = field(default="")
    """Prompt to be used to generate the scene"""
    async_mode: bool = field(default=False)
    """Whether to use async mode"""
    nerf_single_device: bool = True
    """Whether to use single gpu for NeRF model, prior to `nerf_ranks`. If ture, always use the last rank."""
    nerf_ranks: List[int] = field(default_factory=lambda: [3])
    """GPU ranks of the nerf model"""
    model: SceneCraftModelConfig = SceneCraftModelConfig()
    """specifies the model config"""
    datamanager: SceneCraftDataManagerConfig = SceneCraftDataManagerConfig()
    """specifies the datamanager config"""
    guidance_scale: float = markvar(default=7.5)
    """condition(text) guidance scale for SceneCraft"""
    guidance_depth: bool = markvar(default=False)
    """If use depth estimator to produce depth guidance"""
    controlnet_conditioning_scale: List[float] = markvar(default_factory=lambda: [3.5, 1.5])
    """condition(controlnet) guidance scale for SceneCraft"""
    diffusion_steps: int = markvar(default=20)
    """Number of diffusion steps to take for SceneCraft"""
    scheduler_type: str = markvar(default="unipc")
    """Scheduler type used when decoding noise"""
    time_schedule: List[List[int]] = field(default_factory=lambda: [
        [600, -1, -1], [1000, 0.8, 0.98], [1300, 0.7, 0.8], [1600, 0.6, 0.7], [1900, 0.5, 0.6], [2200, 0.2, 0.5]])
    """time schedule for single guide image"""
    fix_init_noise: bool = markvar(default=False)
    """Whether to use the same initial noise"""
    downscale_factor: float = markvar(default=1.)
    """Minimum downscale factor"""
    guidance_use_full_precision: bool = markvar(default=False)
    """Whether to use fp32 or not"""
    checkpoint_path: Union[str, Path] = markvar(default=Path("outputs/finetune/controlnet/scannetpp/"))
    """Path to the pretrained stable diffusion model"""
    checkpoint_subfolder: str = markvar(default="checkpoint-5000")
    """Specific version of checkpoint"""
    unet_checkpoint_path: Union[str, Path] = markvar(default="")
    """"""
    unet_checkpoint_subfolder: str = markvar(default="")
    """"""
    pretrained_model_name_or_path: Union[str, Path] = markvar(default="runwayml/stable-diffusion-v1-5")
    """Pretrained SD version"""
    full_image_every: int = markvar(default=5)
    """Render full image every steps"""
    tracker_log_steps: int = 10
    """Interval steps between two logs if tracker enabled."""
    enable_modelAB: bool = False


class SceneCraftPipeline(VanillaPipeline):
    """SceneCraftect pipeline"""

    config: SceneCraftPipelineConfig
    _modelA: SceneCraftNeRFModel
    _modelB: SceneCraftNeRFModel
    guidance: StableDiffusionGuidance

    def __init__(
        self,
        config: SceneCraftPipelineConfig,
        device: str,
        is_training: bool = True,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        ranks: Dict[str, list] = dict(),
        events: Dict[str, torch.multiprocessing.Event] = dict(),
        buffers: Dict[str, torch.multiprocessing.Queue] = dict(),
        tracker: Any = None,
    ):
        # NOTE: here skip the `VanillaPipeline.__init__`
        # self has an attribute `device` that cannot be changed.
        Pipeline.__init__(self)

        self.config = config
        self.is_training = is_training
        self.test_mode = test_mode
        self.enable_modelAB = self.config.enable_modelAB
        self.nerf_ranks = ranks.get("nerf_ranks", [0])
        self.guide_ranks = ranks.get("guide_ranks", [])

        self.nerf_trainer = events.get("nerf_trainer", None)
        self.call_renderer = events.get("call_renderer", None)
        self.base_buffers = buffers.get("base_buffers", None)
        self.guide_buffer = buffers.get("guide_buffer", None)
        self.render_indice_buffers = buffers.get("render_indice_buffers", None)
        self.step_buffers = buffers.get("step_buffers", None)
        logger.info(f"Set nerf_ranks = {self.nerf_ranks}, guide_ranks = {self.guide_ranks}")
        self.tracker = tracker

        if not self.is_training:

            if len(self.guide_ranks) != 0:
                raise RuntimeError(f"Guide shouldn't be initialized for non-training runtime.")

        if len(self.nerf_ranks) == 0:
            raise RuntimeError(f"NeRF doesn't have ranks.")

        # setup datamanager
        if not self.config.async_mode:
            self.config.datamanager.train_num_images_to_sample_from = 1
        self.guide_image_chunk_size = len(self.guide_ranks) * self.config.datamanager.train_num_images_to_sample_from
        self.datamanager: SceneCraftDataManager = self.config.datamanager.setup(
            device=device, is_training=is_training, test_mode=self.test_mode, world_size=world_size,
            local_rank=local_rank, nerf_ranks=self.nerf_ranks, guide_ranks=self.guide_ranks, prompt=self.config.prompt,
            guide_image_chunk_size=self.guide_image_chunk_size, downscale_factor=self.config.downscale_factor,
            train_num_rays_per_batch=self.config.model.train_num_rays_per_chunk,
        )
        self.dataset_type = self.datamanager.dataparser.config.dataset_type
        device = torch.device(f'cuda:{int(local_rank)}') if device == 'cuda' else torch.device(device)
        self.datamanager.to(device) # `device` == 'cuda:{id}'
        self.num_train_data = len(self.datamanager.eval_dataset) if not is_training else len(self.datamanager.train_dataset)
        assert self.datamanager.train_dataset is not None or self.datamanager.eval_dataset is not None, "Missing input dataset"

        # setup according to gpu rank
        # NOTE: assume that `num_machines` == 1
        self._run_type = None
        self._modelA = None
        self._modelB = None
        self._guidance = None
        if local_rank in self.nerf_ranks or world_size == 1:
            logger.debug(f"Setting up nerf ...")
            self._setup_nerf(device, world_size, local_rank, grad_scaler)
            self._run_type = "train" if is_training else test_mode
            self.run_name = "nerf"
        if (local_rank in self.guide_ranks or world_size == 1) and is_training:
            logger.debug(f"Setting up guidance ...")
            self._setup_guide(device, world_size, local_rank)
            self._run_type = "inference_with_training"
            self.run_name = "guidance"

        # time schedule buffer for logs
        self.T = None
        self.timesteps = None
        self.min_timestep = -1
        self.max_timestep = -1
        time_schedule = self.config.time_schedule
        self.time_schedule = time_schedule

        self.local_rank = local_rank
        if is_training:
            # the index of local_rank in current ranks group 
            self.process_group_rank = torch.distributed.get_process_group_ranks(comms.LOCAL_PROCESS_GROUP).index(local_rank)

    @check_local_main_thread
    def log(self):
        # Log important attributes
        logger.info("")
        logger.info("***** Control settings *****")
        self.config.log(logger=logger)
        logger.info("***** Data settings *****")
        self.datamanager.dataparser.config.log(logger=logger)
        self.datamanager.config.log(logger=logger)
        logger.info("***** Model settings *****")
        self.config.model.log(logger=logger)
        logger.info(f"  Time schedule: {self.time_schedule}.")

        # check data type
        if self.modelB is not None:
            logger.info(f"  Dtype of model: {self.modelB.device_indicator_param.dtype}.")
        if self.guidance is not None:
            logger.info(f"  Dtype of guidance: {self.guidance.weights_dtype}.")

        logger.info("")

    @check_local_main_thread
    def log_step(self, step, **kwargs):
        step_message = f"{self.run_name} step: {step}"
        if kwargs:
            step_message += f", {', '.join([k + ': ' + str(round(v, 4)) for k, v in kwargs.items()])}"
        logger.info(step_message)

    def _setup_nerf(self, device, world_size, local_rank, grad_scaler):

        input_dataset = self.datamanager.train_dataset or self.datamanager.eval_dataset

        # model
        if self.enable_modelAB:
            self._modelA = self.config.model.setup(
                scene_box=input_dataset.scene_box, num_train_data=self.num_train_data,
                metadata=input_dataset.metadata, device=torch.device("cpu"), grad_scaler=grad_scaler,
            )
            self.modelA.eval()
            self.modelA.to(torch.device("cpu"))
            for p in self.modelA.parameters():
                p.requires_grad_(False)

        self._modelB = self.config.model.setup(
            scene_box=input_dataset.scene_box, num_train_data=self.num_train_data,
            metadata=input_dataset.metadata, device=device, grad_scaler=grad_scaler
        )
        self._modelB._device = device
        self.modelB.to(device)
        self.init_state = self.modelB.state_dict()
        if not self.is_training:
            self.modelB.eval()

        self.world_size = world_size
        if world_size > 1 and len(self.nerf_ranks) > 1:
            self._model = cast(SceneCraftNeRFModel, DDP(self._model, device_ids=[local_rank], process_group=
                                                  comms.LOCAL_PROCESS_GROUP, find_unused_parameters=True))
            # only sync among nerf processes
            torch.distributed.barrier(comms.LOCAL_PROCESS_GROUP)

        self.init_latent_loss_mult = self.modelB.config.latent_loss_mult

    def _setup_guide(self, device, world_size, local_rank):

        self._guidance = StableDiffusionGuidance(
            device, self.dataset_type, guidance_use_full_precision=self.config.guidance_use_full_precision,
            checkpoint_path=self.config.checkpoint_path,
            checkpoint_subfolder=self.config.checkpoint_subfolder,
            unet_checkpoint_path=self.config.unet_checkpoint_path,
            unet_checkpoint_subfolder=self.config.unet_checkpoint_subfolder,
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            condition_type=self.config.datamanager.condition_type,
            guidance_scale=self.config.guidance_scale,
            guidance_depth=self.config.guidance_depth,
            controlnet_conditioning_scale=self.config.controlnet_conditioning_scale,
            num_inference_steps=self.config.diffusion_steps,
            scheduler_type=self.config.scheduler_type,
            fix_init_noise=self.config.fix_init_noise)

        # load base text embedding using classifier free guidance and cache them
        self.prompt_processor = StableDiffusionPromptProcessor(
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
            tokenizer=self.guidance.pipe.tokenizer, device=device,
            text_encoder=self.guidance.pipe.text_encoder,
            do_classifier_free_guidance=(self.config.guidance_scale > 1.),
            prompts=self.datamanager.train_dataset.prompts)

        if world_size > 1 and len(self.guide_ranks) > 1:
            self._guidance = cast(StableDiffusionGuidance, DDP(self._guidance, device_ids=[local_rank],
                                process_group=comms.LOCAL_PROCESS_GROUP, find_unused_parameters=True))
            # only sync among guide processes
            torch.distributed.barrier(comms.LOCAL_PROCESS_GROUP)

        if self.config.fix_init_noise:
            width, height = self.datamanager.train_dataset.scale_image_size
            batch_size = self.datamanager.config.train_num_images_to_sample_from
            self.init_latents = self.guidance.prepare_latents(batch_size, height, width, dtype=self.guidance.weights_dtype)

        self.init_controlnet_conditioning_scale = self.config.controlnet_conditioning_scale

    def to(self, dtype: torch.dtype):
        if self.modelB is not None:
            self.modelB.to(dtype)
        return self

    @property
    def modelA(self) -> SceneCraftNeRFModel:
        return module_wrapper(self._modelA)

    @property
    def modelB(self) -> SceneCraftNeRFModel:
        return module_wrapper(self._modelB)

    @property
    def model(self) -> SceneCraftNeRFModel:
        return module_wrapper(self._modelB)

    @property
    def guidance(self) -> StableDiffusionGuidance:
        return module_wrapper(self._guidance)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.modelB.device if self.modelB is not None else self.guidance.device

    def _get_nerf_train_loss_dict(self, step: int, **kwargs):
        self.modelB.train()

        # run renderer once `guidance_pipeline` calls
        while (not self.nerf_trainer.is_set()) or step > 0:
            if self.call_renderer.is_set() or not self.config.async_mode:
                logger.debug("Offline renderer called by guidance pipeline.")
                with torch.no_grad(), time_function(f"Offline {len(self.guide_ranks)} renderer called by guidence"):
                    for i in range(len(self.guide_ranks)):
                        batch_idx = self.render_indice_buffers[i].get()
                        logger.debug(f"Renerer of guide {i} for image idx={batch_idx}.")
                        batch_output_rgb = []
                        for idx in batch_idx:
                            eval_outputs = self._get_nerf_inference(step, image_idx=int(idx))
                            batch_output_rgb.append(eval_outputs["rgb"])
                        batch_output_rgb = torch.stack(batch_output_rgb) # will add the batch dimension
                        self.base_buffers[i].put(batch_output_rgb)
                if not self.config.async_mode:
                    assert len(self.guide_ranks) == 1 and len(batch_idx) == 1
                    kwargs.update(image_idx=int(batch_idx[0]))
                self.call_renderer.clear()
            if step > 0: break

        if step == 0 and kwargs.get("mini_step", 0) == 0 and self.config.async_mode:
            logger.info(f"Nerf training begins exactly.")

        # update `guide_images` at each iteration (on CPU)
        # FIXME: if `train_num_images_to_sample_from` is too large (e.g. 8), conflicts occur
        # when different processes access the buffer, causing the training process to block.
        with time_function("Sync guide images from buffer"):
            while not self.guide_buffer.empty() and self.config.async_mode:
                guide_object = self.guide_buffer.get()
                self.datamanager.train_dataset.put_guide_object(guide_object, step)
                logger.debug(f"Put guide image idx={guide_object['idx']}")
                logger.debug(f"Current all guide image indices: {self.datamanager.train_dataset.all_guide_indices}")
        self.step_buffers["nerf_step"].value = int(step)

        # NeRF training itself (`batch` is not exactly batched)
        ray_bundle, batch = self.datamanager.next_train(step, **kwargs)
        if not self.config.async_mode:
            guide_object = self.guide_buffer.get()
            guide_image = guide_object["image"]
            guide_depth = guide_object["depth"]
            if guide_image.shape[-2:] != batch["image_size"]:
                guide_image = F.interpolate(guide_image, batch["image_size"], mode="bilinear")
            if guide_depth[0] is not None and guide_depth.shape[-2:] != batch["image_size"]:
                guide_depth = F.interpolate(guide_depth, batch["image_size"], mode="nearest")
            batch.update(guide_image=guide_image, guide_depth=guide_depth)

        # Train pixels
        if not kwargs.get("full_image", False):
            rgb, weights_list, ray_samples_list, _, expected_depth, accumulation = self._modelB(ray_bundle)
            model_outputs = dict(rgb=rgb, weights_list=weights_list, ray_samples_list=ray_samples_list,
                                 depth=expected_depth, accumulation=accumulation)
            loss_dict = self.modelB.get_loss_dict(model_outputs, batch)

            self.log_step(step, **{k: v.detach().item() for k, v in loss_dict.items()})
            if os.getenv("RECORD", False) and step % 50 == 0:
                eval_outputs = self._get_nerf_inference(step)
                self.log_image(step, self.datamanager.image_size,
                               dict(rendered_image=Image.fromarray((eval_outputs["rgb"].cpu().numpy() * 255.
                                                                    ).astype(np.uint8)).convert("RGB"),
                                       depth_image=Image.fromarray(colorize_depth((eval_outputs["depth"][..., 0] / eval_outputs["scale_factor"]).cpu().numpy(), 0.1, 3.,
                                                                    ).astype(np.uint8)).convert("RGB")),
                                    texts=[f"step={step}", f"id={eval_outputs['image_id']}"],
                                    locs=[(10, 10), (10, 20)])

            return model_outputs, loss_dict, {}

        # The image size maybe resized in `InputDataset` class by `camera_res_scale_factor` in DataManager.
        # Here uses `self._model` instead of `self.model`, see https://github.com/nerfstudio-project/nerfstudio/pull/1856.
        # And use customed `checkpoint` to save memory, here need to input a dummy_tensor with `requires_grad`= True to
        # make sure that the gradients will be computed during the backward process, refer to
        # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/10.
        dummy_tensor = torch.tensor([1], dtype=torch.float32, device=self.device, requires_grad=True)
        with time_function("Get NeRF Outputs for Ray Bundle"):
            output_rgb, output_regs, output_depth, output_expected_depth, _, _, output_weights_mask = checkpoint(
                                            self._modelB.get_outputs_for_camera_ray_bundle, ray_bundle, True,
                                            dummy_tensor, backward_chunk_size=self.config.model.train_num_rays_per_chunk)
        model_outputs = dict(rgb=output_rgb, depth=output_expected_depth, weights_mask=output_weights_mask)

        # add the batch_size dimension
        model_outputs["rgb"] = model_outputs["rgb"][None, ...]
        with time_function("Get NeRF Loss"):
            loss_dict = self.modelB.get_loss_dict(model_outputs, batch)
            loss_dict["loss_reg"] = output_regs.sum()

        # log something
        self.log_step(step, **{k: v.detach().item() for k, v in loss_dict.items()})
        if os.getenv("RECORD", False):
            self.log_image(step, self.datamanager.image_size,
                        dict(rendered_image=Image.fromarray((output_rgb.detach().cpu().numpy() * 255.
                                                ).astype(np.uint8)).convert("RGB"),
                                guide_image=Image.fromarray((batch["guide_image"][0].cpu().permute(1, 2, 0
                                                ).numpy() * 255.).astype(np.uint8)).convert("RGB"),
                                depth_image=Image.fromarray(colorize_depth((output_depth[..., 0] / batch["scale_factor"]).detach().cpu().numpy(), 0.1, 3.,
                                                ).astype(np.uint8)).convert("RGB") if isinstance(output_depth, torch.Tensor) else None,
                                reference_depth_image=Image.fromarray(colorize_depth(batch["reference_depth"].numpy(), 0.1, 3.
                                                ).astype(np.uint8)).convert("RGB") if "reference_depth" in batch else None,
                                guide_depth_image=Image.fromarray(colorize_depth(batch["guide_depth"][0][0].numpy(), 0.1, 3.
                                                ).astype(np.uint8)).convert("RGB") if batch["guide_depth"][0] is not None else None),
                        texts=[f"step={step}", f"id={batch['image_id']}"],
                        locs=[(10, 10), (10, 20)])

        # force `metrics_dict` to be empty
        return model_outputs, loss_dict, {}

    @torch.no_grad()
    def _get_nerf_inference(self, step: int, **kwargs):
        is_training = self.modelB.training
        if self.enable_modelAB:
            self.modelA.to(self.device).eval()
            model = self.modelA
        else:
            self.modelB.eval()
            model = self.modelB

        ray_bundle, batch = self.datamanager.next_eval(step, **kwargs)
        outputs_ray_bundle = model.get_outputs_for_camera_ray_bundle(ray_bundle)
        outputs = dict(**outputs_ray_bundle, **batch)

        if self.enable_modelAB:
            self.modelA.to(torch.device("cpu"))
        if is_training:
            self.modelB.train()

        return outputs

    @torch.no_grad()
    def _get_guidance_inference(self, step: int, **kwargs):
        batch_base_image = None
        batch = self.datamanager.next_train(step)
        batch_image_idx = batch["image_idx"].tolist()
        logger.debug(f"Guide idx={batch_image_idx}.")

        # nerf trainer
        if not self.config.async_mode and self.process_group_rank == 0 and not self.nerf_trainer.is_set():
            self.nerf_trainer.set()

        # call renderer
        if self.datamanager._data_mode == "standard" or not self.config.async_mode:
            self.render_indice_buffers[self.process_group_rank].put(batch_image_idx)
            # sync to make sure that all indices have been put into buffer
            torch.distributed.barrier(comms.LOCAL_PROCESS_GROUP)
            if self.process_group_rank == 0:
                self.call_renderer.set()

            # wait the renderer and get base image from `base_buffers`
            with time_function(f"Guide {self.process_group_rank}: Get base image from NeRF renderer"):
                batch_base_image = self.base_buffers[self.process_group_rank].get()
        torch.distributed.barrier(comms.LOCAL_PROCESS_GROUP)
        if self.process_group_rank == 0:
            self.step_buffers["diffusion_step"].value = int(step)

        # The image size maybe resized in `InputDataset` class by camera_res_scale_factor in DataManager.
        if self.datamanager.config.condition_type == "rgb":
            batch_source_image = batch["image"].permute(0, 3, 1, 2)
        else:
            batch_source_image = batch["indice_image"]
        if "multi_control" in self.config.checkpoint_path:
            batch_depth_image = batch["depth_image"]
            kwargs.update(cond_depth=batch_depth_image[:, None])

        # check base images from nerf
        if batch_base_image is not None:
            batch_base_image = batch_base_image.permute(0, 3, 1, 2).to(self.device)
            if batch_base_image.shape[-2:] != batch_source_image.shape[-2:]:
                batch_base_image = F.interpolate(batch_base_image, batch_source_image.shape[-2:], mode="bilinear")

        text_embeddings = self.prompt_processor(batch["prompt"])
        if self.config.fix_init_noise:
            kwargs.update(latents_noisy=self.init_latents)
        with time_function(f"Guide {self.process_group_rank}: Process one time guidance"):
            guidance_outputs = self.guidance(cond_image=batch_source_image,
                                             text_embeddings=text_embeddings,
                                             image=batch_base_image,
                                             return_denoised_image=os.getenv("RECORD", False),
                                             **kwargs # T
                                )

        # give a prompt of changed time schedule
        if guidance_outputs.T != self.T:
            self.T = guidance_outputs.T
            self.timesteps = self.guidance.pipe.scheduler.timesteps.tolist()
            logger.debug(f"Step = {step}, set T = {self.T}, timesteps = {self.timesteps}.")
        if self.guidance.min_timestep != self.min_timestep or self.guidance.max_timestep != self.max_timestep:
            self.min_timestep = self.guidance.min_timestep
            self.max_timestep = self.guidance.max_timestep
            logger.info(f"Step = {step}, set min_timestep = {self.min_timestep}, max_timestep = {self.max_timestep}.")
        if step >= self.config.time_schedule[-1][0]:
            logger.warning(f"Current step {step} has exceeds time schedule.")

        # scale the guide image to align with base image
        batch_guide_image = guidance_outputs.images_pt.detach().cpu().float()
        batch_guide_depth = guidance_outputs.depths_pt.detach().cpu().float() \
                            if guidance_outputs.depths_pt is not None else [None] * batch_guide_image.shape[0]
        if batch_guide_image.shape[-2:][::-1] != self.datamanager.train_dataset.scale_image_size:
            new_width, new_height = self.datamanager.train_dataset.scale_image_size
            batch_guide_image = F.interpolate(batch_guide_image, (new_height, new_width), mode="bilinear")
            if batch_guide_depth[0] is not None:
                batch_guide_depth = F.interpolate(batch_guide_depth, (new_height, new_width), mode="bicubic")
        # put the guide images into `guide_buffer`
        with time_function(f"Guide {self.process_group_rank}: Feed guide image into buffer"):
            for i, idx in enumerate(batch_image_idx):
                guide_object = dict(image=batch_guide_image[i:i+1], depth=batch_guide_depth[i:i+1], idx=int(idx))
                self.guide_buffer.put(guide_object)
            if (
                (self.config.async_mode and self.guide_buffer.qsize() >= self.guide_image_chunk_size * 2) or \
                (not self.config.async_mode and not self.guide_buffer.empty())
            ) and self.process_group_rank == 0 and not self.nerf_trainer.is_set():
                self.nerf_trainer.set()

        # log something
        self.log_step(step)
        if os.getenv("RECORD", False):
            self.log_image(step, self.datamanager.image_size,
                           dict(source_image=Image.fromarray((batch["image"][0].detach().cpu().numpy() * 255.
                                                            ).astype(np.uint8)).convert("RGB"),
                                rendered_image = Image.fromarray((batch_base_image[0].detach().cpu().permute(1, 2, 0
                                                                ).numpy() * 255.).astype(np.uint8)).convert("RGB") \
                                                                if batch_base_image is not None else None,
                                guide_image=guidance_outputs.images_pil[0],
                                target_image = Image.fromarray((batch["target_image"][0].detach().cpu().numpy() * 255.
                                                                ).astype(np.uint8)).convert("RGB")
                                                                if "target_image" in batch else None,
                                guide_depth_image = Image.fromarray(colorize_depth(batch_guide_depth[0][0].numpy(), 0.1, 3.
                                                                ).astype(np.uint8)).convert("RGB")
                                                                if batch_guide_depth[0] is not None else None),
                                texts=[f"T={guidance_outputs.T}", f"step={step}", f"id={batch['image_id'][0]}"],
                                locs=[(10, 10), (10, 20), (10, 30)],
                                prompt=batch["prompt"][0])

    @update_train_settings
    def get_train_loss_dict(self, step: int, **kwargs):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.modelB is not None:
            return self._get_nerf_train_loss_dict(step, **kwargs)
        elif self.guidance is not None:
            return self._get_guidance_inference(step, **kwargs)

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.modelA(ray_bundle)
        metrics_dict = self.modelA.get_metrics_dict(model_outputs, batch)
        loss_dict = self.modelA.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()

        # Avoid error when use writer
        if metrics_dict is None:
            metrics_dict = {}
        return model_outputs, loss_dict, metrics_dict

    @torch.no_grad()
    @profiler.time_function
    def get_inference(self, step: int=None) -> None:
        self.eval()

        if self.modelA is not None or self.modelB is not None:
            return self._get_nerf_inference(step)
        elif self.guidance is not None:
            return self._get_guidance_inference(step)

    @check_local_main_thread
    def log_image(self, step: int, image_size: list, images: Dict[str, Image.Image], **kwargs):
        log_dir = kwargs.get("log_dir", None)
        
        image_width, image_height = image_size
        image_count = 0
        merged_image = Image.new("RGB", (image_width * len(images), image_height))

        source_image = images.get("source_image", None)
        if source_image is not None:
            image_count += 1
            merged_image.paste(source_image, (image_count * (image_count - 1), 0))

        rendered_image = images.get("rendered_image", None) or Image.new("RGB", image_size, color="white")
        if rendered_image.size != image_size:
            rendered_image = rendered_image.resize(image_size, resample=Image.Resampling.BILINEAR)
        rendered_image = draw_text_on_image(rendered_image, texts=["rendered image"], locs=[(10, image_count * 10)])
        image_count += 1
        merged_image.paste(rendered_image, (image_width * (image_count - 1), 0))

        guide_image = images.get("guide_image", None)
        if guide_image is not None:
            if guide_image.size != image_size:
                guide_image = guide_image.resize(image_size, resample=Image.Resampling.BILINEAR)
            guide_image = draw_text_on_image(guide_image, texts=["guide image"], locs=[(10, 10)])
            image_count += 1
            merged_image.paste(guide_image, (image_width * (image_count - 1), 0))

        target_image = images.get("target_image", None)
        if target_image is not None:
            image_count += 1
            if target_image.size != image_size:
                target_image = target_image.resize(image_size, resample=Image.Resampling.BILINEAR)
            target_image = draw_text_on_image(target_image, texts=["reference image"], locs=[(10, 10)])
            merged_image.paste(target_image, ((image_width * (image_count - 1), 0)))

        depth_image = images.get("depth_image", None)
        if depth_image is not None:
            image_count += 1
            if depth_image.size != image_size:
                depth_image = depth_image.resize(image_size, resample=Image.Resampling.NEAREST)
            merged_image.paste(depth_image, ((image_width * (image_count - 1), 0)))

        reference_depth_image = images.get("reference_depth_image", None)
        if reference_depth_image is not None:
            image_count += 1
            if reference_depth_image.size != image_size:
                reference_depth_image = reference_depth_image.resize(image_size, resample=Image.Resampling.NEAREST)
            merged_image.paste(reference_depth_image, ((image_width * (image_count - 1), 0)))

        guide_depth_image = images.get("guide_depth_image", None)
        if guide_depth_image is not None:
            image_count += 1
            if guide_depth_image.size != image_size:
                guide_depth_image = guide_depth_image.resize(image_size, resample=Image.Resampling.NEAREST)
            merged_image.paste(guide_depth_image, ((image_width * (image_count - 1), 0)))

        if "texts" in kwargs:
            merged_image = draw_text_on_image(merged_image, texts=kwargs["texts"], locs=kwargs["locs"])

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            merged_image.save(f"{log_dir}/image_{step:06d}.png")

        # log images to tracker
        if step % (self.config.tracker_log_steps // self.config.datamanager.train_num_images_to_sample_from) == 0 and self.tracker:
            formatted_images = [wandb.Image(merged_image, caption=kwargs.get("prompt", ""))]
            self.tracker.log({"training": formatted_images}, step=step)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.modelB.get_param_groups() if self.modelB is not None else dict()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.modelB.get_training_callbacks(training_callback_attributes) if self.modelB is not None else list()
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def export_meta_infos(self) -> dict:

        return {
            "data_mode": self.datamanager._data_mode,
            "diffusion_step": self.step_buffers["diffusion_step"].value,
            "nerf_step": self.step_buffers["nerf_step"].value,
            "nerf_guide_buffers": self.datamanager.train_dataset.guide_images
        }

    def load_meta_infos(self, meta_infos: dict):
        self.datamanager._data_mode = meta_infos["data_mode"]
        self.step_buffers["diffusion_step"].value = int(meta_infos["diffusion_step"] + 1)
        self.step_buffers["nerf_step"].value = int(meta_infos["nerf_step"] + 1)
        CONSOLE.print(f"Set data mode to {self.datamanager._data_mode}")
        CONSOLE.print(f"Set diffusion step to {self.step_buffers['diffusion_step'].value}")
        CONSOLE.print(f"Set nerf step to {self.step_buffers['nerf_step'].value}")
        if hasattr(self.datamanager.train_dataset, "guide_images"):
            self.datamanager.train_dataset.guide_images = meta_infos["nerf_guide_buffers"]
            CONSOLE.print(f"Loaded {len(self.datamanager.train_dataset.guide_images)} guide image buffer")

    def load_pipeline(self, loaded_state: Dict[str, Any], meta_infos: dict, step: int, strict: Optional[bool] = None) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)

        # load meta info
        if self.is_training:
            self.load_meta_infos(meta_infos)

        # load state dict
        is_ddp_model_state = True
        model_state = {}
        for key, value in state.items():
            if key.startswith("_modelB."):
                # remove the "_model." prefix from key
                model_state[key[len("_modelB.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state.items() if not key.startswith("_model.")}

        try:
            self.model.load_state_dict(model_state, strict=True)
            if self.modelA is not None:
                self.modelA.load_state_dict(model_state, strict=True)
        except RuntimeError:
            if not strict:
                self.model.load_state_dict(model_state, strict=False)
                if self.modelA is not None:
                    self.modelA.load_state_dict(model_state, strict=False)
            else:
                raise

        super().load_state_dict(pipeline_state, strict=False)
