import time
import os
import torch
import dataclasses
import functools
import yaml
from pathlib import Path
from typing import Literal, Type, cast, Dict
from dataclasses import dataclass, field
from threading import Lock
from accelerate.tracking import WandBTracker

from nerfstudio.utils import profiler, writer
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.decorators import check_viewer_enabled
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.engine.trainer import Trainer, TrainerConfig, TRAIN_INTERATION_OUTPUT
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager

from scenecraft.utils import time_function, setup_profiler, is_local_main_process, check_nerf_process
from scenecraft.pipeline import SceneCraftPipeline, SceneCraftPipelineConfig, logger


""" Implementation of trainer """

@dataclass
class SceneCraftTrainerConfig(TrainerConfig):
    """Configuration for the InstructNeRF2NeRFTrainer."""
    _target: Type = field(default_factory=lambda: SceneCraftTrainer)

    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    steps_per_save: int = 100
    """Number of steps between saves."""
    pipeline: SceneCraftPipelineConfig = SceneCraftPipelineConfig()
    """Pipeline configuration for scenecraft."""
    save_only_latest_checkpoint: bool = False
    """Whether to only save the latest checkpoint or all checkpoints."""

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        dataset_type = self.pipeline.datamanager.dataparser.dataset_type
        scene_id = self.pipeline.datamanager.dataparser.scene_id
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.timestamp}-{dataset_type}-{scene_id}")

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self, sort_keys=False), "utf8")


class SceneCraftTrainer(Trainer):
    """Trainer for SceneCraft"""

    config: SceneCraftTrainerConfig
    pipeline: SceneCraftPipeline

    def __init__(self, config: SceneCraftTrainerConfig, local_rank: int = 0,
                 world_size: int = 1, ranks: Dict[str, list] = None,
                 events: Dict[str, torch.multiprocessing.Event] = None,
                 buffers: Dict[str, torch.multiprocessing.Queue] = None,
        ) -> None:
        # experiment_name = ...
        # self.config.experiment_name = experiment_name
        super().__init__(config, local_rank, world_size)
        self.nerf_lock = Lock()
        self.guidance_lock = Lock()
        self.async_mode = self.config.machine.num_devices > 1
        self.ranks = ranks
        self.events = events
        self.buffers = buffers
        # wandb tacker
        self.tracker = None
        if is_local_main_process() and os.getenv("RECORD", default=False):
            self.tracker = WandBTracker("instruct-nerf")

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline: SceneCraftPipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            ranks=self.ranks,
            events=self.events,
            buffers=self.buffers,
            tracker=self.tracker,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        self.viewer_state, banner_messages = None, None
        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        setup_profiler(self.config.logging, writer_log_path)

        # logging
        self.pipeline.log()

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state="training",
            eval_dataset=self.pipeline.datamanager.eval_dataset,
        )

    def run(self) -> None:
        if not hasattr(self, self.pipeline._run_type):
            raise ValueError(f"Unknown run type of pipeline, expected one of [train, inference_with_training], got {self.pipeline._run_type}.")
        getattr(self, self.pipeline._run_type)()

    def inference_with_training(self) -> None:
        """Inference the model, refer to `self.eval_iteration`."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatasetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.guidance_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.eval()

                        # inference iteration
                        self.pipeline.get_train_loss_dict(step=step)


    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        assert (
            self.gradient_accumulation_steps > 0
        ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps}"

        # True training process begins after taking one wait
        for i in range(self.gradient_accumulation_steps):
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step, mini_step=i)
                loss = functools.reduce(torch.add, loss_dict.values())
                loss /= self.gradient_accumulation_steps
            with time_function("Loss backward"):
                self.grad_scaler.scale(loss).backward()  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        if self.tracker is not None:
            self.tracker.log(loss_dict, step=step)
            proposal_grads = [param.grad.flatten() for param in self.pipeline.modelB.proposal_networks.parameters()
                                                                                        if param.grad is not None]
            proposal_grad = torch.norm(torch.cat(proposal_grads)) if proposal_grads else torch.Tensor([0.])
            field_grads = [param.grad.flatten() for param in self.pipeline.modelB.field.parameters()
                                                                                        if param.grad is not None]
            field_grad = torch.norm(torch.cat(field_grads)) if field_grads else torch.Tensor([0.])

            lr = dict()
            for param_group_name, scheduler in self.optimizers.schedulers.items():
                lr[f"lr/{param_group_name}"]=scheduler.get_last_lr()[0]

            self.tracker.log({"proposal_grad": proposal_grad.item(), "field_grad": field_grad.item(), **lr})

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.modelB.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale() and self.pipeline.datamanager._data_mode == "standard":
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_nerf_process
    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None or load_checkpoint is not None:
            if load_dir is not None:
                load_step = self.config.load_step
                if load_step is None:
                    print("Loading latest Nerfstudio checkpoint from load_dir...")
                    # NOTE: this is specific to the checkpoint name format
                    load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
                load_checkpoint: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["meta_infos"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    @check_nerf_process
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        pipeline_state_dict = self.pipeline.state_dict()
        pipeline_mete_infos = self.pipeline.export_meta_infos()
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "meta_infos": pipeline_mete_infos,
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint to {ckpt_path}")
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()