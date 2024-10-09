from __future__ import annotations

import torch
import random
import socket
import traceback
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from datetime import timedelta
from typing import Any, Callable, Literal, Optional, Dict

from nerfstudio.utils import profiler, comms
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion

from scenecraft.trainer import SceneCraftTrainerConfig
from scenecraft.utils import seed_everything


""" Implementation of train entripoint """

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: SceneCraftTrainerConfig, global_rank: int = 0, ranks: Dict[str, list] = None,
               events: Dict[str, mp.Event] = None, buffers: Dict[str, mp.Queue] = None):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size, ranks=ranks,
                           events=events, buffers=buffers)
    trainer.setup()
    trainer.run()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    nerf_ranks: list,
    guide_ranks: list,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: SceneCraftTrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
    ranks: Dict[str, list] = None,
    events: Dict[str, mp.Event] = None,
    buffers: Dict[str, mp.Queue] = None,
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    # NOTE: here we force the single machine !!!
    # and `ranks` is for each ranks_group, e.g. `nerf_ranks`, `guide_ranks`.
    if local_rank in nerf_ranks:
        group_ranks = nerf_ranks
    elif local_rank in guide_ranks:
        group_ranks = guide_ranks
    else:
        raise RuntimeError(f"Local rank = {local_rank} not found on current machine, check `num_devices_per_machine`!!")

    comms.LOCAL_PROCESS_GROUP = dist.new_group(group_ranks)

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank, ranks, events, buffers)

    comms.synchronize()
    dist.destroy_process_group()

    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[SceneCraftTrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        raise RuntimeError(f"Only support multi-gpu training.")
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        # NOTE: determin the group ranks
        all_ranks = [i for i in range(num_devices_per_machine)] # all processors
        if not config.pipeline.async_mode and len(all_ranks) > 2:
            raise RuntimeError(f"Only support less than or equal to 2 ranks totally when using sync mode.")
        nerf_ranks = set(config.pipeline.nerf_ranks) if not config.pipeline.nerf_single_device else set([all_ranks[-1]])
        guide_ranks = set(all_ranks) - nerf_ranks
        nerf_ranks = set(all_ranks) - guide_ranks
        if len(nerf_ranks) == 0:
            raise RuntimeError(f"Nerf Model has no ranks! Check your devices and settings of `nerf_ranks` in `pipeline` and `torch.cuda.device_count()`.")
        if len(guide_ranks) == 0:
            raise RuntimeError(f"Guide pipeline has no ranks! Check your devices and settings of `nerf_ranks` in `pipeline` and `torch.cuda.device.count()`.")
        ranks = dict(nerf_ranks=nerf_ranks, guide_ranks=guide_ranks)

        # create communication tools
        nerf_trainer = mp.get_context("spawn").Event()
        call_renderer = mp.get_context("spawn").Event()
        base_buffers = [mp.get_context("spawn").Queue() for _ in range(len(guide_ranks))]
        guide_buffer = mp.get_context("spawn").Queue()
        render_indice_buffers = [mp.get_context("spawn").Queue() for _ in range(len(guide_ranks))]
        step_buffers = dict(nerf_step=mp.get_context("spawn").Value("i"), diffusion_step=mp.get_context("spawn").Value("i"))
        events = dict(nerf_trainer=nerf_trainer, call_renderer=call_renderer)
        buffers = dict(base_buffers=base_buffers, guide_buffer=guide_buffer, render_indice_buffers=render_indice_buffers, step_buffers=step_buffers)
        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, list(nerf_ranks), list(guide_ranks), num_devices_per_machine,
                  machine_rank, dist_url, config, timeout, device_type, ranks, events, buffers),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: SceneCraftTrainerConfig) -> None:
    """Main function."""

    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.prompt:
        CONSOLE.log("Using --prompt alias for --data.pipeline.model.prompt")
        config.pipeline.model.prompt = config.prompt

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    config.set_timestamp()

    # print and save config
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    seed_everything(2024)
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
