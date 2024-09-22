import os
import argparse
import yaml
import torch
import torch.distributed

from sat import mpu
from typing import Dict
from omegaconf import OmegaConf
from functools import partial
from arguments import get_args
from einops import rearrange
from sat.training.deepspeed_training import training_main
from sgm.util import get_obj_from_str

from diffusion_video import SATVideoDiffusionEngine
from logging_utils import log_video

try:
    import wandb
except ImportError:
    print("warning: wandb not installed")


def broad_cast_batch(batch):
    mp_size = mpu.get_model_parallel_world_size()
    global_rank = torch.distributed.get_rank() // mp_size
    src = global_rank * mp_size

    if batch["mp4"] is not None:
        broadcast_shape = [
            batch["mp4"].shape,
            batch["fps"].shape,
            batch["num_frames"].shape,
        ]
    else:
        broadcast_shape = None

    txt = [batch["txt"], broadcast_shape]
    torch.distributed.broadcast_object_list(
        txt, src=src, group=mpu.get_model_parallel_group()
    )
    batch["txt"] = txt[0]

    mp4_shape = txt[1][0]
    fps_shape = txt[1][1]
    num_frames_shape = txt[1][2]

    if mpu.get_model_parallel_rank() != 0:
        batch["mp4"] = torch.zeros(mp4_shape, device="cuda")
        batch["fps"] = torch.zeros(fps_shape, device="cuda", dtype=torch.long)
        batch["num_frames"] = torch.zeros(
            num_frames_shape, device="cuda", dtype=torch.long
        )

    torch.distributed.broadcast(
        batch["mp4"], src=src, group=mpu.get_model_parallel_group()
    )
    torch.distributed.broadcast(
        batch["fps"], src=src, group=mpu.get_model_parallel_group()
    )
    torch.distributed.broadcast(
        batch["num_frames"], src=src, group=mpu.get_model_parallel_group()
    )
    return batch


def forward_step_eval(
    data_iterator, model, args, timers, only_log_video_latents=False, data_class=None
):
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()
        batch_video = next(data_iterator)
        timers("data loader").stop()

        if len(batch_video["mp4"].shape) == 6:
            b, v = batch_video["mp4"].shape[:2]
            batch_video["mp4"] = batch_video["mp4"].view(
                -1, *batch_video["mp4"].shape[2:]
            )
            txt = []
            for i in range(b):
                for j in range(v):
                    txt.append(batch_video["txt"][j][i])
            batch_video["txt"] = txt

        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()
    else:
        batch_video = {"mp4": None, "fps": None, "num_frames": None, "txt": None}
    broad_cast_batch(batch_video)
    if mpu.get_data_parallel_rank() == 0:
        log_video(
            batch_video, model, args, only_log_video_latents=only_log_video_latents
        )

    batch_video["global_step"] = args.iteration
    loss, loss_dict = model.shared_step(batch_video)
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)
    return loss, loss_dict


def forward_step(data_iterator, model, args, timers, data_class=None):
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()
        batch = next(data_iterator)
        timers("data loader").stop()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        if torch.distributed.get_rank() == 0:
            if not os.path.exists(os.path.join(args.save, "training_config.yaml")):
                configs = [OmegaConf.load(cfg) for cfg in args.base]
                config = OmegaConf.merge(*configs)
                os.makedirs(args.save, exist_ok=True)
                OmegaConf.save(
                    config=config, f=os.path.join(args.save, "training_config.yaml")
                )
    else:
        batch = {"mp4": None, "fps": None, "num_frames": None, "txt": None}

    batch["global_step"] = args.iteration

    broad_cast_batch(batch)

    loss, loss_dict = model.shared_step(batch)

    return loss, loss_dict


if __name__ == "__main__":

    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    data_class = get_obj_from_str(args.data_config["target"])
    create_dataset_function = partial(
        data_class.create_dataset_function, **args.data_config["params"]
    )

    # merge configs
    configs = []
    for config in args.base:
        with open(config, "r") as f:
            base_config = yaml.safe_load(f)
        configs.append(base_config)
    args.log_config = configs

    training_main(
        args,
        model_cls=SATVideoDiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class),
        forward_step_eval=partial(
            forward_step_eval,
            data_class=data_class,
            only_log_video_latents=args.only_log_video_latents,
        ),
        create_dataset_function=create_dataset_function,
    )
