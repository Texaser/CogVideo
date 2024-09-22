import os
import imageio
import wandb
import torch
import torch.distributed
import numpy as np

from typing import Dict
from einops import rearrange
from sgm.util import isheatmap


def print_debug(args, s):
    if args.debug:
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        print(s)


def save_texts(texts, save_dir, iterations):
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5, args=None, key=None
):

    os.makedirs(save_path, exist_ok=True)
    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)
        if args is not None and args.wandb:
            wandb.log(
                {
                    key
                    + f"_video_{i}": wandb.Video(now_save_path, fps=fps, format="mp4")
                },
                step=args.iteration + 1,
            )


def log_video(batch: Dict, model, args, only_log_video_latents=False):

    texts = batch["txt"]
    text_save_dir = os.path.join(args.save, "video_texts")

    # save text prompts to output dir
    os.makedirs(text_save_dir, exist_ok=True)
    save_texts(texts, text_save_dir, args.iteration)

    breakpoint()
    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    # i'm not touching the depreciated autocast (:
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)

    if not torch.distributed.get_rank() == 0:
        return

    root = os.path.join(args.save, "video")

    if only_log_video_latents:
        root = os.path.join(root, "latents")
        filename = "{}_gs-{:06}".format("latents", args.iteration)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        os.makedirs(path, exist_ok=True)
        torch.save(videos["latents"], os.path.join(path, "latent.pt"))
    else:
        for k in videos:
            # copy vids -> CPU as needed
            N = videos[k].shape[0]
            if not isheatmap(videos[k]):
                videos[k] = videos[k][:N]
            if isinstance(videos[k], torch.Tensor):
                videos[k] = videos[k].detach().float().cpu()
                if not isheatmap(videos[k]):
                    videos[k] = torch.clamp(videos[k], -1.0, 1.0)

        num_frames = batch["num_frames"][0]
        fps = batch["fps"][0].cpu().item()
        if only_log_video_latents:
            root = os.path.join(root, "latents")
            filename = "{}_gs-{:06}".format("latents", args.iteration)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            os.makedirs(path, exist_ok=True)
            torch.save(videos["latents"], os.path.join(path, "latents.pt"))
        else:
            for k in videos:
                # ???, some normalization thing?
                samples = (videos[k] + 1.0) / 2.0
                filename = "{}_gs-{:06}".format(k, args.iteration)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                save_video_as_grid_and_mp4(
                    samples, path, num_frames // fps, fps, args, k
                )
