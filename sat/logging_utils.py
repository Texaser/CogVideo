import os
import imageio
import wandb
import torch
import torch.distributed
import numpy as np
import cv2

from typing import Dict, Optional
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


def draw_bbxs_on_frame(frame: np.ndarray, bbxs: np.ndarray) -> np.ndarray:
    """
    Draw bbx on frame for visualization purposes.
    
    TODO: we are only logging first-frame conditions atm.
    """
    
    h, w = 480, 720
    for bbx in bbxs:
        breakpoint()
        x1_norm, y1_norm, x2_norm, y2_norm = bbx
        # denormalize coordinates to pixel values
        x1 = int(x1_norm * w)
        x2 = int(x2_norm * w)
        y1 = int(y1_norm * h)
        y2 = int(y2_norm * h)
        # ensure coordinates are within image dimensions
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        # ensure top-left and bottom-right ordering
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        try:
            # draw the rectangle on the frame
            # "god & gary bradski knows why...""
            # we must use a copy of the original frame, otherwise this breaks
            # https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
            frame = cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), color=(0, 255, 0),thickness=3)
        except Exception as e:
            print(f"Error adding bbx to frame: {e}")
            
    return frame


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor,
    save_path: str,
    T: int,
    fps: int = 5,
    args=None,
    key=None,
    bbxs: Optional[torch.Tensor] = None,
):
    """
    Save a batch of video tensors `video_batch` to `save_path` as MP4s.

    Args:
        bbxs (torch.Tensor): Optional tensor containing bboxs [B, T, 10, 4]. Drawn on frames when provided.
    """

    os.makedirs(save_path, exist_ok=True)
    for i, vid in enumerate(video_batch):
        gif_frames = []
        for frame_idx, frame in enumerate(vid):
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            # [10, 4]
            frame_bbxs = bbxs[i, frame_idx, :, :].cpu().numpy()
            # optionally draw gt-bbxs on frame
            if bbxs is not None:
                frame = draw_bbxs_on_frame(frame, frame_bbxs)
            # print(f"frame shape: {frame.shape}")
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")

        # write video to out
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                # breakpoint()
                writer.append_data(frame)

        if args is None or not args.wandb:
            continue

        # log video to wandb
        wandb.log(
            {key + f"_video_{i}": wandb.Video(now_save_path, fps=fps, format="mp4")},
            step=args.iteration + 1,
        )


def log_video(batch: Dict, model, args, only_log_video_latents=False):

    texts = batch["txt"]
    text_save_dir = os.path.join(args.save, "video_texts")

    # save text prompts to output dir
    os.makedirs(text_save_dir, exist_ok=True)
    save_texts(texts, text_save_dir, args.iteration)

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

                # currently seeing len(videos) == 3
                # [B, T, C, H, W]
                samples = (videos[k] + 1.0) / 2.0
                filename = "{}_gs-{:06}".format(k, args.iteration)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)

                # [B, T, 10, 4]
                bbxs: torch.Tensor = batch["bbox"]

                # TODO: draw gt bbxs on vized samples
                # TODO: use an arg to make this functionality optional
                save_video_as_grid_and_mp4(
                    samples, path, num_frames // fps, fps, args, k, bbxs=bbxs
                )
