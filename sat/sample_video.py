import os
import math
import argparse
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
import imageio
import cv2
import torch
import numpy as np
from einops import rearrange
import torchvision.transforms as TT
from PIL import Image
import matplotlib.pyplot as plt

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode

# Import SFTDataset from data_video.py
from data_video import SFTDataset
from torch.utils.data import DataLoader

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input("Please input English text (Ctrl-D quit): ")
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass


def read_from_file(p, rank=0, world_size=1):
    with open(p, "r") as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=math.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N)).reshape(N).tolist()
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None):
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


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def add_noise_to_frame(image):
    sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
    sigma = torch.exp(sigma).to(image.dtype)
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image = image + image_noise
    return image
def add_noise_to_rgb(image, mask):
    """
    Adds noise only to the masked regions of the image.
    """
    # Generate sigma values and noise
    sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(image.device)
    sigma = torch.exp(sigma).to(image.dtype)
    
    # Apply noise only to masked areas
    noise = torch.randn_like(image) * sigma[:, None, None]
    image_noise = noise * mask
    image = image + image_noise
    return image 
def add_color_conditions_to_frames(image, segm_tensor):
    """
    Instead of adding noise, encodes each player with a distinct color from the tab10 colormap.
    First frame is kept as reference, subsequent frames show color-coded players.
    """
    import matplotlib.pyplot as plt
    
    B, C, T, H, W = image.shape 
    _, _, num_objects, _, _ = segm_tensor.shape  # [B, T, 10, H, W]
    
    # Get colormap
    cmap = plt.get_cmap("tab10")
    
    # Only modify frames after the first one (keep first frame as reference)
    for b in range(B):
        for t in range(1, T-1):
            # Start with a black frame
            colored_frame = torch.zeros((C, H, W), device=image.device, dtype=image.dtype)
            # Add each player's colored mask
            for obj_idx in range(num_objects):
                mask = segm_tensor[b, t, obj_idx]  # [H, W]
                mask_rgb = mask[None, :, :]
                if not torch.any(mask):
                    continue
                
                # Get color for this player from colormap
                color = torch.tensor(cmap(obj_idx)[:3], device=image.device, dtype=image.dtype)

                # Add colored mask to frame
                for c in range(C):
                    colored_frame[c] += mask * color[c]
                #colored_frame = add_noise_to_rgb(colored_frame, mask_rgb)
            # Scale colors to [-1, 1] range and assign πto image
            image[b, :, t] = colored_frame

    return image, None

def save_frames(frames, save_dir, frame_prefix):
    os.makedirs(save_dir, exist_ok=True)
    frames = rearrange(frames, 'B C H W -> B H W C')  # [B, H, W, C]
    frames = (255.0 * frames).cpu().numpy().astype(np.uint8)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(save_dir, f"{frame_prefix}_{i:06d}.png"))

def save_segmentation_masks(segmentation_masks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    B, T, num_objects, H, W = segmentation_masks.shape

    for b in range(B):
        for t in range(T):
            colored_mask = torch.zeros((3, H, W), dtype=torch.float32)
            for obj_idx in range(num_objects):
                mask = segmentation_masks[b, t, obj_idx]  # [H, W]
                if not torch.any(mask):
                    continue
                color = torch.tensor(cmap(obj_idx)[:3], dtype=torch.float32)
                colored_mask += mask.unsqueeze(0) * color.view(3, 1, 1)
            # Clamp values
            colored_mask = torch.clamp(colored_mask, 0.0, 1.0)
            # Convert to numpy
            frame = (255.0 * colored_mask.permute(1, 2, 0)).numpy().astype(np.uint8)
            Image.fromarray(frame).save(os.path.join(save_dir, f"mask_{b:06d}_{t:06d}.png"))

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
    load_checkpoint(model, args)
    model.eval()

    # Move model to device and convert to bfloat16
    device = model.device
    model = model.to(device).to(torch.bfloat16)

    image_size = [480, 720]
    num_frames = 49
    # Create the test dataset and DataLoader
    test_dataset = SFTDataset(
        data_dir=args.test_data_dir,
        video_size=[480, 720],
        fps=args.sampling_fps,
        max_num_frames=num_frames,
        skip_frms_num=0,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, args.sampling_fps
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            video_frames = batch["mp4"].to(device)  # [B, T, C, H, W]
            video_frames = video_frames.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
            video_frames = video_frames.to(torch.bfloat16)
            text_prompts = batch["txt"]

            # Prepare save_path
            save_path = os.path.join(
                args.output_dir, f"{batch_idx}_{text_prompts[0].replace(' ', '_').replace('/', '')[:120]}"
            )

            if mpu.get_model_parallel_rank() == 0:
                os.makedirs(save_path, exist_ok=True)
    
                # Save original video frames
                original_video = video_frames.to(torch.float32)
                original_video = original_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
                original_video = torch.clamp((original_video + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                save_video_as_grid_and_mp4(original_video, os.path.join(save_path, 'original_video'), fps=args.sampling_fps)

                # Save first and last frames
                save_frames(original_video[:, 0], os.path.join(save_path, 'first_frames'), 'first_frame')
                save_frames(original_video[:, -1], os.path.join(save_path, 'last_frames'), 'last_frame')

                # Save segmentation masks
                segmentation_masks = batch["mask"].cpu()  # [B, T, num_objects, H, W]
                save_segmentation_masks(segmentation_masks, os.path.join(save_path, 'segmentation_masks'))

            value_dict = {
                "prompt": text_prompts[0],
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch_cond, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                num_samples,
            )

            # Convert conditioning tensors to bfloat16
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch_cond,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if k != "crossattn":
                    c[k] = c[k][: math.prod(num_samples)].to(device).to(torch.bfloat16)
                    uc[k] = uc[k][: math.prod(num_samples)].to(device).to(torch.bfloat16)

            if args.image2video:
                # Extract the first and last frames
                first_frame = add_noise_to_frame(video_frames[:, :, 0, :, :])
                last_frame = add_noise_to_frame(video_frames[:, :, -1, :, :])
                # first_frame = video_frames[:, :, 0, :, :]
                # last_frame = video_frames[:, :, -1, :, :]

                # Calculate the number of padding frames
                num_padding_frames = num_frames - 2
                zeros_padding = torch.zeros(
                    (first_frame.shape[0], first_frame.shape[1], num_padding_frames, first_frame.shape[2], first_frame.shape[3]),
                    device=device,
                    dtype=video_frames.dtype,
                )

                # Concatenate frames and padding along the time dimension
                image = torch.cat(
                    [first_frame.unsqueeze(2), zeros_padding, last_frame.unsqueeze(2)], dim=2
                )  # [B, C, T, H, W]
                
                image, noise = add_color_conditions_to_frames(image, batch["mask"].to('cuda'))
                # Encode the image using the model's first stage
                image = model.encode_first_stage(image, None)
                #image = image * model.scale_factor
                image = image.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]

                # Prepare the shape for sampling
                H, W = image.shape[-2], image.shape[-1]  # Update H and W based on the encoded image

                c["concat"] = image
                uc["concat"] = image
                # Sample using the model, providing the encoded image as the initial latent
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H, W),
                    ofs=torch.tensor([2.0]).to("cuda")
                )
            else:
                # Existing code for the non-image2video case
                image_size = args.sampling_image_size
                H, W = image_size[0], image_size[1]
                F = 8  # 8x downsampled
                image = None

                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                ).to("cuda")

            # Continue with decoding and saving the samples
            samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

            if args.only_save_latents:
                samples_z = 1.0 / model.scale_factor * samples_z
                save_path = os.path.join(
                    args.output_dir, f"{batch_idx}_{text_prompts[0].replace(' ', '_').replace('/', '')[:120]}"
                )
                os.makedirs(save_path, exist_ok=True)
                torch.save(samples_z, os.path.join(save_path, "latent.pt"))
                with open(os.path.join(save_path, "text.txt"), "w") as f:
                    f.write(text_prompts[0])
            else:
                samples_x = model.decode_first_stage(samples_z).to(torch.float32)
                samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                save_path = os.path.join(
                    args.output_dir, f"{batch_idx}_{text_prompts[0].replace(' ', '_').replace('/', '')[:120]}"
                )
                if mpu.get_model_parallel_rank() == 0:
                    save_video_as_grid_and_mp4(samples, save_path, fps=args.sampling_fps)


if __name__ == "__main__":
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    if not hasattr(args, 'test_data_dir'):
        args.test_data_dir = '/mnt/mir/fan23j/data/hq-poses-strict'

    sampling_main(args, model_cls=SATVideoDiffusionEngine)
