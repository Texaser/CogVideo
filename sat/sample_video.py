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


from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from PIL import Image


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
    image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
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

def add_noised_conditions_to_frames(image, segm_tensor, noise_mode='segm'):
    """
    Injects Gaussian noise into each frame of the image based on the segmentation masks,
    excluding the first frame which retains the reference image with added noise.
    """
    B, C, T, H, W = image.shape
    _, _, num_objects, _, _ = segm_tensor.shape  # [B, T, 10, H, W]

    # Initialize a tensor to store noise masks
    noise_masks = torch.zeros_like(image)

    # Only add Gaussian noise to frames after the first
    for b in range(B):
        for t in range(1, T-1):
            # Process segmentation masks for each object
            for obj_idx in range(num_objects):
                # Get segmentation mask for current object
                mask = segm_tensor[b, t, obj_idx]  # [H, W]

                # Skip if mask is empty
                if not torch.any(mask):
                    continue

                # Generate noise for masked region
                noise = torch.randn(C, H, W, device=image.device, dtype=image.dtype)
                scaled_noise = noise * mask.unsqueeze(0)  # Scale noise by mask

                # Add noise to the image in place
                image[b, :, t] += scaled_noise
                
                # Store the noise mask
                noise_masks[b, :, t] += scaled_noise

    return image, noise_masks

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
                colored_frame = add_noise_to_rgb(colored_frame, mask_rgb)
            # Scale colors to [-1, 1] range and assign πto image
            image[b, :, t] = colored_frame

    return image, None

def add_original_color_conditions_to_frames(image, segm_tensor, original_frames):
    """
    Encodes each player with the corresponding RGB values from the original frames 
    based on the segmentation mask, with noise applied only in masked regions.
    """
    B, C, T, H, W = image.shape
    _, _, num_objects, _, _ = segm_tensor.shape  # [B, T, num_objects, H, W]

    # Only modify frames after the first one (keep first frame as reference)
    for b in range(B):
        for t in range(1, T-1):
            # Start with a black frame
            colored_frame = torch.zeros((C, H, W), device=image.device, dtype=image.dtype)
            
            # Add each player's RGB values from the original frame
            for obj_idx in range(num_objects):
                mask = segm_tensor[b, t, obj_idx]  # [H, W]
                
                if not torch.any(mask):
                    continue
                
                mask_rgb = mask[None, :, :]  # Shape [1, H, W] for broadcasting over channels
                
                # Apply the mask to get RGB values from the original frame
                for c in range(C):
                    colored_frame[c] += mask * original_frames[b, c, t, :, :]
                
                # Apply noise only to masked regions using the add_noise_to_rgb method
                colored_frame = add_noise_to_rgb(colored_frame, mask_rgb)
                
            # Assign the colored frame with original RGB values (with noise) to the image tensor
            image[b, :, t] = colored_frame

    return image, None

def load_and_process_masks(mask_paths, indices, num_frames, video_size, original_size):
    """
    Load and process segmentation masks for selected frames for all players
    mask_paths: list of paths to mask files for each player
    Handles sparse CSR matrix format for masks
    """
    # Initialize tensor to store masks for all objects
    processed_masks = torch.zeros(
        (num_frames, 10, video_size[0], video_size[1]), 
        dtype=torch.float16
    )
    
    # Calculate scaling and cropping parameters
    if original_size[1] / original_size[0] > video_size[1] / video_size[0]:
        scale = video_size[0] / original_size[0]
        new_height = video_size[0]
        new_width = int(original_size[1] * scale)
    else:
        scale = video_size[1] / original_size[1]
        new_width = video_size[1]
        new_height = int(original_size[0] * scale)

    delta_h = new_height - video_size[0]
    delta_w = new_width - video_size[1]
    top = delta_h // 2
    left = delta_w // 2

    # Load and process masks for each player
    for player_idx, mask_path in enumerate(mask_paths):
        masks_dict = np.load(mask_path, allow_pickle=True).item()
        
        for i, idx in enumerate(indices[:num_frames]):
            frame_key = f'frame_{idx}'
            if frame_key in masks_dict:
                # Convert sparse matrix to dense numpy array
                sparse_mask = masks_dict[frame_key]
                mask = torch.from_numpy(sparse_mask.toarray()).float()
                
                # Resize and crop mask
                processed_mask = resize_and_crop_mask(
                    mask,
                    original_size,
                    video_size,
                    scale,
                    top,
                    left
                )
                
                processed_masks[i, player_idx] = processed_mask

    return processed_masks

def resize_and_crop_mask(mask, original_size, target_size, scale, top, left):
        """
        Resize and crop a single mask to match the video frame processing
        mask: tensor of shape (H, W)
        """
        # Resize mask to match the scaled size before cropping
        if original_size[1] / original_size[0] > target_size[1] / target_size[0]:
            scale = target_size[0] / original_size[0]
            new_height = target_size[0]
            new_width = int(original_size[1] * scale)
        else:
            scale = target_size[1] / original_size[1]
            new_width = target_size[1]
            new_height = int(original_size[0] * scale)

        # Resize mask using nearest neighbor interpolation
        resized_mask = TT.functional.resize(
            mask.unsqueeze(0).unsqueeze(0),
            size=[new_height, new_width],
            interpolation=InterpolationMode.NEAREST
        ).squeeze()

        # Crop the mask
        cropped_mask = TT.functional.crop(
            resized_mask.unsqueeze(0),
            top=top,
            left=left,
            height=target_size[0],
            width=target_size[1]
        ).squeeze()

        return cropped_mask


def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
    # load_checkpoint(model, args, specific_iteration=5000)
    load_checkpoint(model, args)
    model.eval()

    if args.input_type == "cli":
        data_iter = read_from_cli()
    elif args.input_type == "txt":
        rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        print("rank and world_size", rank, world_size)
        data_iter = read_from_file(args.input_file, rank=rank, world_size=world_size)
    else:
        raise NotImplementedError

    image_size = [480, 720]
    num_frames = 49
    if args.image2video:
        chained_trainsforms = []
        chained_trainsforms.append(TT.ToTensor())
        transform = TT.Compose(chained_trainsforms)

    sample_func = model.sample
    T, H, W, C, F = args.sampling_num_frames, image_size[0], image_size[1], args.latent_channels, 8
    num_samples = [1]
    force_uc_zero_embeddings = ["txt"]
    device = model.device
    with torch.no_grad():
        for text, cnt in tqdm(data_iter):
            if args.image2video:
                text, image_path = text.split("@@")
                assert os.path.exists(image_path), image_path
                folder_path = os.path.dirname(image_path)
                first_image = Image.open(image_path).convert("RGB")
                first_image = transform(first_image).unsqueeze(0).to("cuda")
                first_image = resize_for_rectangle_crop(first_image, image_size, reshape_mode="center").unsqueeze(0)
                first_image = first_image * 2.0 - 1.0
                first_image = first_image.unsqueeze(2).to(torch.bfloat16)
                original_frames = torch.load(os.path.join(folder_path, "frames.pth"))
                # import pudb; pudb.set_trace();
                if args.noised_image_input:                   
                    image = add_noise_to_frame(first_image)
                    if args.noise_last_frame:
                        last_image_path = image_path.replace('_first', '_last')
                        last_image = Image.open(last_image_path).convert("RGB")
                        last_image = transform(last_image).unsqueeze(0).to("cuda")
                        last_image = resize_for_rectangle_crop(last_image, image_size, reshape_mode="center").unsqueeze(0)
                        last_image = last_image * 2.0 - 1.0
                        last_image = last_image.unsqueeze(2).to(torch.bfloat16)
                        last_frame = add_noise_to_frame(last_image)
                        subsequent_frames = torch.zeros(
                            (image.shape[0], image.shape[1], num_frames - 2, image.shape[3], image.shape[4]),
                            device=image.device,
                            dtype=image.dtype
                        )
                        image = torch.cat([image, subsequent_frames, last_frame], dim=2)
                    else:
                        subsequent_frames = torch.zeros(
                            (image.shape[0], image.shape[1], num_frames - 1, image.shape[3], image.shape[4]),
                            device=image.device,
                            dtype=image.dtype
                        )
                        image = torch.cat([image, subsequent_frames], dim=2)
                    # Add noise based on the selected noise_mode
                    
                    # Check if all player masks exist
                    player_mask_paths = []
                    all_masks_exist = True
                    for player_idx in range(10):
                        mask_path = f"{folder_path}/masks/object_{player_idx}_masks.npy"
                        if os.path.exists(mask_path):
                            player_mask_paths.append(mask_path)
                        else:
                            print(f"Warning: Mask not found for video {folder_path}, player {player_idx}")
                            all_masks_exist = False
                            break


                    masks = load_and_process_masks(
                        player_mask_paths, 
                        image, 
                        num_frames,
                        (480, 720),
                        (720, 1280)
                    )
                    masks = masks.unsqueeze(0)
                    image, noise_masks = add_original_color_conditions_to_frames(image, masks, original_frames)
                # import pudb; pudb.set_trace();
                # image = Image.open(image_path).convert("RGB")
                # image = transform(image).unsqueeze(0).to("cuda")
                # image = resize_for_rectangle_crop(image, image_size, reshape_mode="center").unsqueeze(0)
                # image = image * 2.0 - 1.0
                # image = image.unsqueeze(2).to(torch.bfloat16)
                image = model.encode_first_stage(image, None)
                image = image.permute(0, 2, 1, 3, 4).contiguous()
                # pad_shape = (image.shape[0], T - 1, C, H // F, W // F)
                # image = torch.concat([image, torch.zeros(pad_shape).to(image.device).to(image.dtype)], dim=1)
            else:
                image = None

            value_dict = {
                "prompt": text,
                "negative_prompt": "",
                "num_frames": torch.tensor(T).unsqueeze(0),
            }

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples
            )
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    print(key, batch[key].shape)
                elif isinstance(batch[key], list):
                    print(key, [len(l) for l in batch[key]])
                else:
                    print(key, batch[key])
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=force_uc_zero_embeddings,
            )

            for k in c:
                if not k == "crossattn":
                    c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))

            if args.image2video and image is not None:
                c["concat"] = image
                uc["concat"] = image

            for index in range(args.batch_size):
                # reload model on GPU
                model.to(device)
                samples_z = sample_func(
                    c,
                    uc=uc,
                    batch_size=1,
                    shape=(T, C, H // F, W // F),
                )
                samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()

                # Unload the model from GPU to save GPU memory
                model.to("cpu")
                torch.cuda.empty_cache()
                first_stage_model = model.first_stage_model
                first_stage_model = first_stage_model.to(device)

                latent = 1.0 / model.scale_factor * samples_z

                # Decode latent serial to save GPU memory
                recons = []
                loop_num = (T - 1) // 2
                for i in range(loop_num):
                    if i == 0:
                        start_frame, end_frame = 0, 3
                    else:
                        start_frame, end_frame = i * 2 + 1, i * 2 + 3
                    if i == loop_num - 1:
                        clear_fake_cp_cache = True
                    else:
                        clear_fake_cp_cache = False
                    with torch.no_grad():
                        recon = first_stage_model.decode(
                            latent[:, :, start_frame:end_frame].contiguous(), clear_fake_cp_cache=clear_fake_cp_cache
                        )

                    recons.append(recon)

                recon = torch.cat(recons, dim=2).to(torch.float32)
                samples_x = recon.permute(0, 2, 1, 3, 4).contiguous()
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                save_path = os.path.join(
                    args.output_dir, str(cnt) + "_" + text.replace(" ", "_").replace("/", "")[:120], str(index)
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

    sampling_main(args, model_cls=SATVideoDiffusionEngine)