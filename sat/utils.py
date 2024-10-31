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
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from PIL import Image
import wandb

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

def write_noise_masks(noise_masks, output_dir='noise_masks', prefix=''):
    """
    Writes noise masks to image files.

    Args:
    noise_masks (torch.Tensor): Tensor of shape [B, C, T, H, W] containing noise masks.
    output_dir (str): Directory to save the noise mask images.
    prefix (str): Prefix for the saved filenames.

    Returns:
    None
    """
    B, C, T, H, W = noise_masks.shape

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for b in range(B):
        for t in range(T):
            # Extract single noise mask
            noise_mask = noise_masks[b, :, t]

            noise_mask = noise_mask.float()

            # Convert to numpy array and scale to 0-255 range
            noise_mask_np = (noise_mask.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)

            # If the noise mask is single-channel, repeat it to create an RGB image
            if C == 1:
                noise_mask_np = np.repeat(noise_mask_np, 3, axis=2)

            # Create filename
            filename = f'{prefix}batch_{b}_frame_{t}.png'
            filepath = os.path.join(output_dir, filename)

            # Save the noise mask as an image
            cv2.imwrite(filepath, cv2.cvtColor(noise_mask_np, cv2.COLOR_RGB2BGR))

    print(f"Noise masks saved in {output_dir}")

def add_noised_conditions_to_frames(image, bbox_tensor, pose_tensor, noise_mode='bbox', joint_encodings=None, player_encodings=None):
    """
    Injects Gaussian noise into each frame of the image based on the bounding boxes and/or pose keypoints,
    excluding the first frame which retains the reference image with added noise.
    Returns the modified image and the noise masks for visualization.
    """
    B, C, T, H, W = image.shape  # Assuming image shape is [B, C, T, H, W]
    _, _, N, _ = bbox_tensor.shape  # N is the number of bounding boxes per frame
    _, _, _, K, _ = pose_tensor.shape  # K is the number of keypoints per player

    # Initialize a tensor to store noise masks
    noise_masks = torch.zeros_like(image)

    # Initialize joint encodings
    if joint_encodings is not None:
        joint_encodings = torch.tensor(joint_encodings, device=image.device, dtype=image.dtype)

    # Initialize player encodings
    if player_encodings is not None:
        player_encodings = torch.tensor(player_encodings, device=image.device, dtype=image.dtype)

    # Only add Gaussian noise to frames after the first
    for b in range(B):
        for t in range(1, T):
            if noise_mode in ('bbox', 'both'):
                # Process bounding boxes
                bboxes = bbox_tensor[b, t]  # Shape: [N, 4]
                for n in range(N):
                    bbox = bboxes[n]
                    x1_norm, y1_norm, x2_norm, y2_norm = bbox

                    # Convert normalized coordinates to pixel coordinates
                    x1 = x1_norm * W
                    y1 = y1_norm * H
                    x2 = x2_norm * W
                    y2 = y2_norm * H

                    # Ensure coordinates are in the correct order
                    x1, x2 = sorted([x1.item(), x2.item()])
                    y1, y2 = sorted([y1.item(), y2.item()])

                    # Convert to integers and clamp
                    x1 = int(max(0, min(W - 1, x1)))
                    y1 = int(max(0, min(H - 1, y1)))
                    x2 = int(max(x1 + 1, min(W, x2)))
                    y2 = int(max(y1 + 1, min(H, y2)))

                    h = y2 - y1
                    w = x2 - x1

                    if h <= 0 or w <= 0:
                        continue  # Skip invalid bounding boxes

                    # Generate a Gaussian mask
                    y_coords = torch.arange(h, device=image.device).unsqueeze(1).repeat(1, w)
                    x_coords = torch.arange(w, device=image.device).unsqueeze(0).repeat(h, 1)
                    mx = (h - 1) / 2.0
                    my = (w - 1) / 2.0
                    sx = h / 3.0
                    sy = w / 3.0
                    gaussian = (1 / (2 * math.pi * sx * sy)) * torch.exp(
                        -(((x_coords - my) ** 2) / (2 * sy ** 2) + ((y_coords - mx) ** 2) / (2 * sx ** 2))
                    )
                    gaussian = gaussian / gaussian.max()
                    gaussian = gaussian.to(image.dtype)

                    # Apply player-specific encoding if available
                    if player_encodings is not None:
                        player_encoding = player_encodings[n].unsqueeze(-1).unsqueeze(-1)
                        encoded_gaussian = gaussian.unsqueeze(0) * player_encoding
                    else:
                        encoded_gaussian = gaussian.unsqueeze(0)

                    # Generate noise scaled by the encoded Gaussian mask
                    noise = torch.randn(C, h, w, device=image.device, dtype=image.dtype) * encoded_gaussian

                    # Add noise to the image in place
                    image[b, :, t, y1:y2, x1:x2] += noise

                    # Store the noise mask
                    noise_masks[b, :, t, y1:y2, x1:x2] = noise

            if noise_mode in ('pose', 'both'):
                # Process pose keypoints
                keypoints = pose_tensor[b, t]  # Shape: [N, K, 2]
                for n in range(N):
                    player_keypoints = keypoints[n]  # Shape: [K, 2]
                    for k in range(K):
                        x_norm, y_norm = player_keypoints[k]

                        # Convert normalized coordinates to pixel coordinates
                        x = x_norm * W
                        y = y_norm * H

                        # Skip invalid keypoints (e.g., zero coordinates)
                        if x < 0 or x >= W or y < 0 or y >= H:
                            continue

                        x = int(torch.clamp(x, 0, W - 1).item())
                        y = int(torch.clamp(y, 0, H - 1).item())

                        # Define a small window around the keypoint
                        window_size = 15  # Adjust this value as needed
                        x1 = max(0, x - window_size // 2)
                        y1 = max(0, y - window_size // 2)
                        x2 = min(W, x + window_size // 2 + 1)
                        y2 = min(H, y + window_size // 2 + 1)

                        h = y2 - y1
                        w = x2 - x1

                        if h <= 0 or w <= 0:
                            continue

                        # Generate a Gaussian mask centered at the keypoint
                        y_coords = torch.arange(y1, y2, device=image.device).unsqueeze(1).repeat(1, w) - y
                        x_coords = torch.arange(x1, x2, device=image.device).unsqueeze(0).repeat(h, 1) - x
                        sx = h / 3.0
                        sy = w / 3.0
                        gaussian = (1 / (2 * math.pi * sx * sy)) * torch.exp(
                            -(((x_coords) ** 2) / (2 * sy ** 2) + ((y_coords) ** 2) / (2 * sx ** 2))
                        )
                        gaussian = gaussian / gaussian.max()
                        gaussian = gaussian.to(image.dtype)

                        # Apply joint and player-specific encodings if available
                        if joint_encodings is not None and player_encodings is not None:
                            joint_encoding = joint_encodings[k].unsqueeze(-1).unsqueeze(-1)
                            player_encoding = player_encodings[n].unsqueeze(-1).unsqueeze(-1)
                            encoded_gaussian = gaussian.unsqueeze(0) * joint_encoding * player_encoding
                        elif joint_encodings is not None:
                            joint_encoding = joint_encodings[k].unsqueeze(-1).unsqueeze(-1)
                            encoded_gaussian = gaussian.unsqueeze(0) * joint_encoding
                        elif player_encodings is not None:
                            player_encoding = player_encodings[n].unsqueeze(-1).unsqueeze(-1)
                            encoded_gaussian = gaussian.unsqueeze(0) * player_encoding
                        else:
                            encoded_gaussian = gaussian.unsqueeze(0)

                        # Generate noise scaled by the encoded Gaussian mask
                        noise = torch.randn(C, h, w, device=image.device, dtype=image.dtype) * encoded_gaussian

                        # Add noise to the image in place
                        image[b, :, t, y1:y2, x1:x2] += noise

                        # Store the noise mask
                        noise_masks[b, :, t, y1:y2, x1:x2] = noise
    #write_noise_masks(noise_masks)
    return image, noise_masks

def draw_annotations(samples, batch, draw_bbox=True, draw_pose=True):
        """
        Draws bounding boxes and/or pose annotations on the decoded video samples.

        Args:
            samples (Tensor): Decoded video samples of shape [B, T, C, H, W].
            batch (Dict): Batch dictionary containing 'bbox' and 'pose'.
            draw_bbox (bool): Whether to draw bounding boxes.
            draw_pose (bool): Whether to draw pose keypoints.

        Returns:
            Tensor: Video samples with annotations drawn, same shape as input samples.
        """
        B, T, C, H, W = samples.shape
        N = batch['bbox'].shape[2]  # Number of bounding boxes per frame
        K = batch['pose'].shape[3]  # Number of keypoints per player

        samples_with_annotations = samples.clone()

        for b in range(B):
            for t in range(T):
                frame = samples[b, t]  # Shape: [C, H, W]
                # Convert frame to numpy array
                frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                # Convert from [-1, 1] to [0, 255]
                frame_np = ((frame_np + 1.0) * 127.5).astype(np.uint8)
                # Convert from RGB to BGR for OpenCV
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                if draw_bbox:
                    # Get bounding boxes
                    bboxes = batch['bbox'][b, t]  # Shape: [N, 4]
                    for n in range(N):
                        bbox = bboxes[n]  # [4]
                        x1_norm, y1_norm, x2_norm, y2_norm = bbox.cpu().numpy()
                        # Skip invalid bounding boxes
                        if (x1_norm == x2_norm) and (y1_norm == y2_norm):
                            continue
                        x1 = int(x1_norm * W)
                        y1 = int(y1_norm * H)
                        x2 = int(x2_norm * W)
                        y2 = int(y2_norm * H)
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(W - 1, x1))
                        y1 = max(0, min(H - 1, y1))
                        x2 = max(0, min(W - 1, x2))
                        y2 = max(0, min(H - 1, y2))
                        # Draw rectangle on frame_np
                        cv2.rectangle(frame_np, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                if draw_pose:
                    # Get poses
                    poses = batch['pose'][b, t]  # Shape: [N, K, 2]
                    for n in range(N):
                        keypoints = poses[n]  # Shape: [K, 2]
                        for k in range(K):
                            x_norm, y_norm = keypoints[k].cpu().numpy()
                            # Skip invalid keypoints
                            if x_norm == 0 and y_norm == 0:
                                continue
                            x = int(x_norm * W)
                            y = int(y_norm * H)
                            # Ensure coordinates are within image bounds
                            x = max(0, min(W - 1, x))
                            y = max(0, min(H - 1, y))
                            # Draw circle on frame_np
                            cv2.circle(frame_np, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

                # Convert from BGR back to RGB
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                # Convert frame_np back to tensor
                frame_tensor = torch.from_numpy(frame_np.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0).to(samples.device)
                samples_with_annotations[b, t] = frame_tensor

        return samples_with_annotations