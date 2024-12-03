import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import math
from einops import rearrange
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Union


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

def add_noised_conditions_to_frames(image, bbox_tensor):
        """
        Injects Gaussian noise into each frame of the image based on the bounding boxes and/or pose keypoints,
        excluding the first frame which retains the reference image with added noise.
        Returns the modified image and the noise masks for visualization.
        """
        B, C, T, H, W = image.shape  # Assuming image shape is [B, C, T, H, W]
        _, _, N, _ = bbox_tensor.shape  # N is the number of bounding boxes per frame

        # Initialize a tensor to store noise masks
        noise_masks = torch.zeros_like(image)

        # Only add Gaussian noise to frames after the first
        for b in range(B):
            for t in range(1, T):
            
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

                    encoded_gaussian = gaussian.unsqueeze(0)

                    # Generate noise scaled by the encoded Gaussian mask
                    noise = torch.randn(C, h, w, device=image.device, dtype=image.dtype) * encoded_gaussian

                    image[b, :, t, y1:y2, x1:x2].copy_(image[b, :, t, y1:y2, x1:x2] + noise)

                    # Store the noise mask
                    noise_masks[b, :, t, y1:y2, x1:x2] = noise
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

    
def draw_segmentation_overlay(samples, batch):
    """
    Draws segmentation mask overlays on the decoded video samples.
    
    Args:
        samples (Tensor): Decoded video samples of shape [B, T, C, H, W]
        batch (Dict): Batch dictionary containing 'segm' with shape [B, 10, T, H, W]
    
    Returns:
        Tensor: Video samples with segmentation overlays, same shape as input samples
    """
    B, T, C, H, W = samples.shape
    num_objects = batch['mask'].shape[2]

    # Define colors for different objects (in BGR format for OpenCV)
    colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255), # Yellow
        (128, 0, 0),   # Dark red
        (0, 128, 0),   # Dark green
        (0, 0, 128),   # Dark blue
        (128, 128, 0)  # Olive
    ]

    samples_with_overlay = samples.clone()

    for b in range(B):
        for t in range(T):
            frame = samples[b, t]  # Shape: [C, H, W]
            # Convert frame to numpy array
            frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            # Convert from [-1, 1] to [0, 255]
            frame_np = ((frame_np + 1.0) * 127.5).astype(np.uint8)
            # Convert from RGB to BGR for OpenCV
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Create overlay mask
            overlay = np.zeros_like(frame_np)
            
            # Draw each object's segmentation
            for obj_idx in range(num_objects):
                mask = batch['mask'][b, t, obj_idx].cpu().numpy()
                if not np.any(mask):
                    continue
                    
                # Create colored mask for current object
                color_mask = np.zeros_like(frame_np)
                color_mask[mask > 0] = colors[obj_idx]
                
                # Add to overlay with transparency
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

            # Combine frame with overlay
            frame_with_overlay = cv2.addWeighted(frame_np, 1.0, overlay, 0.5, 0)

            # Convert back to RGB and tensor format
            frame_with_overlay = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(
                frame_with_overlay.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
            ).to(samples.device)
            
            samples_with_overlay[b, t] = frame_tensor

    return samples_with_overlay

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

            # Convert from BGR back to RGB
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            # Convert frame_np back to tensor
            frame_tensor = torch.from_numpy(frame_np.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0).to(samples.device)
            samples_with_annotations[b, t] = frame_tensor

    return samples_with_annotations