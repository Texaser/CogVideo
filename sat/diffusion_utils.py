from torchvision.utils import save_image
import os
import cv2
import numpy as np 
from PIL import Image
import torch


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
            noise_mask_np = (noise_mask.permute(1, 2, 0).cpu().numpy()).astype(np.uint8) * 255

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
    #self.write_noise_masks(noise_masks)
    return image, noise_masks

def add_noised_pose_to_frames(image, bbox_tensor, pose_tensor, noise_mode='bbox'):
    """
    Draws a confidence-aware pose skeleton encoding in-place onto the image based on the pose keypoints,
    excluding the first frame which retains the reference image.
    Returns the modified image and the pose masks for visualization.
    """
    B, C, T, H, W = image.shape  # Image shape: [B, C, T, H, W]
    _, _, N, K, _ = pose_tensor.shape  # Pose tensor shape: [B, T, N, K, 3]
    
    # Initialize a tensor to store pose masks (for visualization, optional)
    pose_masks = torch.zeros_like(image)
    
    # Define the limb connections (based on COCO keypoint indices)
    limb_seq = [
        [0, 1], [1, 3], [0, 2], [2, 4],  # Head
        [0, 5], [5, 7], [7, 9],          # Left arm
        [0, 6], [6, 8], [8, 10],         # Right arm
        [5, 6],                          # Shoulders
        [5, 11], [6, 12],                # Torso
        [11, 12],                        # Hips
        [11, 13], [13, 15],              # Left leg
        [12, 14], [14, 16],              # Right leg
    ]
    
    # Draw the confidence-aware pose skeleton onto frames after the first
    for b in range(B):
        for t in range(1, T):
            for n in range(N):
                # Process pose keypoints
                keypoints = pose_tensor[b, t, n]  # Shape: [K, 3]
                kp_coords = []
                for k in range(K):
                    x_norm, y_norm, c = keypoints[k]
                    x = x_norm * W
                    y = y_norm * H
                    kp_coords.append((x, y, c))
                    # Skip invalid keypoints (e.g., zero confidence)
                    if c <= 0 or x < 0 or x >= W or y < 0 or y >= H:
                        continue
                    # Draw point at (x, y) with intensity c
                    radius = 3  # Adjust the radius as needed
                    draw_point(image[b, :, t], x, y, radius, c)
                    draw_point(pose_masks[b, :, t], x, y, radius, c)
                
                # Draw limbs
                for limb in limb_seq:
                    k1, k2 = limb
                    x1, y1, c1 = kp_coords[k1]
                    x2, y2, c2 = kp_coords[k2]
                    # If both keypoints have confidence > 0
                    if c1 <= 0 or c2 <= 0:
                        continue
                    c = min(c1, c2)
                    # Draw line from (x1, y1) to (x2, y2) with intensity c
                    draw_line(image[b, :, t], x1, y1, x2, y2, c)
                    draw_line(pose_masks[b, :, t], x1, y1, x2, y2, c)
    
    # Optionally, write pose masks for visualization
    #self.write_noise_masks(pose_masks)
    return image, pose_masks

def draw_point(image, x, y, radius, intensity):
    """
    Draws a filled circle (point) on the image tensor at (x, y) with the given radius and intensity.
    """
    H, W = image.shape[-2], image.shape[-1]
    x = int(x)
    y = int(y)
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(W, x + radius + 1)
    y2 = min(H, y + radius + 1)
    
    # Create a meshgrid for the patch
    patch_x = torch.arange(x1, x2, device=image.device)
    patch_y = torch.arange(y1, y2, device=image.device)
    grid_x, grid_y = torch.meshgrid(patch_x, patch_y, indexing='ij')
    
    # Compute the distance from the center
    dist_squared = (grid_x - x) ** 2 + (grid_y - y) ** 2
    radius_squared = radius ** 2
    mask = dist_squared <= radius_squared
    
    # Apply the intensity scaled by the mask
    image[:, grid_y[mask], grid_x[mask]] += intensity

def draw_line(image, x1, y1, x2, y2, intensity):
    """
    Draws a line on the image tensor from (x1, y1) to (x2, y2) with the given intensity.
    """
    x1 = x1.item()
    y1 = y1.item()
    x2 = x2.item()
    y2 = y2.item()
    
    num_points = int(max(abs(x2 - x1), abs(y2 - y1))) * 2  # Increase for smoother lines
    if num_points == 0:
        num_points = 1
    x_coords = torch.linspace(x1, x2, num_points, device=image.device)
    y_coords = torch.linspace(y1, y2, num_points, device=image.device)
    x_coords = x_coords.round().long()
    y_coords = y_coords.round().long()
    
    # Ensure coordinates are within image bounds
    H, W = image.shape[-2], image.shape[-1]
    valid_idx = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
    x_coords = x_coords[valid_idx]
    y_coords = y_coords[valid_idx]
    
    # Draw the line by setting the intensity at the line coordinates
    image[:, y_coords, x_coords] += intensity