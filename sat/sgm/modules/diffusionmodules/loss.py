from typing import List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig
from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from sat import mpu
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
import os
import imageio
import numpy as np
class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss

def tokencompose_loss(attention_maps: torch.Tensor, 
                     instance_masks: torch.Tensor,
                     spatial_dims: tuple[int, int, int] = (13, 30, 45),  # (frames, height, width)
                     normalize_attention: bool = True) -> dict:
    """
    Args:
        instance_masks: Shape [B, frames=49, instances=10, height, width]
        spatial_dims: Target spatial dimensions (frames=13, height=30, width=45)
    """
    # Reshape attention maps to match spatial dimensions
    instance_masks = instance_masks.to(attention_maps.device, dtype=attention_maps.dtype)
    attention_maps = rearrange(attention_maps, 
                             '... (f h w) -> ... f h w',
                             f=spatial_dims[0], h=spatial_dims[1], w=spatial_dims[2])
    attention_maps = attention_maps.squeeze(1)
    
    B, _F , I, H, W = instance_masks.shape  # B=1, F=49, I=10, H=480, W=720
    # Sum across all instance masks for each frame
    instance_masks = instance_masks.sum(dim=2).clamp(max=1)  # Shape: [B, F, H, W]
    
    first_frame = instance_masks[:, 0:1, :, :]  # [B,1,H,W]

    remaining_frames = instance_masks[:, 1:, :, :]  # [B,48,H,W]

    remaining_frames = remaining_frames.reshape(B, 12, 4, H, W)

    remaining_frames = remaining_frames.mean(dim=2)  # [B,12,H,W]

    instance_masks = torch.cat([first_frame, remaining_frames], dim=1)  # [B,13,H,W]
    
    # Now resize spatial dimensions
    instance_masks = F.interpolate(
        instance_masks,  # Shape: [B, 13, H, W]
        size=(spatial_dims[1], spatial_dims[2]),  # Target: [B, 13, 30, 45]
        mode='bilinear',
        align_corners=False
    )
    
    # Skip first frame - only compute loss for remaining frames
    instance_masks = instance_masks[:, 1:]  # Shape: [B, 12, 30, 45]
    attention_maps = attention_maps[:, 1:]  # Shape: [B, 12, 30, 45]

    avg_attention = attention_maps.mean(dim=0, keepdim=True)
    # save_video_as_grid_and_mp4(instance_masks.unsqueeze(2), save_path_1)
    # save_video_as_grid_and_mp4(avg_attention.unsqueeze(2), save_path_2)
    # exit(0)
    if normalize_attention:
        # Normalize attention maps per frame
        avg_attention = avg_attention / (avg_attention.reshape(*avg_attention.shape[:-2], -1)
                                                    .sum(dim=-1)[..., None, None] + 1e-8)
    
    # Token loss
    activation_values = (avg_attention * instance_masks).reshape(*avg_attention.shape[:-2], -1).sum(dim=-1) / \
                       (avg_attention.reshape(*avg_attention.shape[:-2], -1).sum(dim=-1) + 1e-8)
    token_loss = (1.0 - activation_values).pow(2).mean()
    
    # Pixel loss
    bce_loss = torch.nn.BCELoss(reduction='mean')
    pixel_loss = bce_loss(avg_attention, instance_masks)
    
    return {
        "token_loss": token_loss,
        "pixel_loss": pixel_loss
    }

class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch, token_compose_loss=None):
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}
    
        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)
    
        noise = torch.randn_like(input)
    
        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())
    
        additional_model_inputs["idx"] = idx
    
        if self.offset_noise_level > 0.0:
            noise = (
                noise + append_dims(torch.randn(input.shape[0]).to(input.device), input.ndim) * self.offset_noise_level
            )
    
        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims(
            (1 - alphas_cumprod_sqrt**2) ** 0.5, input.ndim
        )
    
        if "concat_images" in batch.keys():
            cond["concat"] = batch["concat_images"]
    
        # Model output
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1 / (1 - alphas_cumprod_sqrt**2), input.ndim)  # v-pred
    
        # Compute the diffusion loss
        loss = self.get_loss(model_output, input, w)

        if token_compose_loss:
            # Compute tokencompose loss
            attention_module = network.diffusion_model.get_mixin("adaln_layer")

            attn_maps = attention_module.attention_extractor.get_attention_maps(
                # text - video
                source_range=(0, 226),
                target_range=(226, 17776),
                # full
                # source_range=(0, 17776),
                # target_range=(0, 17776),
                # video - video
                # source_range=(226, 17776),
                # target_range=(226, 17776),
                normalize=True
            )

            # attention_module.attention_extractor.visualize_attention(
            #     attn_maps=attn_maps,
            #     spatial_dims=(13, 30, 45),  # (frames, height, width)
            #     layer_range=(0, 42)
            # )

            compose_losses = tokencompose_loss(
                attention_maps=attn_maps,
                instance_masks=batch['mask'],
                spatial_dims=(13, 30, 45)  # Match visualization dimensions
            )
            # print("loss", loss)
            # print("token_loss", compose_losses['token_loss'])
            # print("pixel_loss", compose_losses['pixel_loss'])
            scale = 0.5
            loss = loss + scale * (compose_losses['token_loss'] + compose_losses['pixel_loss'])
            # print("loss_all", loss)
            attention_module.attention_extractor.clear()

    
        # Return additional values needed for x0 reconstruction
        return loss, model_output, noised_input, alphas_cumprod_sqrt

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
