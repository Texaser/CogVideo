import random
import math
from typing import Any, Dict, List, Tuple, Union
from omegaconf import ListConfig
import torch.nn.functional as F
from sat.helpers import print_rank0
import torch
from torch import nn
from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
import torchvision.transforms as transforms
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
)
import gc
from sat import mpu
from torchvision.utils import save_image
import os
import cv2
import numpy as np 
from sgm import draw_segmentation_overlay, draw_annotations, add_color_conditions_to_frames, add_original_color_conditions_to_frames, add_noised_conditions_to_frames, add_noise_to_rgb

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self._init_configs(args.model_config)
        self._init_model_components(args)
        self.device = args.device

    def _init_configs(self, model_config: Dict):
        """Initialize configuration parameters from model config"""
        # Basic configs
        self.log_keys = model_config.get("log_keys", None)
        self.input_key = model_config.get("input_key", "mp4")
        self.not_trainable_prefixes = model_config.get("not_trainable_prefixes", ["first_stage_model", "conditioner"])
        self.en_and_decode_n_samples_a_time = model_config.get("en_and_decode_n_samples_a_time", None)
        self.lr_scale = model_config.get("lr_scale", None)
        self.scale_factor = model_config.get("scale_factor", 1.0)
        
        # Model behavior configs
        self.latent_input = model_config.get("latent_input", False)
        self.lora_train = model_config.get("lora_train", False)
        self.use_pd = model_config.get("use_pd", False)
        self.disable_first_stage_autocast = model_config.get("disable_first_stage_autocast", False)
        self.no_cond_log = model_config.get("no_cond_log", False)
        
        # Noise and image processing configs
        self.noised_image_input = model_config.get("noised_image_input", False)
        self.noised_image_all_concat = model_config.get("noised_image_all_concat", False)
        self.noised_image_dropout = model_config.get("noised_image_dropout", 0.0)
        self.noise_last_frame = model_config.get("noise_last_frame", False)
        self.pixel_space_loss = model_config.get("pixel_space_loss", False)
        self.noise_mode = model_config.get("noise_mode", "bbox")

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    def _init_model_components(self, args):
        """Initialize model components and set datatypes"""
        # Set dtype based on args
        self.dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
        self.dtype_str = "fp16" if args.fp16 else "bf16" if args.bf16 else "fp32"

        # Initialize network
        network_config = args.model_config.get("network_config")
        network_config["params"]["dtype"] = self.dtype_str
        network_wrapper = args.model_config.get("network_wrapper", None)
        compile_model = args.model_config.get("compile_model", False)
        
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=self.dtype
        )

        # Initialize other components
        self.denoiser = instantiate_from_config(args.model_config.get("denoiser_config"))
        self.sampler = instantiate_from_config(args.model_config.get("sampler_config")) if args.model_config.get("sampler_config") else None
        self.conditioner = instantiate_from_config(default(args.model_config.get("conditioner_config"), UNCONDITIONAL_CONFIG))
        self._init_first_stage(args.model_config.get("first_stage_config"))
        self.loss_fn = instantiate_from_config(args.model_config.get("loss_fn_config")) if args.model_config.get("loss_fn_config") else None
        self.control_net = instantiate_from_config(args.model_config.get("control_net_config")) if args.model_config.get("control_net_config") else None

        # Instantiate control net
        if self.control_net is not None:
            controlnet_state_dict = {}
            for name, params in self.model.state_dict().items():
                if 'patch_embed.proj.weight' in name:
                    continue
                controlnet_state_dict[name] = params
            m, u = self.control_net.load_state_dict(controlnet_state_dict, strict=False)
            print(f'[ Weights from transformer was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

        # if args.pretrained_controlnet_path:
        #     ckpt = torch.load(args.pretrained_controlnet_path, map_location='cpu', weights_only=False)
        #     controlnet_state_dict = {}
        #     for name, params in ckpt['state_dict'].items():
        #         controlnet_state_dict[name] = params
        #     m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        #     print(f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

        # Freeze backbone
        if args.model_config.get("freeze_backbone", False):
            for param in self.model.parameters():
                param.requires_grad = False
            print_rank0("***** Frozen backbone *****")

    def _init_first_stage(self, config):
        """Initialize first stage model"""
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def disable_untrainable_params(self):
        """Disable gradients for untrainable parameters"""
        total_trainable = 0
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
                
            should_train = not any(n.startswith(prefix) or prefix == "all" 
                                 for prefix in self.not_trainable_prefixes)
            
            # Keep LoRA parameters trainable
            if any(prefix in n for prefix in ["matrix_A", "matrix_B"]):
                should_train = True
                
            p.requires_grad_(should_train)
            if should_train:
                total_trainable += p.numel()

        print_rank0(f"***** Total trainable parameters: {total_trainable} *****")

    def forward(self, x, batch):
        """Forward pass with optional pixel space loss"""
        loss, model_output, noised_input, alphas_cumprod_sqrt = self.loss_fn(
            self.model, self.denoiser, self.conditioner, x, batch
        )

        if not self.pixel_space_loss:
            return loss.mean(), {"loss": loss.mean()}

        return self._compute_pixel_space_loss(
            loss, model_output, noised_input, alphas_cumprod_sqrt, batch, x
        )

    def _compute_pixel_space_loss(self, loss, model_output, noised_input, alphas_cumprod_sqrt, batch, x):
        """
        Compute pixel-space loss using segmentation masks
        
        Args:
            loss: Base diffusion loss
            model_output: Model predictions
            noised_input: Input with noise added
            alphas_cumprod_sqrt: Square root of cumulative product of alpha values
            batch: Dictionary containing input batch data
            x: Input tensor
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        segm_tensor = batch['mask']
        B, T, num_objects, H, W = segm_tensor.shape
        
        # Calculate sigma and predicted noise
        sigma_t = (1 - alphas_cumprod_sqrt**2) ** 0.5
        pred_noised = alphas_cumprod_sqrt * x + sigma_t * model_output
        
        # Decode predictions
        x_hat = self.decode_first_stage(pred_noised.permute(0, 2, 1, 3, 4).contiguous().to(self.dtype))
        x_noised = self.decode_first_stage(noised_input.permute(0, 2, 1, 3, 4).contiguous().to(self.dtype))
        
        # Compute segmentation loss
        seg_loss = 0.0
        valid_seg_count = 0
        
        for b in range(B):
            for t in range(T):
                for obj_idx in range(num_objects):
                    mask = segm_tensor[b, t, obj_idx]
                    if not torch.any(mask):
                        continue
                        
                    pred = x_hat[b, :, t]
                    target = x_noised[b, :, t]
                    
                    seg_loss += F.mse_loss(
                        pred * mask.unsqueeze(0),
                        target * mask.unsqueeze(0),
                        reduction='sum'
                    )
                    valid_seg_count += mask.sum().item() * pred.shape[0]
        
        # Compute final loss
        if valid_seg_count > 0:
            seg_loss = seg_loss / valid_seg_count
            total_loss = loss.mean() + seg_loss
            return total_loss, {"loss": total_loss, "seg_loss": seg_loss}
        
        return loss.mean(), {"loss": loss.mean()}

    def add_noise_to_frame(self, image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image
    
    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        if self.lr_scale is not None:
            lr_x = F.interpolate(x, scale_factor=1 / self.lr_scale, mode="bilinear", align_corners=False)
            lr_x = F.interpolate(lr_x, scale_factor=self.lr_scale, mode="bilinear", align_corners=False)
            lr_z = self.encode_first_stage(lr_x, batch)
            batch["lr_input"] = lr_z

        x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.noised_image_input:
            image = x[:, :, 0:1]
            image = self.add_noise_to_frame(image)
            num_frames = batch['mask'].shape[1]  # Get number of frames from mask tensor

            if self.noise_last_frame:
                last_frame = self.add_noise_to_frame(x[:, :, -1:])
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
            
            image, noise_masks = add_noised_conditions_to_frames(
                image, batch['bbox']
            ) if self.noise_mode == 'bbox' else add_color_conditions_to_frames(image, batch['mask'])
        

            image = self.encode_first_stage(image, batch)

        x = self.encode_first_stage(x, batch)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if self.noised_image_input:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
            if random.random() < self.noised_image_dropout:
                image = torch.zeros_like(image)
            batch["concat_images"] = image

        gc.collect()
        torch.cuda.empty_cache()
        loss, loss_dict = self(x, batch)
        
        return loss, loss_dict

    def get_input(self, batch):
        return batch[self.input_key].to(self.dtype)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch):
        frame = x.shape[2]

        if frame > 1 and self.latent_input:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            return x * self.scale_factor  # already encoded

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        prefix=None,
        concat_images=None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)

        if prefix is not None:
            randn = torch.cat([prefix, randn[:, prefix.shape[1] :]], dim=1)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        if mp_size > 1:
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(randn, src=src, group=mpu.get_model_parallel_group())

        scale = None
        scale_emb = None

        def denoiser(input, sigma, c, **additional_model_inputs):
            return self.denoiser(
                self.model, 
                input, 
                sigma, 
                c,
                concat_images=concat_images,
                **additional_model_inputs
            )

        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb, control_net=self.control_net)
        samples = samples.to(self.dtype)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
        self,
        batch: Dict,
        N: int = 8,
        ucg_keys: List[str] = None,
        only_log_video_latents=False,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        if not self.latent_input:
            log["inputs"] = x.to(torch.float32)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.encode_first_stage(x, batch)
        if not only_log_video_latents:
            log["reconstructions"] = self.decode_first_stage(z).to(torch.float32)
            log["reconstructions"] = log["reconstructions"].permute(0, 2, 1, 3, 4).contiguous()
        z = z.permute(0, 2, 1, 3, 4).contiguous()

        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], torch.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if self.noised_image_input:
            image = x[:, :, 0:1]
            image = self.add_noise_to_frame(image)
            num_frames = batch['mask'].shape[1]

            if self.noise_last_frame:
                last_frame = self.add_noise_to_frame(x[:, :, -1:])
                subsequent_frames = torch.zeros(
                    (image.shape[0], image.shape[1], num_frames - 2, image.shape[3], image.shape[4]),
                    device=image.device,
                    dtype=image.dtype
                )
                # subsequent_frames = self.add_noise_to_frame(x[:, :, 1:-1])
                image = torch.cat([image, subsequent_frames, last_frame], dim=2)
            else:
                subsequent_frames = torch.zeros(
                    (image.shape[0], image.shape[1], num_frames - 1, image.shape[3], image.shape[4]),
                    device=image.device,
                    dtype=image.dtype
                )
                image = torch.cat([image, subsequent_frames], dim=2)

            # Add noise based on the selected noise_mode
            image, noise_masks = add_noised_conditions_to_frames(
                image, batch['bbox']
            ) if self.noise_mode == 'bbox' else add_color_conditions_to_frames(image, batch['mask'])
            
            image = self.encode_first_stage(image, batch)
            image = image.permute(0, 2, 1, 3, 4).contiguous()

            c["concat"] = image
            uc["concat"] = image
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                
                # Store samples
                log["samples_raw"] = samples.clone()
                
                # Visualize samples with segmentation overlay
                samples_with_segm = draw_segmentation_overlay(samples.clone(), batch)
                log["samples_segm"] = samples_with_segm

                # Draw bounding boxes on the samples
                samples_with_bbox = draw_annotations(samples.clone(), batch, draw_bbox=True, draw_pose=False)
                log["samples_bbox"] = samples_with_bbox

                log["inputs_samples_comparison"] = torch.cat([log["inputs"], log["samples_raw"]], dim=4)

                log["inputs_samples_bbox_comparison"] = torch.cat([log["inputs"], log["samples_raw"], log["samples_bbox"]], dim=4)

                log["inputs_samples_segm_comparison"] = torch.cat([log["inputs"], log["samples_raw"], log["samples_segm"]], dim=4)
                
        return log
