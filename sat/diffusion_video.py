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

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get("log_keys", None)
        input_key = model_config.get("input_key", "mp4")
        network_config = model_config.get("network_config", None)
        network_wrapper = model_config.get("network_wrapper", None)
        denoiser_config = model_config.get("denoiser_config", None)
        sampler_config = model_config.get("sampler_config", None)
        conditioner_config = model_config.get("conditioner_config", None)
        first_stage_config = model_config.get("first_stage_config", None)
        loss_fn_config = model_config.get("loss_fn_config", None)
        scale_factor = model_config.get("scale_factor", 1.0)
        latent_input = model_config.get("latent_input", False)
        disable_first_stage_autocast = model_config.get("disable_first_stage_autocast", False)
        no_cond_log = model_config.get("disable_first_stage_autocast", False)
        not_trainable_prefixes = model_config.get("not_trainable_prefixes", ["first_stage_model", "conditioner"])
        compile_model = model_config.get("compile_model", False)
        en_and_decode_n_samples_a_time = model_config.get("en_and_decode_n_samples_a_time", None)
        lr_scale = model_config.get("lr_scale", None)
        lora_train = model_config.get("lora_train", False)
        self.use_pd = model_config.get("use_pd", False)  # progressive distillation

        self.log_keys = log_keys
        self.input_key = input_key
        self.not_trainable_prefixes = not_trainable_prefixes
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lr_scale = lr_scale
        self.lora_train = lora_train
        self.noised_image_input = model_config.get("noised_image_input", False)
        self.noised_image_all_concat = model_config.get("noised_image_all_concat", False)
        self.noised_image_dropout = model_config.get("noised_image_dropout", 0.0)
        self.noise_last_frame = model_config.get("noise_last_frame", False)

        # Add noise_mode configuration option
        self.noise_mode = model_config.get('noise_mode', 'pose')  # 'bbox', 'pose', or 'both'

        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        network_config["params"]["dtype"] = dtype_str
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config) if sampler_config is not None else None
        self.conditioner = instantiate_from_config(default(conditioner_config, UNCONDITIONAL_CONFIG))

        self._init_first_stage(first_stage_config)

        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config is not None else None

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

    def disable_untrainable_params(self):
        total_trainable = 0
        for n, p in self.named_parameters():
            if p.requires_grad == False:
                continue
            flag = False
            for prefix in self.not_trainable_prefixes:
                if n.startswith(prefix) or prefix == "all":
                    flag = True
                    break

            lora_prefix = ["matrix_A", "matrix_B"]
            for prefix in lora_prefix:
                if prefix in n:
                    flag = False
                    break

            if flag:
                p.requires_grad_(False)
            else:
                total_trainable += p.numel()

        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def forward(self, x, batch):
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"loss": loss_mean}
        return loss_mean, loss_dict

    def add_noise_to_frame(self, image):
        sigma = torch.normal(mean=-3.0, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image

    def add_noised_conditions_to_frames(self, image, bbox_tensor, pose_tensor, noise_mode='bbox'):
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

                        # Generate noise scaled by the Gaussian mask
                        noise = torch.randn(C, h, w, device=image.device, dtype=image.dtype) * gaussian.unsqueeze(0)

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

                            # Generate noise scaled by the Gaussian mask
                            noise = torch.randn(C, h, w, device=image.device, dtype=image.dtype) * gaussian.unsqueeze(0)

                            # Add noise to the image in place
                            image[b, :, t, y1:y2, x1:x2] += noise

                            # Store the noise mask
                            noise_masks[b, :, t, y1:y2, x1:x2] = noise

        return image, noise_masks

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
            num_frames = batch['bbox'].shape[1]

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

            # Add noise based on the selected noise_mode
            image, noise_masks = self.add_noised_conditions_to_frames(
                image, batch['bbox'], batch['pose'], noise_mode=self.noise_mode
            )

            # Encode the noised image
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

        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, **addtional_model_inputs
        )

        samples = self.sampler(denoiser, randn, cond, uc=uc, scale=scale, scale_emb=scale_emb)
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
            num_frames = batch['bbox'].shape[1]
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
            # Add noise based on the selected noise_mode
            image, noise_masks = self.add_noised_conditions_to_frames(
                image, batch['bbox'], batch['pose'], noise_mode=self.noise_mode
            )

            # Encode the noised image
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
                # Store the raw samples
                log["samples_raw"] = samples.clone()
                # Draw bounding boxes on the samples
                samples_with_bbox = self.draw_annotations(samples.clone(), batch, draw_bbox=True, draw_pose=False)
                log["samples_bbox"] = samples_with_bbox
                # Draw pose keypoints on the samples
                samples_with_pose = self.draw_annotations(samples.clone(), batch, draw_bbox=False, draw_pose=True)
                log["samples_pose"] = samples_with_pose
        else:
            samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)
            samples = samples.permute(0, 2, 1, 3, 4).contiguous()
            if only_log_video_latents:
                latents = 1.0 / self.scale_factor * samples
                log["latents"] = latents
            else:
                samples = self.decode_first_stage(samples).to(torch.float32)
                samples = samples.permute(0, 2, 1, 3, 4).contiguous()
                # Store the raw samples
                log["samples_raw"] = samples.clone()
                # Draw bounding boxes on the samples
                samples_with_bbox = self.draw_annotations(samples.clone(), batch, draw_bbox=True, draw_pose=False)
                log["samples_bbox"] = samples_with_bbox
                # Draw pose keypoints on the samples
                samples_with_pose = self.draw_annotations(samples.clone(), batch, draw_bbox=False, draw_pose=True)
                log["samples_pose"] = samples_with_pose
        return log
    
    def draw_annotations(self, samples, batch, draw_bbox=True, draw_pose=True):
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