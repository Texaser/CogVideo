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
from utils import (
    save_video_as_grid_and_mp4, 
    resize_for_rectangle_crop, 
    add_noise_to_frame, 
    write_noise_masks, 
    add_noised_conditions_to_frames
)

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
                bbox_path = os.path.join(folder_path, "bbox.pth")
                pose_path = os.path.join(folder_path, "pose.pth")
                bbox = torch.load(bbox_path)
                bbox = bbox.unsqueeze(0)
                pose = torch.load(pose_path)
                pose = pose.unsqueeze(0)
                first_image = Image.open(image_path).convert("RGB")
                first_image = transform(first_image).unsqueeze(0).to("cuda")
                first_image = resize_for_rectangle_crop(first_image, image_size, reshape_mode="center").unsqueeze(0)
                first_image = first_image * 2.0 - 1.0
                first_image = first_image.unsqueeze(2).to(torch.bfloat16)
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
                    image, noise_masks = add_noised_conditions_to_frames(
                        image, bbox, pose, noise_mode=args.noise_mode
                    )
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
