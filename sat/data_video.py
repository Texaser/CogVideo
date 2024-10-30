import io
import os
import sys
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
import numpy as np
import torch
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import decord
from decord import VideoReader
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import time
# from video_caption import predict
sys.path.append("..")
from tools.caption.video_caption import predict

def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    with av.open(filename, metadata_errors="ignore") as container:
        if container.streams.audio:
            audio_timebase = container.streams.audio[0].time_base
        if container.streams.video:
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)

        if container.streams.audio:
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],
                {"audio": 0},
            )
            info["audio_fps"] = container.streams.audio[0].rate

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    original_height, original_width = arr.shape[2], arr.shape[3]
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        scale = image_size[0] / arr.shape[2]
        new_height = image_size[0]
        new_width = int(arr.shape[3] * scale)
    else:
        scale = image_size[1] / arr.shape[3]
        new_width = image_size[1]
        new_height = int(arr.shape[2] * scale)

    arr = resize(arr, size=[new_height, new_width], interpolation=InterpolationMode.BICUBIC)
    h, w = arr.shape[2], arr.shape[3]

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

    return arr, scale, top, left, original_width, original_height



def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    start = random.randint(skip_frms_num, max_seek + 1)
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(start, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    return pad_last_frame(tensor_frms, num_frames)


import threading


def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    timeout = 20
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None).contiguous()


def process_video(
    video_path,
    image_size=None,
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    """
    video_path: str or io.BytesIO
    image_size: .
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
    num_frames: wanted num_frames.
    wanted_fps: .
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    """

    video = load_video_with_timeout(
        video_path,
        duration=duration,
        num_frames=num_frames,
        wanted_fps=wanted_fps,
        actual_fps=actual_fps,
        skip_frms_num=skip_frms_num,
        nb_read_frames=nb_read_frames,
    )

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:
        r = next(src)
        if "mp4" in r:
            video_data = r["mp4"]
        elif "avi" in r:
            video_data = r["avi"]
        else:
            print("No video data found")
            continue

        if txt_key not in r:
            txt = ""
        else:
            txt = r[txt_key]

        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        else:
            txt = str(txt)

        duration = r.get("duration", None)
        if duration is not None:
            duration = float(duration)
        else:
            continue

        actual_fps = r.get("fps", None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        else:
            continue

        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps

        if duration is not None and duration < required_duration:
            continue

        try:
            frames = process_video(
                io.BytesIO(video_data),
                num_frames=num_frames,
                wanted_fps=fps,
                image_size=image_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5
        except Exception as e:
            print(e)
            continue

        item = {
            "mp4": frames,
            "txt": txt,
            "num_frames": num_frames,
            "fps": fps,
        }

        yield item


class VideoDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        num_frames,
        fps,
        skip_frms_num=0.0,
        nshards=sys.maxsize,
        seed=1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        txt_key="caption",
        **kwargs,
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []

        if path.startswith(";"):
            path, include_dirs = path.split(";", 1)
        super().__init__(
            path,
            partial(
                process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
            ),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs,
        )

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(path, **kwargs)


class SFTDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset, self).__init__()

        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.video_paths = []
        self.captions = []
        self.tracklets = []
        self.pose_tracklets = []  # New list to store pose tracklets
        # self.json_folders = []
        start_time = time.time()
        
        total_files = sum(1 for root, _, filenames in os.walk(data_dir) 
                          for filename in filenames if filename.endswith(".json"))
        with open("/mnt/bum/hanyi/repo/CogVideo/sat/configs/prompt.json", "r") as f:
            captions = json.load(f)
        with tqdm(total=total_files, desc="Loading Data") as pbar:
            for root, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    if filename.endswith(".json"):
                        with open(os.path.join(root, filename), "r") as f:
                            data = json.load(f)

                        # self.json_folders.append(root)
                        # os.mkdir(json_folder)
                        # TODO: fix path in annotations
                        video_path = data["video_path"].replace("/playpen-storage", "/mnt/mir")
                        self.video_paths.append(video_path)

                        caption = captions[video_path]
                        # prompt = "Please describe this video in detail."
                        # temperature = 0.1
                        # video_data = open(video_path, 'rb').read()
                        # caption = predict(prompt, video_data, temperature)
                        # caption = data['caption']
                        self.captions.append(caption)
                        # with open('/mnt/bum/hanyi/repo/CogVideo/sat/configs/captions.txt', 'a') as f:
                        #     f.write(f"{caption}@@{video_path}\n")
                        bounding_boxes = data['bounding_boxes']
                        trajectory_data, keypoints_data = self.encode_bbox_tracklet(bounding_boxes)
                        self.tracklets.append(trajectory_data)
                        self.pose_tracklets.append(keypoints_data)  # Store the pose data

                        pbar.update(1)
        # import pudb; pudb.set_trace();
        end_time = time.time()
        loading_time = end_time - start_time
        print(f"\nData loading completed in {loading_time:.2f} seconds.")
        print(f"Loaded {len(self.video_paths)} video paths, and captions.")

    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")
        video_path = self.video_paths[index]
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)
        import pudb; pudb.set_trace();
        assert ori_vlen / actual_fps * self.fps > self.max_num_frames
        num_frames = self.max_num_frames
        start = int(self.skip_frms_num)
        end = int(start + num_frames / self.fps * actual_fps)
        end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))
        indices = np.arange(start, end, (end - start) // num_frames).astype(int)
        temp_frms = vr.get_batch(np.arange(start, end_safty))
        assert temp_frms is not None
        tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
        tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        tracklet_frms = self.tracklets[index][torch.tensor((indices).tolist())][:num_frames]
        pose_frms = self.pose_tracklets[index][torch.tensor((indices).tolist())][:num_frames]  # Get pose data

        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # the len of indices may be less than num_frames, due to round error
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms, scale, top, left, orig_w, orig_h = resize_for_rectangle_crop(
            tensor_frms, self.video_size, reshape_mode="center"
        )
        tensor_frms = (tensor_frms - 127.5) / 127.5
        tracklet_frms = self.adjust_bounding_boxes(tracklet_frms, scale, top, left, orig_w, orig_h)
        pose_frms = self.adjust_keypoints(pose_frms, scale, top, left, orig_w, orig_h)  # Adjust keypoints

        #save tensor
        # torch.save(tensor_frms, os.path.join(self.json_folders[index], 'frames.pth'))
        # torch.save(tracklet_frms, os.path.join(self.json_folders[index], 'bbox.pth'))
        # torch.save(pose_frms, os.path.join(self.json_folders[index], 'pose.pth'))
        item = {
            "mp4": tensor_frms,
            "bbox": tracklet_frms,
            "pose": pose_frms,  # Include pose data in the item
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def adjust_bounding_boxes(self, bounding_boxes, scale, top, left, orig_w, orig_h):
        # Convert normalized coordinates to pixel coordinates in the original frame
        bounding_boxes[:, :, 0] *= orig_w  # x1
        bounding_boxes[:, :, 1] *= orig_h  # y1
        bounding_boxes[:, :, 2] *= orig_w  # x2
        bounding_boxes[:, :, 3] *= orig_h  # y2

        # Apply scaling
        bounding_boxes *= scale

        # Apply cropping offsets
        bounding_boxes[:, :, [0, 2]] -= left
        bounding_boxes[:, :, [1, 3]] -= top

        # Convert back to normalized coordinates in the new frame size
        bounding_boxes[:, :, 0] /= self.video_size[1]  # x1
        bounding_boxes[:, :, 1] /= self.video_size[0]  # y1
        bounding_boxes[:, :, 2] /= self.video_size[1]  # x2
        bounding_boxes[:, :, 3] /= self.video_size[0]  # y2

        # Clip values to [0, 1]
        bounding_boxes = bounding_boxes.clip(0, 1)

        return bounding_boxes

    def adjust_keypoints(self, keypoints, scale, top, left, orig_w, orig_h):
        # Convert normalized coordinates to pixel coordinates in the original frame
        keypoints[:, :, :, 0] *= orig_w  # x
        keypoints[:, :, :, 1] *= orig_h  # y

        # Apply scaling
        keypoints *= scale

        # Apply cropping offsets
        keypoints[:, :, :, 0] -= left
        keypoints[:, :, :, 1] -= top

        # Convert back to normalized coordinates in the new frame size
        keypoints[:, :, :, 0] /= self.video_size[1]  # x
        keypoints[:, :, :, 1] /= self.video_size[0]  # y

        # Clip values to [0, 1]
        keypoints = keypoints.clamp(0, 1)

        return keypoints

    def encode_bbox_tracklet(self, bounding_boxes):
        num_frames = len(bounding_boxes)
        num_players = 10
        num_keypoints = 17

        trajectory_data = [[[0, 0, 0, 0] for _ in range(num_players)] for _ in range(num_frames)]
        keypoints_data = [[[[0, 0] for _ in range(num_keypoints)] for _ in range(num_players)] for _ in range(num_frames)]

        for frame_idx, frame in enumerate(bounding_boxes):
            assert len(frame['bounding_box_instances']) == num_players
            for player_idx, box in enumerate(frame['bounding_box_instances']):
                if box is not None:
                    trajectory_data[frame_idx][player_idx] = [box['x1'], box['y1'], box['x2'], box['y2']]
                    if 'keypoints' in box and box['keypoints']:
                        keypoints = box['keypoints']
                        if len(keypoints) == 0:
                            # pad with zeros
                            keypoints_xy = [[0, 0] for _ in range(num_keypoints)]
                        else:
                            keypoints_xy = [[kp[0], kp[1]] for kp in keypoints]
                        keypoints_data[frame_idx][player_idx] = keypoints_xy

        # Convert to tensors
        trajectory_data = torch.tensor(trajectory_data, dtype=torch.float16)
        keypoints_data = torch.tensor(keypoints_data, dtype=torch.float16)
        return trajectory_data, keypoints_data

    def __len__(self):
        return len(self.video_paths)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
