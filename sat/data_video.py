import os
import sys
import json
import time
import random
import torch
import decord
import numpy as np

from typing import List, Dict
from glob import glob
from sgm.webds import MetaDistributedWebDataset
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
from torchvision.io.video import (
    _check_av_available,
    _read_from_stream,
    _align_audio_frames,
)
from torchvision.transforms.functional import center_crop, resize
from functools import partial
from decord import VideoReader
from torch.utils.data import Dataset
from tqdm import tqdm

from video_processing_utils import (
    resize_for_rectangle_crop,
    pad_last_frame,
    process_fn_video,
)


MAX_VIDS = 1
BBX_NUM_PLAYERS = 10


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
                process_fn_video,
                num_frames=num_frames,
                image_size=image_size,
                fps=fps,
                skip_frms_num=skip_frms_num,
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

    def __init__(self, data_dir: str, video_size, fps, max_num_frames, skip_frms_num=3):
        """

        Args:
            data_dir (str): Path to the data directory with the following structure:
                - 'game'
                    - 'period'
                        - 'annotation.json'
            skip_frms_num (int):
                - ignore the first and the last xx frames, avoiding transitions.
        """

        assert os.path.isdir(data_dir), f"Error: could not find dir @{data_dir}"

        super(SFTDataset, self).__init__()

        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.video_paths = []
        self.captions = []
        self.tracklets = []
        start_time = time.time()

        # find all `.json` files in `data_dir`
        self.file_paths = glob(os.path.join(data_dir, "*", "*", "*.json"))[: MAX_VIDS]
        for fp in tqdm(self.file_paths, desc="Loading Training Data"):
            with open(fp, "r") as f:
                data = json.load(f)
                # TODO: fix path in annotations
                video_path = data["video_path"].replace("/playpen-storage", "/mnt/mir")
                self.video_paths.append(video_path)
                caption = data["caption"]
                self.captions.append(caption)
                bounding_boxes = data["bounding_boxes"]
                self.tracklets.append(self.encode_bbox_tracklet(bounding_boxes))

        end_time = time.time()
        loading_time = end_time - start_time
        print(f"\nData loading completed in {loading_time:.2f} seconds.")
        print(f"Loaded {len(self.video_paths)} video paths, and captions.")

    def __getitem__(self, index) -> Dict:
        """
        Returns:

            ```
            item = {
                "mp4": tensor_frms,
                "bbox": tracklet_frms,
                "txt": self.captions[index],
                "num_frames": num_frames,
                "fps": self.fps,
            }
            ```
        """

        decord.bridge.set_bridge("torch")
        video_path = self.video_paths[index]
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        assert ori_vlen / actual_fps * self.fps > self.max_num_frames
        num_frames = self.max_num_frames

        # first idx of data sample
        start = int(self.skip_frms_num)

        # last idx of data sample
        end = int(start + num_frames / self.fps * actual_fps)
        end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))

        # frame indicies we sample from the original video
        indices = np.arange(start, end, (end - start) // num_frames).astype(int)

        temp_frms = vr.get_batch(np.arange(start, end_safty))

        # TODO: could this be a bit dangerous?
        assert temp_frms is not None

        tensor_frms = (
            torch.from_numpy(temp_frms)
            if type(temp_frms) is not torch.Tensor
            else temp_frms
        )
        tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        tracklet_frms = self.tracklets[index][torch.tensor((indices - start).tolist())][
            :num_frames
        ]

        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # the len of indices may be less than num_frames, due to round error
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms, scale, top, left, orig_w, orig_h = resize_for_rectangle_crop(
            tensor_frms, self.video_size, reshape_mode="center"
        )
        tensor_frms = (tensor_frms - 127.5) / 127.5
        tracklet_frms = self.adjust_bounding_boxes(
            tracklet_frms, scale, top, left, orig_w, orig_h
        )
        item = {
            "mp4": tensor_frms,
            "bbox": tracklet_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def adjust_bounding_boxes(self, bounding_boxes, scale, top, left, orig_w, orig_h):
        """
        Convert bbxs from normalized: [0-1] format to standard, pixel value format.
        """

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

    def encode_bbox_tracklet(self, bounding_boxes: List[Dict]) -> torch.Tensor:
        """
        Encode bounding box instances into a tensor: [T, 10, 4]
        - # frames in original video
        - # players / frame
        - # coords
        """

        num_frames = len(bounding_boxes)
        num_players = BBX_NUM_PLAYERS
        trajectory_data = [
            [[0, 0, 0, 0] for _ in range(num_players)] for _ in range(num_frames)
        ]

        for frame_idx, frame in enumerate(bounding_boxes):

            # we should always have 10 players to frame
            # TODO: this seems dangerous
            bbx_instances = frame["bounding_box_instances"]
            assert len(bbx_instances) == num_players

            for player_idx, box in enumerate(bbx_instances):
                if box is not None:  # TODO: handle in data
                    trajectory_data[frame_idx][player_idx] = [
                        box["x1"],
                        box["y1"],
                        box["x2"],
                        box["y2"],
                    ]

        # TODO: handle type, should be bf16?
        return torch.tensor(trajectory_data, dtype=torch.float16)

    def __len__(self):
        return len(self.video_paths)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
