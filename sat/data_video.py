import glob
import io
import os
import sys
import threading
import math
import random
import torch
import decord
import torchvision.transforms as TT
import numpy as np

from sgm.webds import MetaDistributedWebDataset
from functools import partial
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
from torchvision.io import _video_opt
from torchvision.io.video import (
    _check_av_available,
    _read_from_stream,
    _align_audio_frames,
)
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
from decord import VideoReader
from torch.utils.data import Dataset
from utils.video_helpers import (
    resize_video_tensor_bilinear,
    pad_last_frame,
    process_fn_video,
)

CLIPS_DIR = "/mnt/mir/levlevi/hq-basketball-dataset/filtered-clips-aggressive-thresh"
VIDEO_EXT = ".mp4"


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
        Unified dataset for all conditions, including videos, text, and more to come.
        :param skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """

        super(SFTDataset, self).__init__()
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num
        self.video_paths = []
        self.captions = []

        # HACK: hard-coded data dir
        data_dir = CLIPS_DIR

        # check that data dir exists
        assert os.path.isdir(data_dir), f"Error: data-dir {data_dir} DNE"

        self.video_paths = glob.glob(data_dir, "*", "*", "*" + VIDEO_EXT)
        assert (
            len(self.video_paths) > 0
        ), f"Error: no valid samples found in dir: {data_dir}"

        # HACK: hard-coded data-dir + global caption for all clips
        CAPTION = "A basketball player making a three-point shot"
        self.captions = [CAPTION] * len(self.video_paths)
        print(f"Processing: {len(self.video_paths)} samples")

    def __getitem__(self, index: int) -> Dict:
        """
        Returns:
            Dict: ```{
            "mp4": tensor_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
            }```
        """

        # we could probably do this globally
        decord.bridge.set_bridge("torch")

        if index < 0 or index >= len(self.video_paths):
            raise IndexError(
                f"Error: index {index} is out of bounds for SFTDataset with {len(self.video_paths)} video sample!"
            )

        # get the next sample
        video_path = self.video_paths[index]
        try:
            assert os.path.isfile(video_path)
        except:
            print(f"Error: sample @{video_path} DNE, skipping...")
            return self.__getitem__(index + 1)

        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        # HACK: sometimes we read videos w/ 0 frames?
        num_frames = self.max_num_frames
        if num_frames == 0:
            print(f"Error: 0 frames; skipping video at index: {index}")
            return self.__getitem__(index + 1)

        # TODO: this massive try/except seems unwieldy
        # figure out what is throwing errors and why
        try:
            # cond 1.
            if ori_vlen / actual_fps * self.fps > self.max_num_frames:
                # e.g., 49 frames
                num_frames = self.max_num_frames
                # TODO: we always start sampling at i: 0 + `skip_frames`
                # would be nice to sample from a random, valid start point when possible
                start = int(self.skip_frms_num)
                end = int(start + num_frames / self.fps * actual_fps)
                end_safty = min(
                    int(start + num_frames / self.fps * actual_fps), int(ori_vlen)
                )
                # frames we will sample from a clip
                indices = np.arange(start, end, (end - start) // num_frames).astype(int)
                temp_frms = vr.get_batch(np.arange(start, end_safty))
                assert (
                    temp_frms is not None
                ), f"Error: could not sample frame batch from video @{video_path}"
                # convert the frame batch to a tensor on CPU
                tensor_frms = (
                    torch.from_numpy(temp_frms)
                    if type(temp_frms) is not torch.Tensor
                    else temp_frms
                )
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:
                # cond 2.
                if ori_vlen > self.max_num_frames:
                    num_frames = self.max_num_frames
                    start = int(self.skip_frms_num)
                    end = int(ori_vlen - self.skip_frms_num)
                    indices = np.arange(start, end, (end - start) // num_frames).astype(
                        int
                    )
                    temp_frms = vr.get_batch(np.arange(start, end))
                    assert temp_frms is not None
                    tensor_frms = (
                        torch.from_numpy(temp_frms)
                        if type(temp_frms) is not torch.Tensor
                        else temp_frms
                    )
                    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
                else:
                    # cond 3.
                    # e.g., vid len is 48 frames, sample #frames is 49 frames
                    def nearest_smaller_4k_plus_1(n):
                        remainder = n % 4
                        if remainder == 0:
                            return n - 3
                        else:
                            return n - remainder + 1

                    start = int(self.skip_frms_num)
                    end = int(ori_vlen - self.skip_frms_num)
                    num_frames = nearest_smaller_4k_plus_1(
                        end - start
                    )  # 3D VAE requires the number of frames to be 4k+1
                    end = int(start + num_frames)
                    temp_frms = vr.get_batch(np.arange(start, end))
                    assert temp_frms is not None
                    tensor_frms = (
                        torch.from_numpy(temp_frms)
                        if type(temp_frms) is not torch.Tensor
                        else temp_frms
                    )
        except:
            # HACK: catch-all except block; try to load another sample
            return self.__getitem__(index + 1)

        # HACK: the len of indices may be less than num_frames, due to round error
        tensor_frms = pad_last_frame(tensor_frms, self.max_num_frames)
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        # TODO: we should replace this cropping strat with a resize/rescale
        # tensor_frms = resize_for_rectangle_crop(
        #     tensor_frms, self.video_size, reshape_mode="center"
        # )

        # reshape video tensor w/ bilinear interpolation
        tensor_frms = resize_video_tensor_bilinear(tensor_frms, self.video_size)

        # normalize video : [-1, 1]
        tensor_frms = (tensor_frms - 127.5) / 127.5

        item = {
            "mp4": tensor_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def __len__(self):
        return len(self.video_paths)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
