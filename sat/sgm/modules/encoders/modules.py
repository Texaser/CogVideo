import math
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import os
import kornia
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import ListConfig
from torch.utils.checkpoint import checkpoint
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
)

from ...util import (
    append_dims,
    autocast,
    count_params,
    default,
    disabled_train,
    expand_dims_like,
    instantiate_from_config,
)


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 1, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig], cor_embs=[], cor_p=[]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

        if len(cor_embs) > 0:
            assert len(cor_p) == 2 ** len(cor_embs)
        self.cor_embs = cor_embs
        self.cor_p = cor_p

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def surely_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict, cond_or_not) -> Dict:
        assert embedder.legacy_ucg_val is not None
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if cond_or_not[i]:
                batch[embedder.input_key][i] = val
        return batch

    def get_single_embedding(
        self,
        embedder,
        batch,
        output,
        cond_or_not: Optional[np.ndarray] = None,
        force_zero_embeddings: Optional[List] = None,
    ):
        embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
        with embedding_context():
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                if embedder.legacy_ucg_val is not None:
                    if cond_or_not is None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    else:
                        batch = self.surely_get_ucg_val(embedder, batch, cond_or_not)
                emb_out = embedder(batch[embedder.input_key])
            elif hasattr(embedder, "input_keys"):
                emb_out = embedder(*[batch[k] for k in embedder.input_keys])
        assert isinstance(
            emb_out, (torch.Tensor, list, tuple)
        ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
        if not isinstance(emb_out, (list, tuple)):
            emb_out = [emb_out]
        for emb in emb_out:
            out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
            if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                if cond_or_not is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli((1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)),
                            emb,
                        )
                        * emb
                    )
                else:
                    emb = (
                        expand_dims_like(
                            torch.tensor(1 - cond_or_not, dtype=emb.dtype, device=emb.device),
                            emb,
                        )
                        * emb
                    )
            if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                emb = torch.zeros_like(emb)
            if out_key in output:
                output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
            else:
                output[out_key] = emb
        return output

    def forward(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []

        if len(self.cor_embs) > 0:
            batch_size = len(batch[list(batch.keys())[0]])
            rand_idx = np.random.choice(len(self.cor_p), size=(batch_size,), p=self.cor_p)
            for emb_idx in self.cor_embs:
                cond_or_not = rand_idx % 2
                rand_idx //= 2
                output = self.get_single_embedding(
                    self.embedders[emb_idx],
                    batch,
                    output=output,
                    cond_or_not=cond_or_not,
                    force_zero_embeddings=force_zero_embeddings,
                )

        for i, embedder in enumerate(self.embedders):
            if i in self.cor_embs:
                continue
            output = self.get_single_embedding(
                embedder, batch, output=output, force_zero_embeddings=force_zero_embeddings
            )
        return output

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        cor_embs = self.cor_embs
        cor_p = self.cor_p
        self.cor_embs = []
        self.cor_p = []

        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        self.cor_embs = cor_embs
        self.cor_p = cor_p

        return c, uc


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self,
        model_dir="google/t5-v1_1-xxl",
        device="cuda",
        max_length=77,
        freeze=True,
        cache_dir=None,
    ):
        super().__init__()
        if model_dir is not "google/t5-v1_1-xxl":
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
            self.transformer = T5EncoderModel.from_pretrained(model_dir, cache_dir=cache_dir)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()

        for param in self.parameters():
            param.requires_grad = False

    # @autocast
    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.autocast("cuda", enabled=False):
            outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class PrecomputedT5Embedder(AbstractEmbModel):
    """Uses precomputed T5 transformer embeddings for text"""

    def __init__(
        self,
        model_dir="",
        device="cuda",
        max_length=77,
        freeze=True,
        cache_dir=None,
    ):
        super().__init__()
        self.device = device

        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
        # first load to cpu
        embeddings = torch.load('action_embs.pth', map_location='cpu')
        self.embeddings = embeddings.to(self.local_rank).requires_grad_(False)

        self.basketball_actions = [
            "",
            "A basketball player missing a three-point shot",
            "A basketball player assisting on a play",
            "A basketball player setting a screen",
            "A basketball player grabbing a rebound",
            "A basketball player committing a turnover",
            "A basketball player making a free throw",
            "A basketball player missing a free throw",
            "A basketball player scoring and being fouled",
            "A basketball player missing a two-point shot",
            "A basketball player making a two-point shot",
            "A basketball player committing a foul",
            "A basketball player executing a pick and roll",
            "A basketball player posting up",
            "A basketball player stealing the ball",
            "A basketball player receiving a technical foul",
            "A basketball player making a three-point shot",
            "A basketball player committing their second foul",
            "A basketball player committing their third foul",
            "A basketball player committing an unsportsmanlike foul",
            "A basketball player making a three-pointer and being fouled",
            "A basketball player getting a second chance opportunity",
            "A basketball player making two free throws",
            "A basketball player missing two free throws",
            "A basketball player making three free throws",
            "A basketball player missing three free throws",
            "A basketball player committing a disqualifying foul"
        ]

        self.action_to_index = {action: i for i, action in enumerate(self.basketball_actions)}

    def get_embeddings_from_prompts(self, prompts, embeddings):
        indices = np.array([self.action_to_index[prompt] for prompt in prompts])
        assert len(indices) == len(prompts), f"Not all prompts found in action_to_index. Missing: {set(prompts) - set(self.action_to_index.keys())}"
        
        indices_tensor = torch.from_numpy(indices).to(self.local_rank)
        filtered_embeddings = torch.index_select(embeddings, 0, indices_tensor)
        
        return filtered_embeddings

    # @autocast
    def forward(self, text):
        return self.get_embeddings_from_prompts(text, self.embeddings)

    def encode(self, text):
        return self(text)

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)

class TrackletEmbedder(AbstractEmbModel):
    """Simple encoder for bounding box tracklets."""

    def __init__(
        self,
        out_dim,
        fourier_freqs=8,
        num_tracklets=10,
        use_bf16=False,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.out_dim = out_dim 
        self.num_tracklets = num_tracklets
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        self.temporal_conv = nn.Conv2d(self.position_dim + 64, 256, kernel_size=(3,1), padding=(1,0), dtype=self.dtype)

        self.tracklet_embeddings = nn.Parameter(torch.randn(num_tracklets, 64, dtype=self.dtype))
        self.mlp = nn.Sequential(
            nn.Linear(256, 512, dtype=self.dtype),
            nn.SiLU(),
            nn.Linear(512, 512, dtype=self.dtype),
            nn.SiLU(),
            nn.Linear(512, out_dim, dtype=self.dtype),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize tracklet embeddings
        nn.init.normal_(self.tracklet_embeddings, mean=0.0, std=0.02)

    def forward(self, tracklets):
        B, T, N, _ = tracklets.shape
        assert N == self.num_tracklets, f"Expected {self.num_tracklets} players, got {N}"
        
        # visibility mask for null tracklets
        visibility_mask = (tracklets.sum(dim=-1) != 0).to(self.dtype)  # B, T, N
        
        pos_embedding = self.fourier_embedder(tracklets)  # B, T, N, position_dim
        
        # tracklet-specific embeddings
        tracklet_emb = self.tracklet_embeddings.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        tracklet_embedding = torch.cat([pos_embedding, tracklet_emb], dim=-1)
        
        # temporal convolution
        temp_features = self.temporal_conv(tracklet_embedding.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # apply visibility mask
        temp_features = temp_features * visibility_mask.unsqueeze(-1)
        
        pooled_features = temp_features.mean(dim=1)
        
        output = self.mlp(pooled_features)

        assert output.shape == torch.Size([B, N, self.out_dim])
        return output.to(self.dtype)