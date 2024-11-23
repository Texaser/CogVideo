import math
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

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
import os 

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
            "A basketball player committing a disqualifying foul",
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


class BboxTokenEmbedder(AbstractEmbModel):
    def __init__(
        self, 
        hidden_dim, 
        max_length=40,
        max_distance=2,
        token_dim=1920,
        tokens_per_trajectory=4,
        num_attn_heads=8,
        device="cuda",
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.max_distance = max_distance
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.tokens_per_trajectory = tokens_per_trajectory
        self.device = device
        self.dtype = dtype

        # Relative position embeddings
        self.dist_embedding = nn.Embedding(max_distance + 1, hidden_dim).to(device).to(dtype)
        
        # Main attention for processing trajectory
        self.attention = nn.MultiheadAttention(hidden_dim, num_attn_heads, batch_first=True).to(device).to(dtype)
        
        # Learnable token queries for attention pooling
        self.token_queries = nn.Parameter(torch.randn(1, tokens_per_trajectory, hidden_dim, dtype=dtype)).to(device)
        self.token_attention = nn.MultiheadAttention(hidden_dim, num_attn_heads, batch_first=True).to(device).to(dtype)
            
        # Initial projection layer
        self.input_proj = nn.Linear(4, hidden_dim).to(device).to(dtype)
            
        # Final projection to desired token dimension
        self.token_proj = nn.Linear(hidden_dim, token_dim).to(device).to(dtype)
        
        # Layer norms for stability
        self.norm1 = nn.LayerNorm(hidden_dim).to(device).to(dtype)
        self.norm2 = nn.LayerNorm(hidden_dim).to(device).to(dtype)
        
    def compute_relative_positions(self, trajectory):
        """Compute relative position matrix for a sequence"""
        batch_size, seq_len, _ = trajectory.shape
        positions = torch.arange(seq_len, device=self.device)
        rel_pos = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        rel_pos = torch.clamp(rel_pos, 0, self.max_distance)
        
        # Get embeddings [seq_len, seq_len, hidden_dim]
        rel_pos_emb = self.dist_embedding(rel_pos)
        
        # Expand for batch dimension [batch_size, seq_len, seq_len, hidden_dim]
        rel_pos_emb = rel_pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Reshape to match trajectory shape
        rel_pos_emb = rel_pos_emb.sum(dim=2)  # [batch_size, seq_len, hidden_dim]
        
        return rel_pos_emb
        
    def process_single_player_trajectory(self, trajectory):
        """Process a single player's trajectory
        Args:
            trajectory: [seq_len, 4] tensor of bbox coordinates
        """
        # Ensure trajectory is on correct device and dtype
        trajectory = trajectory.to(device=self.device, dtype=self.dtype)
        
        # Add batch dimension
        trajectory = trajectory.unsqueeze(0)  # [1, seq_len, 4]
        
        # Project to hidden dim
        trajectory = self.input_proj(trajectory)  # [1, seq_len, hidden_dim]
        
        # Add position embeddings
        rel_pos_emb = self.compute_relative_positions(trajectory)
        trajectory = trajectory + rel_pos_emb
        
        # Process sequence with attention
        sequence, _ = self.attention(trajectory, trajectory, trajectory)
        sequence = self.norm1(sequence)
        
        # Extract tokens using learned queries
        tokens, _ = self.token_attention(
            self.token_queries,
            sequence,
            sequence
        )
        tokens = self.norm2(tokens)
        
        # Project to final dimension
        tokens = self.token_proj(tokens)
        
        return tokens.squeeze(0)  # [tokens_per_trajectory, token_dim]
        
    def forward(self, bbox_tensor):
        """
        Args:
            bbox_tensor: [batch_size, num_frames, num_players, 4] bbox coordinates
                where batch_size=1, num_frames=49, num_players=10
        Returns:
            tokens: [1, max_length, token_dim]
        """
        # Ensure input is on correct device and dtype
        bbox_tensor = bbox_tensor.to(device=self.device, dtype=self.dtype)
        
        batch_size, num_frames, num_players, _ = bbox_tensor.shape
        
        all_tokens = []
        # Process each batch
        for b in range(batch_size):
            # Process each player's trajectory
            for p in range(num_players):
                # Extract single player trajectory [num_frames, 4]
                player_trajectory = bbox_tensor[b, :, p]
                # Process trajectory
                player_tokens = self.process_single_player_trajectory(player_trajectory)
                all_tokens.append(player_tokens)
                
        # Concatenate all tokens
        tokens = torch.cat(all_tokens, dim=0)  # [batch_size * num_players * tokens_per_trajectory, token_dim]
        
        # Pad or truncate to max_length
        if tokens.shape[0] < self.max_length:
            padding = torch.zeros(
                self.max_length - tokens.shape[0], 
                self.token_dim, 
                device=self.device,
                dtype=self.dtype
            )
            tokens = torch.cat([tokens, padding], dim=0)
        else:
            tokens = tokens[:self.max_length]
            
        return tokens.unsqueeze(0)  # [1, max_length, token_dim]

    def encode(self, bbox_tensor):
        return self(bbox_tensor)