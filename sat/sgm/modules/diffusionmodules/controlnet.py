from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from diffusers.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput, CogVideoXBlock
from diffusers.utils import is_torch_version
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

class CogVideoXControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        time_embed_dim: int = 512,
        num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        out_proj_dim = None,
    ):
        super().__init__()
        self._dtype = torch.bfloat16
        inner_dim = num_attention_heads * attention_head_dim

        # Single zero conv for conditioning features
        self.zero_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.zero_conv.to(dtype=self._dtype)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        
        # Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels * 2,
            embed_dim=inner_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.patch_embed.to(dtype=self._dtype)
        
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                ).to(dtype=self._dtype)
                for _ in range(num_layers)
            ]
        )

        # Output projections
        if out_proj_dim is not None:
            self.out_projectors = nn.ModuleList(
                [nn.Linear(inner_dim, out_proj_dim).to(dtype=self._dtype) for _ in range(num_layers)]
            )
            self.zero_projs = nn.ModuleList(
                [nn.Linear(out_proj_dim, out_proj_dim).to(dtype=self._dtype) for _ in range(num_layers)]
            )
            for zero_proj in self.zero_projs:
                nn.init.zeros_(zero_proj.weight)
                nn.init.zeros_(zero_proj.bias)
        else:
            self.out_projectors = None
            self.zero_projs = None
            
        self.gradient_checkpointing = False

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_states: torch.Tensor,
        timestep: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
    ):
        # Cast inputs to the model's dtype
        hidden_states = hidden_states.to(dtype=self._dtype)
        controlnet_states = controlnet_states.to(dtype=self._dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype=self._dtype)
            
        batch_size = hidden_states.shape[0]
        
        batch_size, num_frames, channels, height, width = controlnet_states.shape
        
        # Process controlnet states
        controlnet_states = rearrange(controlnet_states, 'b f c h w -> (b f) c h w')
        controlnet_states = self.zero_conv(controlnet_states)
        controlnet_states = rearrange(controlnet_states, '(b f) c h w -> b f c h w', b=batch_size)

        # Concat with hidden states
        hidden_states = torch.cat([hidden_states, controlnet_states], dim=2)
        
        # Patch embedding and processing
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        if text_seq_length > 0:
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]
        
        # Process through transformer blocks
        controlnet_hidden_states = ()
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    image_rotary_emb,
                    use_reentrant=False
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=timestep,
                    image_rotary_emb=image_rotary_emb,
                )
                
            if self.out_projectors is not None:
                projected = self.out_projectors[i](hidden_states)
                projected = self.zero_projs[i](projected)
                controlnet_hidden_states += (projected,)
            else:
                controlnet_hidden_states += (hidden_states,)
            
        if not return_dict:
            return (controlnet_hidden_states,)
        return Transformer2DModelOutput(sample=controlnet_hidden_states)