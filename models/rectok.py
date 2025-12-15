from typing import Optional
import logging
import random
from functools import partial
from math import sqrt
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor

from models.model_utils import SIZE_DICT

from models.gaussian import DiagonalGaussianDistribution
from utils.foundation_models import create_foundation_model

from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger("RecTok")


# ================================
# Utility Functions
# ================================

def _to_tensor(x):
    return x.clone().detach() if isinstance(x, torch.Tensor) else torch.tensor(x)

def rotate_half(x: Tensor) -> Tensor:
    """rotate half of the input tensor for rotary position embedding."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """apply rotary position embedding to input tensor."""
    freqs_cos, freqs_sin = freqs_cis.unsqueeze(1).chunk(2, dim=-1)
    return x * freqs_cos + rotate_half(x) * freqs_sin


def interpolate_pos_embed_2d(pos_embed, new_seq_len):
    """
    Interpolate position embeddings to a new sequence length using bicubic interpolation.
    """
    N, seq_len, dim = pos_embed.shape
    if seq_len == new_seq_len:
        return pos_embed
    
    size = int(sqrt(seq_len))
    
    assert size * size == seq_len, "Original sequence length is not a perfect square."

    new_size = int(sqrt(new_seq_len))
    logger.info(f"Resizing position embedding from {size}x{size} to {new_size}x{new_size}")

    pos_embed = pos_embed.permute(0, 2, 1).view(1, dim, size, size)
    new_pos_embed = F.interpolate(
        pos_embed, 
        size=(new_size, new_size), 
        mode='bicubic', 
        align_corners=False
    )
    new_pos_embed = new_pos_embed.flatten(2).permute(0, 2, 1)
    return new_pos_embed


def get_rope_tensor(
    dim: int,
    seq_h: int,
    seq_w: int,
    max_freq: float = 7.0,
    min_freq: float = 7e-4,
    add_cls: bool = False,
    n_register: int = 0,
    device=None,
    dtype=None,
) -> Tensor:
    """
    Build a 2D Rotary Position Embedding (RoPE) table for an H W token grid,
    optionally prepending a [CLS] token and N register tokens.

    Layout (row order): [CLS] [register x n_register] [grid tokens (seq_h * seq_w)]
    Output shape: (L, 2*dim), where L = (1 if add_cls else 0) + n_register + seq_h*seq_w

    Design choice:
    - CLS and register tokens receive identity rotation (angle=0 → cos=1, sin=0).
      This keeps them "position-agnostic" while grid tokens carry spatial phase.

    Args:
        dim: Head dimension used by RoPE. Must be divisible by 4 for 2D splitting.
        seq_h: Grid height (number of tokens along H).
        seq_w: Grid width (number of tokens along W).
        max_freq, min_freq: Frequency band range for RoPE.
        add_cls: If True, prepend one CLS row with identity rotation.
        n_register: Number of register rows to prepend after CLS, identity rotation.
        device, dtype: Torch device/dtype for created tensors.

    Returns:
        rope_table: Tensor of shape (L, 2*dim), concatenated [cos, sin] along last dim.
    """
    # Each axis (H and W) consumes dim//2 features; each needs even pairing → dim % 4 == 0.
    assert dim % 4 == 0, "dim must be a multiple of 4 for 2D RoPE."

    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.get_default_dtype()

    # Build 1D frequencies for half the channels (dim//2), mirrored to form even/odd pairs.
    # We create a geometric progression from max_freq down to min_freq.
    freqs_1d = max_freq * (max_freq / min_freq) ** torch.linspace(
        0, -1, dim // 4, device=device, dtype=dtype
    )  # length = dim//4
    freqs_1d = torch.cat([freqs_1d, freqs_1d])  # length = dim//2 (paired)

    # Place H-axis freqs in the first half, W-axis freqs in the second half of channels.
    freqs_2d = torch.zeros(2, dim, device=device, dtype=dtype)  # shape (2, dim)
    freqs_2d[0, : dim // 2] = freqs_1d             # H axis
    freqs_2d[1, -dim // 2 :] = freqs_1d            # W axis
    freqs_2d = freqs_2d * 2 * torch.pi             # angular frequencies

    # Build normalized coordinates in [0, 1] for H and W, then the full grid (N = H*W).
    coord_x = torch.linspace(0, 1, seq_h, device=device, dtype=dtype)
    coord_y = torch.linspace(0, 1, seq_w, device=device, dtype=dtype)
    coords_all = torch.cartesian_prod(coord_x, coord_y)  # (N, 2) with columns [x, y]

    # Compute per-token angles by multiplying coords with axis frequencies.
    # angle_grid: (N, dim), each column corresponds to one channel's angle.
    angle_grid = coords_all @ freqs_2d

    # Special tokens (CLS + registers) should not be rotated → zero angles.
    num_special = (1 if add_cls else 0) + int(n_register)
    if num_special > 0:
        angle_special = torch.zeros(num_special, dim, device=device, dtype=dtype)
        angle = torch.cat([angle_special, angle_grid], dim=0)  # (num_special + N, dim)
    else:
        angle = angle_grid

    # Return concatenated cos/sin for downstream rotary application on Q/K.
    # Final shape: (L, 2*dim)
    rope_tensor = torch.cat([angle.cos(), angle.sin()], dim=-1)
    return rope_tensor


# ================================
# Neural Network Components
# ================================


class SwiGLUFFN(nn.Module):
    """Swish-Gated Linear Unit Feed-Forward Network."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Attention(nn.Module):
    """multi-head attention with rotary position embedding."""

    def __init__(self, dim: int, num_heads: int = 8, use_qknorm: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.use_qknorm = use_qknorm
        if self.use_qknorm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        bsz, n_ctx, ch = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, rope), apply_rotary_emb(k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(bsz, n_ctx, ch))


class Block(nn.Module):
    """transformer block with attention and feed-forward layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = partial(nn.RMSNorm, eps=1e-6),
        use_qknorm: bool = False,
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)
        self.attn = Attention(dim, num_heads, use_qknorm=use_qknorm)
        self.mlp = SwiGLUFFN(dim, int(2 / 3 * dim * mlp_ratio))

    def forward(self, x: Tensor, rope: Tensor = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


# ================================
# Encoder and Decoder
# ================================


class Encoder(nn.Module):
    """vision Transformer encoder with masked autoencoding capability."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base",
        token_channels: int = 16,
        mask_ratio: float = 0.75,
        mask_ratio_min: float = -0.1,
        random_mask_ratio: bool = True,
        cls_token_type: str = "none",
        diff_cls_token: bool = False,
        disable_kl: bool = False,
        use_qknorm: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        if disable_kl:
            self.token_channels = token_channels
        else:
            # needs to split into mean and std
            self.token_channels = token_channels * 2

        self.mask_ratio = mask_ratio
        self.mask_ratio_min = mask_ratio_min
        self.random_mask_ratio = random_mask_ratio
        self.seq_len = self.grid_size ** 2
        self.cls_token_type = cls_token_type
        self.diff_cls_token = diff_cls_token
        self.disable_kl = disable_kl
        self.use_qknorm = use_qknorm
        
        size_dict = SIZE_DICT[self.model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]
        self.width = width

        # patch embedding layer
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, width, self.patch_size, self.patch_size),
            Rearrange("b c h w -> b (h w) c", h=self.grid_size, w=self.grid_size),
        )

        # learnable embeddings
        scale = width ** -0.5
        if self.cls_token_type == "learnable":
            self.sem_cls_token_embedding = nn.Parameter(scale * torch.randn(1, 1, width))
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, 1 + self.seq_len, width))
        else:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))
            
        # transformer layers
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer, use_qknorm=self.use_qknorm) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)
        self.latent_head = nn.Linear(width, self.token_channels)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size, add_cls=self.cls_token_type == "learnable").unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[RecTok-Encoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}, random mask ratio: {self.random_mask_ratio}")

    def unpatchify(self, x: Tensor, chans: int, patch_size: int) -> Tensor:
        """convert patches back to image format."""
        bsz = x.shape[0]
        h_ = w_ = self.grid_size
        x = x.reshape(bsz, h_, w_, chans, patch_size, patch_size)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, chans, h_ * patch_size, w_ * patch_size)
        return x

    def mae_random_masking(self, x: Tensor):
        """apply masked autoencoding random masking."""
        bsz, seq_len, chans = x.shape
        # mask: 0 for visible, 1 for masked
        if self.mask_ratio == 0 or not self.training:
            # no masking
            rope = self.rope_tensor.expand(bsz, -1, -1)
            return x, torch.zeros(bsz, seq_len, device=x.device), None, rope, None, None

        if self.random_mask_ratio:
            mask_ratio = max(0.0, random.uniform(self.mask_ratio_min, self.mask_ratio))
        else:
            mask_ratio = self.mask_ratio

        len_keep = int(np.ceil(seq_len * (1 - mask_ratio)))
        noise = torch.rand(bsz, seq_len, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        # ids_restore[:, i] = j means ith token in the image ranks jth in the shuffled sequence: ids_shuffle
        ids_restore = torch.argsort(ids_shuffle, dim=1) # [bsz, seq_len]
        ids_keep = ids_shuffle[:, :len_keep] # [bsz, len_keep]
        ids_masked = ids_shuffle[:, len_keep:] # [bsz, seq_len - len_keep]
        x_visible = torch.gather(x, 1, ids_keep[..., None].repeat(1, 1, chans)) # x_visible[i, j, k] = x[i, ids_keep[i, j, k], k]
        rope = self.rope_tensor.expand(bsz, -1, -1) # [bsz, seq_len, head_dim]
        rope_visible = torch.gather(rope, 1, ids_keep[..., None].repeat(1, 1, rope.shape[-1]))

        mask = torch.ones(bsz, seq_len, device=x.device)
        mask[:, :len_keep] = 0
        # ids_restore[:, i] >= len_keep means ith token in the original sequence is masked
        mask = torch.gather(mask, dim=1, index=ids_restore) # mask[i, j] = mask[i, ids_restore[i, j]]
        return x_visible, mask, ids_restore, rope_visible, ids_keep, ids_masked

    def forward(self, x: Tensor):
        """forward pass through encoder."""
        # if self.use_skip_connection:
        #     x_skip = x
        #     x_skip = F.pixel_unshuffle(x_skip, self.patch_size)
        #     x_skip = x_skip.flatten(2).transpose(1, 2) # [bsz, seq_len, chans]
        #     num_chunks = x_skip.shape[-1] // self.width
        #     assert num_chunks * self.width == x_skip.shape[-1], f"num_chunks * width != chans, got {num_chunks} and {self.width}"
        #     x_skip = x_skip.unflatten(-1, (self.width, num_chunks)).mean(dim=-1)
        #     x = self.patch_embed(x) + x_skip
        # else:
        x = self.patch_embed(x)
        
        if self.cls_token_type == "learnable":
            x = torch.cat([self.sem_cls_token_embedding.expand(x.shape[0], -1, -1), x], dim=1)
            
        x = x + self.positional_embedding
            
        x, _, ids_restore, rope, ids_keep, ids_masked = self.mae_random_masking(x)
        
        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x, rope)
        
        x = self.ln_post(x)
        z = self.latent_head(x)

        if self.cls_token_type == "pooling":
            z_cls = z.mean(1).unsqueeze(1)
            z = torch.cat([z_cls, z], dim=1)

        ret = dict(
            z=z,    # [bsz, seq_len + 1, dim] if cls_token_type is not none, [bsz, seq_len, dim] otherwise
            ids_restore=ids_restore,
            ids_keep=ids_keep,
            ids_masked=ids_masked,
        )

        return ret


class DINOv3Encoder(nn.Module):
    """vision Transformer encoder with masked autoencoding capability."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        frozen_dinov3: bool = False,
        img_size: int = 256,
        token_channels: int = 16,
        last_layer_feature: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        if os.path.exists("offline_models/dinov3_vit_base_patch14"):
            pretrained_model_name_or_path = "offline_models/dinov3_vit_base_patch14"
            
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.frozen_dinov3 = frozen_dinov3
        if frozen_dinov3:
            self.model.eval()
            self.model.requires_grad_(False)
        self.config = self.model.config
        self.img_size = img_size
        # needs to split into mean and std
        self.token_channels = token_channels * 2
        self.last_layer_feature = last_layer_feature
        
        # output layer
        self.width = self.config.hidden_size
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_post = norm_layer(self.width)
        self.latent_head = nn.Linear(self.width, self.token_channels)

        total_params_M = sum(p.numel() for p in self.parameters()) / 1e6
        trainable_params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(
            f"[DeTok-Encoder] params: total {total_params_M:.2f}M, trainable {trainable_params_M:.2f}M, DINOv3 {os.path.basename(pretrained_model_name_or_path)}"
        )

    def forward(self, x: Tensor):
        """forward pass through encoder."""
        x = (x + 1) * 0.5
        x = (x * 255).to(torch.uint8)
        inputs = self.processor(x, return_tensors="pt").to(self.model.device)
        inputs["pixel_values"] = F.interpolate(
            inputs["pixel_values"], 
            size=(self.img_size, self.img_size), 
            mode="bilinear", 
            align_corners=False
        )
        if self.frozen_dinov3:
            with torch.inference_mode():
                outputs = self.model(**inputs)
            x = outputs.last_hidden_state
        else:
            x = self.model(**inputs).last_hidden_state
            
        x = x[:, 1 + self.config.num_register_tokens:, :]
        
        x = self.ln_post(x)
        z = self.latent_head(x)

        ret = dict(
            z=z,
            z_sem=z,
            ids_restore=None,
            ids_keep=None,
            ids_masked=None,
        )

        return ret


class Decoder(nn.Module):
    """vision Transformer decoder with mask tokens for image reconstruction."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base",
        token_channels: int = 16,
        diff_cls_token: bool = False,
        use_qknorm: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        self.token_channels = token_channels
        self.seq_len = self.grid_size ** 2
        self.diff_cls_token = diff_cls_token
        self.use_qknorm = use_qknorm
        
        params = SIZE_DICT[self.model_size]
        num_layers, num_heads, width = params["layers"], params["heads"], params["width"]

        # learnable embeddings
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))
        
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, width))

        # decoder layers
        self.decoder_embed = nn.Linear(self.token_channels, width)
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer, use_qknorm=self.use_qknorm) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)

        # output layers
        self.ffn = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=self.grid_size, w=self.grid_size),
            nn.Conv2d(width, self.patch_size * self.patch_size * 3, 1, padding=0),
            Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size),
        )
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(
            head_dim, 
            self.grid_size, 
            self.grid_size, 
            add_cls=False
        ).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-Decoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")

    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None) -> Tensor:
        """forward pass through decoder."""
        # z_latents: [bsz, seq_len, token_channels] or [bsz, 1 + seq_len, token_channels] if diff_cls_token
        z = self.decoder_embed(z_latents)
        bsz, seq_len, _ = z.shape

        if ids_restore is not None:
            num_mask_tokens = ids_restore.shape[1] + 1 - seq_len
            mask_tokens = self.mask_token.repeat(bsz, num_mask_tokens, 1)
            z_ = torch.cat([z, mask_tokens], dim=1)
            expanded_ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, z_.shape[-1])
            z = torch.gather(z_, dim=1, index=expanded_ids_restore)
            
        z = z + self.positional_embedding

        z = self.ln_pre(z)
        rope = self.rope_tensor.expand(bsz, -1, -1)
        for block in self.transformer:
            z = block(z, rope)
        z = self.ln_post(z)

        if self.diff_cls_token:
            z = z[:, 1:]

        z = self.ffn(z)  # embed -> patch
        z = self.conv_out(z)  # final 3x3 conv

        return z


# ================================
# Sem Decoder
# ================================


class TransformerDecoder(nn.Module):
    """Sem decoder for training the model."""
    
    def __init__(
        self, 
        img_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base", 
        token_channels: int = 16,
        sem_embed_dim: int = 1024,
        sem_cls_token: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.seq_len = self.grid_size ** 2
        
        self.model_size = model_size
        size_dict = SIZE_DICT[self.model_size]
        num_layers, num_heads, width = size_dict["layers"], size_dict["heads"], size_dict["width"]
        self.width = width
        
        self.token_channels = token_channels
        self.sem_embed_dim = sem_embed_dim
        self.sem_cls_token = sem_cls_token
        
        # learnable embeddings
        scale = width ** -0.5
        if self.sem_cls_token:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len + 1, width))
        else:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))
        
        # token embedding
        self.token_embedding = nn.Linear(self.token_channels, width)
        
        # mask embedding
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, width))
        
        # transformer layers
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)
        
        # output layers
        self.out = nn.Linear(self.width, self.sem_embed_dim)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size, add_cls=sem_cls_token).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-SemDecoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")
    
    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None):
        """forward pass through Sem decoder."""            
        z = self.token_embedding(z_latents)
        bsz, seq_len, _ = z.shape

        if ids_restore is not None:
            num_mask_tokens = ids_restore.shape[1] + 1 - seq_len
            mask_tokens = self.mask_embedding.repeat(bsz, num_mask_tokens, 1)
            z_ = torch.cat([z, mask_tokens], dim=1)
            expanded_ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, z_.shape[-1])
            z = torch.gather(z_, dim=1, index=expanded_ids_restore)
        
        z = z + self.positional_embedding
        
        z = self.ln_pre(z)
        rope = self.rope_tensor.expand(bsz, -1, -1)
        for block in self.transformer:
            z = block(z, rope=rope)
        z = self.ln_post(z)
        
        return self.out(z)
    

# ================================
# MLP Decoder
# ================================


class MLPDecoder(nn.Module):
    """Sem decoder for training the model."""
    
    def __init__(
        self, 
        token_channels: int = 16,
        sem_embed_dim: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.token_channels = token_channels
        self.sem_embed_dim = sem_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.token_channels, self.token_channels * 4),
            nn.GELU(),
            nn.Linear(self.token_channels * 4, self.sem_embed_dim),
        )

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-SemDecoder] params: {params_M:.2f}M")
    
    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None):
        """forward pass through Sem decoder."""
        return self.mlp(z_latents)


# ================================
# Main RecTok Model
# ================================


class RecTok(nn.Module):
    """
    RecTok: latent denoising makes good visual tokenizers.
    """

    _logged = False

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        vit_enc_model_size: str = "base",
        vit_dec_model_size: str = "base",
        encoder_type: str = "default",
        vit_sem_model_size: str = "tiny",
        token_channels: int = 16,
        foundation_model_type: str = "",
        cls_token_type: str = "none",
        diff_cls_token: bool = False,
        sem_dec_type: str = "transformer",
        sem_input_type: str = "noisy",
        sem_target: str = "rec+align",
        mask_ratio: float = 0.4,
        mask_ratio_min: float = -0.1,
        mask_ratio_type: str = "random",
        gamma: float = 1.0,
        noise_schedule: str = "shift",
        disable_kl: bool = False,
        use_qknorm: bool = False,
        # normalization parameters used for generative model training
        mean=0.0,
        std=1.0,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        if len(kwargs) > 0:
            logger.warning(f"Unknown arguments: {kwargs}")
            
        # initialize encoder and decoder
        if encoder_type == "dinov3":
            self.encoder = DINOv3Encoder(
                pretrained_model_name_or_path="facebook/dinov3-vitb16-pretrain-lvd1689m",
                frozen_dinov3=False,
                img_size=img_size,
                token_channels=token_channels,
            )
        else:
            self.encoder = Encoder(
                img_size=img_size,
                patch_size=patch_size,
                model_size=vit_enc_model_size,
                token_channels=token_channels,
                mask_ratio=mask_ratio,
                mask_ratio_min=mask_ratio_min,
                random_mask_ratio=mask_ratio_type.lower() == "random",
                cls_token_type=cls_token_type,
                diff_cls_token=diff_cls_token,
                disable_kl=disable_kl,
                use_qknorm=use_qknorm,
            )
            
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            model_size=vit_dec_model_size,
            token_channels=token_channels,
            diff_cls_token=diff_cls_token,
            use_qknorm=use_qknorm,
        )

        # model configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.seq_h = img_size // patch_size
        self.seq_w = self.seq_h
        self.width = self.encoder.width
        self.token_channels = token_channels
        self.gamma = gamma
        self.noise_schedule = noise_schedule
        self.scale_factor = scale_factor
        self.foundation_model_type = foundation_model_type
        self.sem_input_type = sem_input_type
        self.sem_target = sem_target
        self.cls_token_type = cls_token_type
        self.diff_cls_token = diff_cls_token
        self.disable_kl = disable_kl
        self.use_qknorm = use_qknorm
        
        self.timestep_shift = sqrt(self.seq_h * self.seq_w * self.token_channels / 4096)
        
        # initialize weights
        self.apply(self._init_weights)
        
        self.use_sem = False
        if foundation_model_type != "":
            self.use_sem = True
            sem_dec = SemDecoder_models[sem_dec_type]
            
            self.sem_foundation_models = nn.ModuleDict()
            self.sem_foundation_models_transforms = dict()
            self.sem_decoders = nn.ModuleDict()
            sem_token_channels = self.token_channels

            if "dinov2" in foundation_model_type:
                sem_foundation_model, transforms = create_foundation_model("dinov2")
                sem_foundation_model.eval()
                sem_foundation_model.requires_grad_(False)
                self.sem_foundation_models["dinov2"] = sem_foundation_model
                self.sem_foundation_models_transforms["dinov2"] = transforms
                
                self.sem_decoders["dinov2"] = sem_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_sem_model_size,
                    token_channels=sem_token_channels,
                    sem_embed_dim=sem_foundation_model.num_features,
                    sem_cls_token=cls_token_type != "none",
                )

            if "dinov3" in foundation_model_type:
                sem_foundation_model, transforms = create_foundation_model("dinov3")
                sem_foundation_model.eval()
                sem_foundation_model.requires_grad_(False)
                self.sem_foundation_models["dinov3"] = sem_foundation_model
                self.sem_foundation_models_transforms["dinov3"] = transforms
                
                self.sem_decoders["dinov3"] = sem_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_sem_model_size,
                    token_channels=sem_token_channels,
                    sem_embed_dim=sem_foundation_model.config.hidden_size,
                    sem_cls_token=cls_token_type != "none",
                )
            
            if "siglip" in foundation_model_type:
                sem_foundation_model, transforms = create_foundation_model("siglip")
                sem_foundation_model.eval()
                sem_foundation_model.requires_grad_(False)
                self.sem_foundation_models["siglip"] = sem_foundation_model
                self.sem_foundation_models_transforms["siglip"] = transforms
                
                self.sem_decoders["siglip"] = sem_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_sem_model_size,
                    token_channels=sem_token_channels,
                    sem_embed_dim=sem_foundation_model.num_features,
                )

        # setup to-posteriors function
        self.to_posteriors = partial(DiagonalGaussianDistribution, channel_dim=-1)

        # logging
        if not RecTok._logged:
            RecTok._logged = True
            logger.info(f"[RecTok] Gamma: {self.gamma}, Max Mask Ratio: {mask_ratio}")

        # setup normalization parameters
        if isinstance(mean, np.ndarray) or isinstance(mean, list):
            mean = np.array(mean).reshape(1, -1, 1, 1)
            std = np.array(std).reshape(1, -1, 1, 1)
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("std", torch.tensor(std), persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[RecTok] trainable params: {params_M:.2f}M")

    def _init_weights(self, module: nn.Module) -> None:
        """initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def freeze_everything_but_decoder(self) -> None:
        """freeze all parameters except the decoder, used for decoder fine-tuning"""
        for param in self.parameters():
            param.requires_grad = False

        if self.decoder is not None:
            for param in self.decoder.parameters():
                param.requires_grad = True

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[RecTok] trainable params: {params_M:.2f}M (after freezing all but decoder)")

    def reset_stats(self, mean: Tensor | np.ndarray | float, std: Tensor | np.ndarray | float) -> None:
        if isinstance(mean, float) and isinstance(std, float) or (mean.ndim == 0 and std.ndim == 0):
            # a single digit global mean and global std
            self.register_buffer("mean", _to_tensor(mean), persistent=False)
            self.register_buffer("std", _to_tensor(std), persistent=False)
        else:
            n_chans = mean.shape[-1]
            self.register_buffer("mean", _to_tensor(mean).reshape(1, 1, n_chans), persistent=False)
            self.register_buffer("std", _to_tensor(std).reshape(1, 1, n_chans), persistent=False)
        logger.info(f"Resetting mean and std ({mean.shape}, {std.shape})")
        logger.info(f"Mean: {self.mean}")
        logger.info(f"Std: {self.std}")

    def denormalize_z(self, z: Tensor) -> Tensor:
        """denormalize latent tokens."""
        return z * self.std.to(z) / self.scale_factor + self.mean.to(z)

    def normalize_z(self, z: Tensor) -> Tensor:
        """normalize latent tokens."""
        return (z - self.mean.to(z)) * self.scale_factor / self.std.to(z)

    def encode_into_posteriors(self, x: Tensor):
        """encode image into posterior distributions."""
        z = self.encoder(x)["z"]
        return self.to_posteriors(z)

    def encode(self, x: Tensor, sampling: bool = False, noise_level: float = -1.0):
        """encode image into latent tokens."""
        ret = self.encoder(x)
        z = ret["z"]    # [bsz, seq_len, token_channels] or [bsz, 1 + seq_len, token_channels] if cls_token_type is not none
        ids_restore = ret["ids_restore"]
        ids_keep = ret["ids_keep"]
        ids_masked = ret["ids_masked"]

        if self.disable_kl:
            posteriors = None
            z_latents = z
        else:
            posteriors = self.to_posteriors(z)
            z_latents = posteriors.sample() if sampling else posteriors.mean
            
        sem_input = z_latents   # clean latents

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            bsz, n_tokens, chans = z_latents.shape
            
            if self.noise_schedule == "lognorm":
                normal_samples = torch.randn(bsz, 1, 1, device=device)
                noise_level_tensor = torch.sigmoid(normal_samples)
            elif self.noise_schedule == "shift":
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
                noise_level_tensor = self.timestep_shift * noise_level_tensor / (1 + (self.timestep_shift - 1) * noise_level_tensor)
            elif self.noise_schedule == "uniform":
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
            else:
                raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
            
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise
                
            if self.sem_input_type == "noisy":
                sem_input = z_latents   # noisy latents
            
        if self.cls_token_type != "none" and not self.diff_cls_token:
            z_latents = z_latents[:, 1:]    # remove the cls token

        ret = dict(
            z_latents=z_latents,
            sem_input=sem_input,
            posteriors=posteriors,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
            ids_masked=ids_masked,
        )

        return ret

    def forward(self, x: Tensor):
        """forward pass through the entire model."""
        ret = self.encode(x, sampling=self.training)
        z_latents = ret["z_latents"]    # for pixel decoder
        sem_input = ret["sem_input"]    # for sem decoder
        posteriors = ret["posteriors"]
        ids_restore = ret["ids_restore"]
        ids_masked = ret["ids_masked"]

        if self.use_sem and self.training:
            x_sem = (x + 1) * 0.5
            sem_features = []
            pred_sem_features = []

            for i, model_type in enumerate(self.sem_foundation_models.keys()):
                sem_foundation_model = self.sem_foundation_models[model_type]
                transforms = self.sem_foundation_models_transforms[model_type]
                sem_decoder = self.sem_decoders[model_type]

                if model_type == "dinov2":
                    x_dino = transforms(x_sem)
                    dino_img_size = self.img_size / self.patch_size * 14   # DINOv2 uses patch size 14
                    x_dino = F.interpolate(
                        x_dino, 
                        size=(dino_img_size, dino_img_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    x_dino = x_dino.to(dtype=x.dtype)
                    with torch.inference_mode():
                        if self.cls_token_type != "none":
                            sem_feature = sem_foundation_model.forward_features(x_dino)   # [B, 257, dim]
                        else:
                            sem_feature = sem_foundation_model.forward_features(x_dino)[:, 1:]   # [B, 256, dim]

                elif model_type == "dinov3":
                    x_dinov3 = (x_sem * 255).to(torch.uint8)
                    inputs = transforms(x_dinov3, return_tensors="pt").to(x.device)
                    inputs["pixel_values"] = F.interpolate(
                        inputs["pixel_values"],
                        size=(self.img_size, self.img_size), 
                        mode="bilinear", 
                        align_corners=False
                    )
                    with torch.inference_mode():
                        sem_feature = sem_foundation_model(**inputs).last_hidden_state
                    
                    if self.cls_token_type != "none":
                        sem_feature = torch.cat(
                            [
                                sem_feature[:, 0, :].unsqueeze(1), # keep cls token for sem loss
                                sem_feature[:, 1 + sem_foundation_model.config.num_register_tokens:, :]
                            ], 
                            dim=1
                        )
                    else:
                        sem_feature = sem_feature[:, 1 + sem_foundation_model.config.num_register_tokens:, :]

                elif model_type == "siglip":
                    x_siglip = transforms(x_sem)
                    x_siglip = x_siglip.to(dtype=x.dtype)
                    with torch.inference_mode():
                        sem_feature = sem_foundation_model.forward_features(x_siglip)   # [B, 256, dim]

                else:
                    raise ValueError(f"Unknown foundation model type: {model_type}")
                
                pred_sem_feature = sem_decoder(sem_input, ids_restore=ids_restore)
            
                if self.sem_target == "reconstruction":
                    if ids_masked.shape[1] > 0:
                        expanded_ids_masked = ids_masked.unsqueeze(-1).expand(-1, -1, sem_feature.shape[-1])
                        sem_feature = torch.gather(sem_feature, dim=1, index=expanded_ids_masked)
                        pred_sem_feature = torch.gather(pred_sem_feature, dim=1, index=expanded_ids_masked)
                    else:
                        # do not update sem_decoder, use pseudo loss
                        pred_sem_feature = sem_feature.clone() + pred_sem_feature * 0
                
                sem_features.append(sem_feature)
                pred_sem_features.append(pred_sem_feature)

        else:
            sem_features = None
            pred_sem_features = None


        decoded = self.decoder(z_latents, ids_restore=ids_restore)  # [bsz, 3, img_size, img_size]

        result_dict = dict(
            posteriors=posteriors,
            ids_restore=ids_restore,
            sem_features=sem_features,
            pred_sem_features=pred_sem_features,
        )

        return decoded, result_dict

    def tokenize(self, x: Tensor, sampling: bool = False) -> Tensor:
        """tokenize input image and normalize the latent tokens."""
        ret = self.encode(x, sampling=sampling)
        z = ret["z_latents"]            
        z = self.normalize_z(z)
        
        if self.diff_cls_token:
            return z
        else:
            return rearrange(z, "b (h w) c -> b c h w", h=self.seq_h)

    def detokenize(self, z: Tensor) -> Tensor:
        """detokenize latent representation back to image."""
        if z.ndim == 4:
            z = rearrange(z, "b c h w -> b (h w) c")

        z = self.denormalize_z(z)
        decoded_images = self.decoder(z)
        return torch.clamp(decoded_images * 0.5 + 0.5, 0.0, 1.0)

    def sample_from_moments(self, moments: Tensor) -> Tensor:
        """sample from latent moments."""
        z = DiagonalGaussianDistribution(moments, channel_dim=-1).sample()
        z = self.normalize_z(z)
        return rearrange(z, "b (h w) c -> b c h w", h=self.seq_h)

    @torch.inference_mode()
    def reconstruct(self, x: Tensor) -> Tensor:
        """reconstruct input image."""
        return self.detokenize(self.tokenize(x))


# ================================
# Model Factory Functions
# ================================


def rectok_BB(**kwargs) -> RecTok:
    return RecTok(vit_enc_model_size="base", vit_dec_model_size="base", **kwargs)


# ================================
# Model Registry
# ================================

RecTok_models = {
    "rectok_BB": rectok_BB,
}

SemDecoder_models = {
    "transformer": TransformerDecoder,
    "mlp": MLPDecoder,
}