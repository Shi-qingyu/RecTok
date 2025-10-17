from typing import Optional
import logging
import random
from functools import partial
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

logger = logging.getLogger("DeTok")


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

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim % num_heads !=0, got {dim} and {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        bsz, n_ctx, ch = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads).unbind(0)
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
        norm_layer=partial(nn.RMSNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = norm_layer(dim), norm_layer(dim)
        self.attn = Attention(dim, num_heads)
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
        use_skip_connection: bool = False,
        last_layer_feature: bool = False,
        aux_cls_token: bool = False,
        pooling_cls_token: bool = False,
        diff_cls_token: bool = False,
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        # needs to split into mean and std
        self.token_channels = token_channels * 2
        self.mask_ratio = mask_ratio
        self.mask_ratio_min = mask_ratio_min
        self.random_mask_ratio = random_mask_ratio
        self.seq_len = self.grid_size ** 2
        self.use_skip_connection = use_skip_connection
        self.last_layer_feature = last_layer_feature
        self.aux_cls_token = aux_cls_token
        self.pooling_cls_token = pooling_cls_token
        self.diff_cls_token = diff_cls_token
        self.num_register_tokens = num_register_tokens

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
        if self.aux_cls_token and not self.pooling_cls_token:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.num_register_tokens + 1 + 256, width))
        else:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.num_register_tokens + 256, width))

        if self.aux_cls_token and not self.pooling_cls_token:
            self.aux_cls_token_embedding = nn.Parameter(scale * torch.randn(1, 1, width))
            
        if self.num_register_tokens > 0:
            self.register_token_embedding = nn.Parameter(scale * torch.randn(1, self.num_register_tokens, width))

        # transformer layers
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)
        self.latent_head = nn.Linear(width, self.token_channels)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size, add_cls=self.aux_cls_token and not self.pooling_cls_token, n_register=self.num_register_tokens).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-Encoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}, random mask ratio: {self.random_mask_ratio}")

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

    def show_attention_map_last_layer(self, x: Tensor):
        """show attention map of last layer."""
        x = self.patch_embed(x)
        if self.aux_cls_token and not self.pooling_cls_token:
            x = torch.cat([self.aux_cls_token_embedding.expand(x.shape[0], -1, -1), x], dim=1)
        if self.num_register_tokens > 0:
            x = torch.cat([self.register_token_embedding.expand(x.shape[0], -1, -1), x], dim=1)

        x = x + self.positional_embedding

        x = self.ln_pre(x)
        for block in self.transformer[:-1]:
            x = block(x, self.rope_tensor.expand(x.shape[0], -1, -1))

        last_block = self.transformer[-1]
        x = last_block.norm1(x)
        bsz, n_ctx, ch = x.shape
        qkv = last_block.attn.qkv(x)
        q, k, v = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=last_block.attn.num_heads).unbind(0)  # [bsz, heads, seq_len, head_dim]
        q, k = apply_rotary_emb(q, self.rope_tensor.expand(bsz, -1, -1)), apply_rotary_emb(k, self.rope_tensor.expand(bsz, -1, -1))
        attn_map = (q @ k.transpose(-2, -1)) * last_block.attn.head_dim ** -0.5
        attn_map = attn_map.softmax(dim=-1).mean(dim=1)
        if self.aux_cls_token and not self.pooling_cls_token:
            attn_map_cls = attn_map[:, self.num_register_tokens, 1 + self.num_register_tokens:]
        else:
            attn_map_cls = attn_map[:, self.num_register_tokens:, self.num_register_tokens:].mean(dim=1)
        return attn_map_cls

    def forward(self, x: Tensor):
        """forward pass through encoder."""
        if self.use_skip_connection:
            x_skip = x
            x_skip = F.pixel_unshuffle(x_skip, self.patch_size)
            x_skip = x_skip.flatten(2).transpose(1, 2) # [bsz, seq_len, chans]
            num_chunks = x_skip.shape[-1] // self.width
            assert num_chunks * self.width == x_skip.shape[-1], f"num_chunks * width != chans, got {num_chunks} and {self.width}"
            x_skip = x_skip.unflatten(-1, (self.width, num_chunks)).mean(dim=-1)
            x = self.patch_embed(x) + x_skip
        else:
            x = self.patch_embed(x)
        
        if self.aux_cls_token and not self.pooling_cls_token:
            x = torch.cat([self.aux_cls_token_embedding.expand(x.shape[0], -1, -1), x], dim=1)
        if self.num_register_tokens > 0:
            x = torch.cat([self.register_token_embedding.expand(x.shape[0], -1, -1), x], dim=1)

        if x.shape[1] != self.positional_embedding.shape[1]:
            # [1, seq_len, width] -> [1, x.shape[1], width]
            pos = self.positional_embedding.permute(0, 2, 1)
            position_embedding = F.interpolate(pos, size=x.shape[1], mode="linear")
            position_embedding = position_embedding.permute(0, 2, 1)
        else:
            position_embedding = self.positional_embedding
        x = x + position_embedding
            
        x, _, ids_restore, rope, ids_keep, ids_masked = self.mae_random_masking(x)

        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x, rope)
        
        if self.last_layer_feature:
            z_aux = x
            x = self.ln_post(x)
            z = self.latent_head(x)    # [bsz, seq_len, token_channels]
        else:
            x = self.ln_post(x)
            z = self.latent_head(x)
            z_aux = z
        
        if self.num_register_tokens > 0:
            z = z[:, self.num_register_tokens:]
            z_aux = z_aux[:, self.num_register_tokens:]

        if self.aux_cls_token:
            if self.pooling_cls_token:
                z_cls = z.mean(1).unsqueeze(1)
                z_aux_cls = z_aux.mean(1).unsqueeze(1)
                z_aux = torch.cat([z_aux_cls, z_aux], dim=1)
            else:
                z_cls = z[:, 0].unsqueeze(1)
                z = z[:, 1:]
        
        if self.diff_cls_token:
            z = torch.cat([z_cls, z], dim=1)

        ret = dict(
            z=z,    # [bsz, seq_len + 1, dim] if diff_cls_token, [bsz, seq_len, dim] otherwise
            z_aux=z_aux,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
            ids_masked=ids_masked,
        )

        return ret
    

class DualEncoder(nn.Module):
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
        last_layer_feature: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        # needs to split into mean and std
        self.token_channels = token_channels * 2
        self.mask_ratio = mask_ratio
        self.mask_ratio_min = mask_ratio_min
        self.random_mask_ratio = random_mask_ratio
        self.seq_len = self.grid_size**2
        
        self.last_layer_feature = last_layer_feature

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
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))

        # transformer layers
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)
        self.latent_head = nn.Linear(width, self.token_channels)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-Encoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}, random mask ratio: {self.random_mask_ratio}")

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
        x_full = self.patch_embed(x) + self.positional_embedding
        
        if self.training:
            x_visible, _, ids_restore, rope_visible, ids_keep, ids_masked = self.mae_random_masking(x_full)
            
            x_visible = self.ln_pre(x_visible)
            for block in self.transformer:
                x_visible = block(x_visible, rope_visible)
            x_visible = self.ln_post(x_visible)
            
            if self.last_layer_feature:
                z_aux = x_visible
            else:
                z_aux = self.latent_head(x_visible)    # [bsz, visible_seq_len, token_channels]
        else:
            z_aux = None
            ids_restore = None
            ids_keep = None
            ids_masked = None
        
        bsz = x_full.shape[0]
        rope_full = self.rope_tensor.expand(bsz, -1, -1)
        
        x_full = self.ln_pre(x_full)
        for block in self.transformer:
            x_full = block(x_full, rope_full)
        x_full = self.ln_post(x_full)
        
        z = self.latent_head(x_full)
        
        ret = dict(
            z=z,
            z_aux=z_aux,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
            ids_masked=ids_masked,
        )

        return ret
    
    
class PostMaskEncoder(nn.Module):
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
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        # needs to split into mean and std
        self.token_channels = token_channels * 2
        self.mask_ratio = mask_ratio
        self.mask_ratio_min = mask_ratio_min
        self.random_mask_ratio = random_mask_ratio
        self.seq_len = self.grid_size**2

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
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))

        # transformer layers
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer) for _ in range(num_layers)]
        )
        self.ln_post = norm_layer(width)
        self.latent_head = nn.Linear(width, self.token_channels)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-Encoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}, random mask ratio: {self.random_mask_ratio}")

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
            return x, torch.zeros(bsz, seq_len, device=x.device), None, None, None

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

        mask = torch.ones(bsz, seq_len, device=x.device)
        mask[:, :len_keep] = 0
        # ids_restore[:, i] >= len_keep means ith token in the original sequence is masked
        mask = torch.gather(mask, dim=1, index=ids_restore) # mask[i, j] = mask[i, ids_restore[i, j]]
        return x_visible, mask, ids_restore, ids_keep, ids_masked

    def forward(self, x: Tensor):
        """forward pass through encoder."""
        x = self.patch_embed(x) + self.positional_embedding
        bsz = x.shape[0]
        rope = self.rope_tensor.expand(bsz, -1, -1)

        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x, rope)
        x = self.ln_post(x)

        z = self.latent_head(x)    # [bsz, seq_len, token_channels]
        z, mask, ids_restore, ids_keep, ids_masked = self.mae_random_masking(z)
        
        ret = dict(
            z=z,
            z_aux=None,
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
        frozen_dinov3: bool = True,
        img_size: int = 256,
        token_channels: int = 16,
        **kwargs
    ) -> None:
        super().__init__()
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
        num_register_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = self.img_size // self.patch_size
        self.model_size = model_size
        self.token_channels = token_channels
        self.seq_len = self.grid_size ** 2
        self.num_register_tokens = num_register_tokens
        self.diff_cls_token = diff_cls_token
        
        params = SIZE_DICT[self.model_size]
        num_layers, num_heads, width = params["layers"], params["heads"], params["width"]

        # learnable embeddings
        scale = width ** -0.5
        if self.num_register_tokens > 0:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.num_register_tokens + 256, width))
            self.register_token_embedding = nn.Parameter(scale * torch.randn(1, self.num_register_tokens, width))
        else:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, 256, width))
        
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, width))

        # decoder layers
        self.decoder_embed = nn.Linear(self.token_channels, width)
        norm_layer = partial(nn.RMSNorm, eps=1e-6)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList(
            [Block(dim=width, num_heads=num_heads, norm_layer=norm_layer) for _ in range(num_layers)]
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
            n_register=self.num_register_tokens, 
            add_cls=False
        ).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-Decoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")

    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None) -> Tensor:
        """forward pass through decoder."""
        # z_latents: [bsz, seq_len, token_channels] or [bsz, 1 + seq_len, token_channels] if diff_cls_token
        if self.diff_cls_token:
            # we never send cls token to decoder
            z_latents = z_latents[:, 1:]

        z = self.decoder_embed(z_latents)
        bsz, seq_len, _ = z.shape

        if ids_restore is not None:
            num_mask_tokens = ids_restore.shape[1] + 1 - seq_len
            mask_tokens = self.mask_token.repeat(bsz, num_mask_tokens, 1)
            z_ = torch.cat([z, mask_tokens], dim=1)
            expanded_ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, z_.shape[-1])
            z = torch.gather(z_, dim=1, index=expanded_ids_restore)
            
        if self.num_register_tokens > 0:
            z = torch.cat([self.register_token_embedding.expand(z.shape[0], -1, -1), z], dim=1)
            
        if z.shape[1] != self.positional_embedding.shape[1]:
            pos = self.positional_embedding.permute(0, 2, 1)
            position_embedding = F.interpolate(pos, size=z.shape[1], mode="linear")
            position_embedding = position_embedding.permute(0, 2, 1)
        else:
            position_embedding = self.positional_embedding
        z = z + position_embedding

        z = self.ln_pre(z)
        rope = self.rope_tensor.expand(bsz, -1, -1)
        for block in self.transformer:
            z = block(z, rope)
        z = self.ln_post(z)

        if self.num_register_tokens > 0:
            z = z[:, self.num_register_tokens:]

        z = self.ffn(z)  # embed -> patch
        z = self.conv_out(z)  # final 3x3 conv

        return z


# ================================
# Auxiliary Decoder
# ================================


class TransformerDecoder(nn.Module):
    """auxiliary decoder for training the model."""
    
    def __init__(
        self, 
        img_size: int = 256,
        patch_size: int = 16,
        model_size: str = "base", 
        token_channels: int = 16,
        aux_embed_dim: int = 1024,
        aux_cls_token: bool = False,
        use_adaptive_channels: bool = False,
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
        self.aux_embed_dim = aux_embed_dim
        self.aux_cls_token = aux_cls_token
        self.use_adaptive_channels = use_adaptive_channels
        
        # learnable embeddings
        scale = width ** -0.5
        if self.aux_cls_token:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len + 1, width))
        else:
            self.positional_embedding = nn.Parameter(scale * torch.randn(1, self.seq_len, width))
        
        # token embedding
        if self.use_adaptive_channels:
            self.token_embedding = nn.Sequential(
                nn.Linear(self.token_channels, 16),
                nn.SiLU(),
                nn.Linear(16, width),
            )
        else:
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
        self.out = nn.Linear(self.width, self.aux_embed_dim)

        # rotary position embedding
        head_dim = self.transformer[0].attn.head_dim
        rope_tensor = get_rope_tensor(head_dim, self.grid_size, self.grid_size, add_cls=aux_cls_token).unsqueeze(0)
        self.register_buffer("rope_tensor", rope_tensor, persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-AuxiliaryDecoder] params: {params_M:.2f}M, {model_size}-{num_layers}-{width}")
    
    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None):
        """forward pass through auxiliary decoder."""            
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
    """auxiliary decoder for training the model."""
    
    def __init__(
        self, 
        token_channels: int = 16,
        aux_embed_dim: int = 1024,
        **kwargs,
    ):
        super().__init__()
        self.token_channels = token_channels
        self.aux_embed_dim = aux_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.token_channels, 4 * self.aux_embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.aux_embed_dim, self.aux_embed_dim),
        )

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok-AuxiliaryDecoder] params: {params_M:.2f}M")
    
    def forward(self, z_latents: Tensor, ids_restore: Tensor | None = None):
        """forward pass through auxiliary decoder."""
        return self.mlp(z_latents)


# ================================
# Main DeTok Model
# ================================


class DeTok(nn.Module):
    """
    l-DeTok: latent denoising makes good visual tokenizers.
    """

    _logged = False

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        vit_enc_model_size: str = "small",
        pretrained_model_name_or_path: str = "",
        frozen_dinov3: bool = True,
        num_register_tokens: int = 0,
        vit_dec_model_size: str = "base",
        vit_aux_model_size: str = "tiny",
        token_channels: int = 16,
        use_adaptive_channels: bool = False,
        last_layer_feature: bool = False,
        vf_model_type: str = "",
        aux_model_type: str = "",
        aux_cls_token: bool = False,
        pooling_cls_token: bool = False,
        diff_cls_token: bool = False,
        aux_dec_type: str = "transformer",
        aux_input_type: str = "noisy",
        aux_target: str = "reconstruction",
        mask_ratio: float = 0.7,
        mask_ratio_min: float = -0.1,
        mask_ratio_type: str = "random",
        use_skip_connection: bool = False,
        gamma: float = 3.0,
        use_additive_noise: bool = False,
        # normalization parameters used for generative model training
        mean=0.0,
        std=1.0,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        # initialize encoder and decoder
        if "dinov3" in pretrained_model_name_or_path:
            self.encoder = DINOv3Encoder(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                frozen_dinov3=frozen_dinov3,
                img_size=img_size,
                token_channels=token_channels,
                num_register_tokens=num_register_tokens,
            )
        elif "postmask" in pretrained_model_name_or_path:
            self.encoder = PostMaskEncoder(
                img_size=img_size,
                patch_size=patch_size,
                model_size=vit_enc_model_size,
                token_channels=token_channels,
                mask_ratio=mask_ratio,
                mask_ratio_min=mask_ratio_min,
                random_mask_ratio=mask_ratio_type.lower() == "random",
            )
        elif "dual" in pretrained_model_name_or_path:
            self.encoder = DualEncoder(
                img_size=img_size,
                patch_size=patch_size,
                model_size=vit_enc_model_size,
                token_channels=token_channels,
                mask_ratio=mask_ratio,
                mask_ratio_min=mask_ratio_min,
                random_mask_ratio=mask_ratio_type.lower() == "random",
                last_layer_feature=last_layer_feature,
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
                use_skip_connection=use_skip_connection,
                last_layer_feature=last_layer_feature,
                aux_cls_token=aux_cls_token,
                pooling_cls_token=pooling_cls_token,
                diff_cls_token=diff_cls_token,
                num_register_tokens=0,
            )
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            model_size=vit_dec_model_size,
            token_channels=token_channels,
            diff_cls_token=diff_cls_token,
            num_register_tokens=0,
        )

        # model configuration
        self.seq_h = img_size // patch_size
        self.seq_w = self.seq_h
        self.width = self.encoder.width
        self.token_channels = token_channels
        self.use_additive_noise = use_additive_noise
        self.gamma = gamma
        self.scale_factor = scale_factor
        self.vf_model_type = vf_model_type
        self.aux_model_type = aux_model_type
        self.use_adaptive_channels = use_adaptive_channels
        self.aux_input_type = aux_input_type
        self.aux_target = aux_target
        self.aux_cls_token = aux_cls_token
        self.pooling_cls_token = pooling_cls_token
        self.diff_cls_token = diff_cls_token
        
        # initialize weights
        self.apply(self._init_weights)

        # initialize vf loss
        self.use_vf = False
        if vf_model_type != "":
            self.use_vf = True
            if vf_model_type == "dinov2":
                self.foundation_model = create_foundation_model(vf_model_type)[0]
                self.foundation_model.eval()
                self.foundation_model.requires_grad_(False)

                self.vf_feature_dim = self.foundation_model.num_features
                self.linear_proj = nn.Linear(int(self.vf_feature_dim), self.token_channels)
            else:
                raise ValueError(f"Unknown foundation model type: {vf_model_type}")
        
        self.use_aux = False
        if aux_model_type != "":
            self.use_aux = True
            aux_dec = AuxiliaryDecoder_models[aux_dec_type]
            
            self.aux_foundation_models = nn.ModuleDict()
            self.aux_foundation_models_transforms = dict()
            self.aux_decoders = nn.ModuleDict()

            if last_layer_feature:
                aux_token_channels = self.width
            else:
                aux_token_channels = self.token_channels

            if "dinov2" in aux_model_type:
                aux_foundation_model, transforms = create_foundation_model("dinov2")
                aux_foundation_model.eval()
                aux_foundation_model.requires_grad_(False)
                self.aux_foundation_models["dinov2"] = aux_foundation_model
                self.aux_foundation_models_transforms["dinov2"] = transforms
                
                self.aux_decoders["dinov2"] = aux_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_aux_model_size,
                    token_channels=aux_token_channels,
                    aux_embed_dim=aux_foundation_model.num_features,
                    aux_cls_token=aux_cls_token,
                    pooling_cls_token=pooling_cls_token,
                    use_adaptive_channels=use_adaptive_channels,
                )

            if "dinov3" in aux_model_type:
                aux_foundation_model, transforms = create_foundation_model("dinov3")
                aux_foundation_model.eval()
                aux_foundation_model.requires_grad_(False)
                self.aux_foundation_models["dinov3"] = aux_foundation_model
                self.aux_foundation_models_transforms["dinov3"] = transforms
                
                self.aux_decoders["dinov3"] = aux_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_aux_model_size,
                    token_channels=aux_token_channels,
                    aux_embed_dim=aux_foundation_model.config.hidden_size,
                    aux_cls_token=aux_cls_token,
                    pooling_cls_token=pooling_cls_token,
                    use_adaptive_channels=use_adaptive_channels,
                )
            
            if "sam" in aux_model_type:
                aux_foundation_model, transforms = create_foundation_model("sam")
                aux_foundation_model.eval()
                aux_foundation_model.requires_grad_(False)
                self.aux_foundation_models["sam"] = aux_foundation_model
                self.aux_foundation_models_transforms["sam"] = transforms
                
                self.aux_decoders["sam"] = aux_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_aux_model_size,
                    token_channels=aux_token_channels,
                    aux_embed_dim=aux_foundation_model.vision_encoder.config.output_channels,
                    aux_cls_token=aux_cls_token,
                    pooling_cls_token=pooling_cls_token,
                )
            
            if "radio" in aux_model_type:
                aux_foundation_model, transforms = create_foundation_model("radio")
                aux_foundation_model.eval()
                aux_foundation_model.requires_grad_(False)
                self.aux_foundation_models["radio"] = aux_foundation_model
                self.aux_foundation_models_transforms["radio"] = transforms
                
                self.aux_decoders["radio"] = aux_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_aux_model_size,
                    token_channels=aux_token_channels,
                    aux_embed_dim=1024,
                    aux_cls_token=aux_cls_token,
                    pooling_cls_token=pooling_cls_token,
                )
            
            if "siglip" in aux_model_type:
                aux_foundation_model, transforms = create_foundation_model("siglip")
                aux_foundation_model.eval()
                aux_foundation_model.requires_grad_(False)
                self.aux_foundation_models["siglip"] = aux_foundation_model
                self.aux_foundation_models_transforms["siglip"] = transforms
                
                self.aux_decoders["siglip"] = aux_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_aux_model_size,
                    token_channels=aux_token_channels,
                    aux_embed_dim=aux_foundation_model.num_features,
                    aux_cls_token=aux_cls_token,
                    pooling_cls_token=pooling_cls_token,
                )
                
            if "detok_BB" in aux_model_type:
                aux_foundation_model, transforms = create_foundation_model("detok_BB")
                aux_foundation_model.eval()
                aux_foundation_model.requires_grad_(False)
                self.aux_foundation_models["detok_BB"] = aux_foundation_model
                self.aux_foundation_models_transforms["detok_BB"] = transforms
                
                self.aux_decoders["detok_BB"] = aux_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_aux_model_size,
                    token_channels=aux_token_channels,
                    aux_embed_dim=16,
                    aux_cls_token=False,
                    pooling_cls_token=False,
                )

            if "pixel" in aux_model_type:
                self.aux_decoders["pixel"] = aux_dec(
                    img_size=img_size,
                    patch_size=patch_size,
                    model_size=vit_aux_model_size,
                    token_channels=aux_token_channels,
                    aux_embed_dim=3,
                    aux_cls_token=aux_cls_token,
                    pooling_cls_token=pooling_cls_token,
                )

        # setup to-posteriors function
        self.to_posteriors = partial(DiagonalGaussianDistribution, channel_dim=-1)

        # logging
        if not DeTok._logged:
            DeTok._logged = True
            logger.info(f"[DeTok] Gamma: {self.gamma}, Max Mask Ratio: {mask_ratio}")

        # setup normalization parameters
        if isinstance(mean, np.ndarray) or isinstance(mean, list):
            mean = np.array(mean).reshape(1, -1, 1, 1)
            std = np.array(std).reshape(1, -1, 1, 1)
        self.register_buffer("mean", torch.tensor(mean), persistent=False)
        self.register_buffer("std", torch.tensor(std), persistent=False)

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] params: {params_M:.2f}M")

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

        for param in self.decoder.parameters():
            param.requires_grad = True

        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] trainable params: {params_M:.2f}M (after freezing all but decoder)")
        
    def freeze_encoder(self) -> None:
        """freeze all parameters except the latent head, used for latent head fine-tuning"""
        for name, param in self.encoder.named_parameters():
            if "latent_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        params_M = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[DeTok] trainable params: {params_M:.2f}M (after freezing encoder)")

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
        z = ret["z"]
        ids_restore = ret["ids_restore"]
        ids_keep = ret["ids_keep"]
        ids_masked = ret["ids_masked"]

        posteriors = self.to_posteriors(z)
        z_latents = posteriors.sample() if sampling else posteriors.mean
        
        if isinstance(self.encoder, DualEncoder):
            if self.encoder.last_layer_feature:
                z_latents_aux = ret["z_aux"]
            else:
                z_aux = ret["z_aux"]
                posteriors_aux = self.to_posteriors(z_aux)
                z_latents_aux = posteriors_aux.sample() if sampling else posteriors_aux.mean
        else:
            z_latents_aux = ret["z_aux"]
            if not self.encoder.last_layer_feature:
                posteriors_aux = self.to_posteriors(z_latents_aux)
                z_latents_aux = posteriors_aux.sample() if sampling else posteriors_aux.mean

        if self.training and self.gamma > 0.0:
            device = z_latents.device
            bsz, n_tokens, chans = z_latents.shape
            if noise_level > 0.0:
                noise_level_tensor = torch.full((bsz, 1, 1), noise_level, device=device)
            else:
                # noise_level_tensor = torch.rand(bsz, 1, 1, device=device)

                # Sample noise level using logit normal distribution for better control
                # Generate from normal distribution and apply sigmoid to get values in (0,1)
                normal_samples = torch.randn(bsz, 1, 1, device=device)
                noise_level_tensor = torch.sigmoid(normal_samples)
                
            noise_level_tensor = noise_level_tensor.expand(-1, n_tokens, chans)
            noise = torch.randn(bsz, n_tokens, chans, device=device) * self.gamma
            if self.use_additive_noise:
                z_latents = z_latents + noise_level_tensor * noise
            else:
                z_latents = (1 - noise_level_tensor) * z_latents + noise_level_tensor * noise
                
            if self.aux_input_type == "noisy":
                noise_level_tensor = torch.rand(bsz, 1, 1, device=device)
                noise_aux = torch.randn_like(z_latents_aux) * self.gamma
                z_latents_aux = (1 - noise_level_tensor) * z_latents_aux + noise_level_tensor * noise_aux

        ret = dict(
            z_latents=z_latents,
            z_latents_aux=z_latents_aux,
            posteriors=posteriors,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
            ids_masked=ids_masked,
        )

        return ret

    def forward(self, x: Tensor):
        """forward pass through the entire model."""
        ret = self.encode(x, sampling=self.training)
        z_latents = ret["z_latents"]
        z_latents_aux = ret["z_latents_aux"]
        posteriors = ret["posteriors"]
        ids_restore = ret["ids_restore"]
        ids_keep = ret["ids_keep"]
        ids_masked = ret["ids_masked"]

        if self.use_aux and self.training:
            x_aux = (x + 1) * 0.5
            aux_features = []
            pred_aux_features = []

            for i, model_type in enumerate(self.aux_foundation_models.keys()):
                aux_foundation_model = self.aux_foundation_models[model_type]
                transforms = self.aux_foundation_models_transforms[model_type]
                aux_decoder = self.aux_decoders[model_type]

                if model_type == "dinov2":
                    x_dino = transforms(x_aux)
                    x_dino = F.interpolate(x_dino, size=(224, 224), mode='bilinear', align_corners=False)
                    x_dino = x_dino.to(dtype=x.dtype)
                    with torch.inference_mode():
                        if self.encoder.aux_cls_token:
                            aux_feature = aux_foundation_model.forward_features(x_dino)   # [B, 257, dim]
                        else:
                            aux_feature = aux_foundation_model.forward_features(x_dino)[:, 1:]   # [B, 256, dim]

                elif model_type == "dinov3":
                    x_dinov3 = (x_aux * 255).to(torch.uint8)
                    inputs = transforms(x_dinov3, return_tensors="pt").to(x.device)
                    inputs["pixel_values"] = F.interpolate(
                        inputs["pixel_values"],
                        size=(256, 256), 
                        mode="bilinear", 
                        align_corners=False
                    )
                    with torch.inference_mode():
                        aux_feature = aux_foundation_model(**inputs).last_hidden_state
                    
                    if self.encoder.aux_cls_token:
                        aux_feature = torch.cat([aux_feature[:, 0, :].unsqueeze(1), aux_feature[:, 1 + aux_foundation_model.config.num_register_tokens:, :]], dim=1)
                    else:
                        aux_feature = aux_feature[:, 1 + aux_foundation_model.config.num_register_tokens:, :]

                elif model_type == "sam":
                    x_sam = transforms(
                        images=x_aux,
                        do_resize=True,
                        size={"longest_edge": 256},
                        do_pad=True,
                        pad_size={"height": 256, "width": 256},
                        input_data_format="channels_first",
                        do_rescale=False,
                        do_normalize=True,
                        return_tensors="pt",
                    )
                    pixel_values = x_sam["pixel_values"].to(dtype=x.dtype, device=x.device)
                    with torch.inference_mode():
                        aux_feature = aux_foundation_model(pixel_values)

                    aux_feature = aux_feature.last_hidden_state
                    B, C, H, W = aux_feature.shape
                    aux_feature = aux_feature.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

                elif model_type == "radio":
                    x_radio = x_aux
                    x_radio = x_radio.to(dtype=x.dtype, device=x.device)
                    with torch.inference_mode():
                        _, aux_feature = aux_foundation_model(x_radio)   # [B, 256, dim]

                elif model_type == "siglip":
                    x_siglip = transforms(x_aux)
                    x_siglip = x_siglip.to(dtype=x.dtype)
                    with torch.inference_mode():
                        aux_feature = aux_foundation_model.forward_features(x_siglip)   # [B, 256, dim]
                
                elif model_type == "detok_BB":
                    x_detok = x
                    with torch.inference_mode():
                        aux_feature = aux_foundation_model.encode(x_detok, sampling=False)["z_latents"]
                    
                elif model_type == "pixel":
                    aux_feature = x_aux

                else:
                    raise ValueError(f"Unknown foundation model type: {model_type}")
                
                if model_type == "detok_BB" and self.encoder.aux_cls_token:
                    pred_aux_feature = aux_decoder(z_latents_aux[:, 1:], ids_restore=ids_restore)
                else:
                    pred_aux_feature = aux_decoder(z_latents_aux, ids_restore=ids_restore)
                
                if aux_feature.shape[1] != pred_aux_feature.shape[1]:
                    bsz, seq_len, dim = aux_feature.shape
                    aux_feature_h = int(seq_len ** 0.5)
                    aux_feature = aux_feature.reshape(bsz, aux_feature_h, aux_feature_h, dim).permute(0, 3, 1, 2)
                    aux_feature = F.interpolate(aux_feature, size=(self.seq_h, self.seq_w), mode='bilinear', align_corners=False)
                    aux_feature = aux_feature.permute(0, 2, 3, 1).reshape(bsz, self.seq_h * self.seq_w, dim)

                if self.aux_target == "reconstruction":
                    if ids_masked.shape[1] > 0:
                        expanded_ids_masked = ids_masked.unsqueeze(-1).expand(-1, -1, aux_feature.shape[-1])
                        aux_feature = torch.gather(aux_feature, dim=1, index=expanded_ids_masked)
                        pred_aux_feature = torch.gather(pred_aux_feature, dim=1, index=expanded_ids_masked)
                    else:
                        # do not update aux_decoder, use pseudo loss
                        pred_aux_feature = aux_feature.clone() + pred_aux_feature * 0
                
                aux_features.append(aux_feature)
                pred_aux_features.append(pred_aux_feature)

        else:
            aux_features = None
            pred_aux_features = None

        if isinstance(self.encoder, DualEncoder):
            decoded = self.decoder(z_latents[:z_latents.shape[0] // 4], ids_restore=None)
        else:
            decoded = self.decoder(z_latents, ids_restore=ids_restore)  # [bsz, 3, img_size, img_size]

        result_dict = dict(
            posteriors=posteriors,
            z_latents=z_latents,
            ids_restore=ids_restore,
            vf_feature=None,
            aux_features=aux_features,
            pred_aux_features=pred_aux_features,
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


def detok_SS(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="small", vit_dec_model_size="small", **kwargs)


def detok_SB(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="small", vit_dec_model_size="base", **kwargs)


def detok_SL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="small", vit_dec_model_size="large", **kwargs)


def detok_BS(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="base", vit_dec_model_size="small", **kwargs)


def detok_BB(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="base", vit_dec_model_size="base", **kwargs)


def detok_BL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="base", vit_dec_model_size="large", **kwargs)


def detok_LS(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="large", vit_dec_model_size="small", **kwargs)


def detok_LB(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="large", vit_dec_model_size="base", **kwargs)


def detok_LL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="large", vit_dec_model_size="large", **kwargs)


def detok_XLXL(**kwargs) -> DeTok:
    return DeTok(vit_enc_model_size="xl", vit_dec_model_size="xl", **kwargs)


# ================================
# Model Registry
# ================================

DeTok_models = {
    "detok_SS": detok_SS,
    "detok_SB": detok_SB,
    "detok_SL": detok_SL,
    "detok_BS": detok_BS,
    "detok_BB": detok_BB,
    "detok_BL": detok_BL,
    "detok_LS": detok_LS,
    "detok_LB": detok_LB,
    "detok_LL": detok_LL,
    "detok_XLXL": detok_XLXL,
}

AuxiliaryDecoder_models = {
    "transformer": TransformerDecoder,
    "mlp": MLPDecoder,
}