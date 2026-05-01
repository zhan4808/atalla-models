"""Minimal ViT-style stack: LayerNorm, single-head attention, GELU FFN.

LayerNorm needs ``dim % 32 == 0``. Attention is ``atalla_sdpa`` (graph leaf); the
fused flash kernel in the emitter matches ``n_tokens=dim=32`` (input ``(1, 32, 32)``) when
``ATALLA_SDPA_FLASH=1``. Other shapes use a reference path in validate until a parameterized kernel exists.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.atalla_ops import AtallaSdpa


class ViTMicro(nn.Module):
    def __init__(self, dim: int = 32, n_tokens: int = 32):
        super().__init__()
        if dim % 32 != 0:
            raise ValueError("ViTMicro uses dim divisible by 32 for hardware LayerNorm.")
        self.dim = dim
        self.n_tokens = n_tokens
        self.patch_embed = nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        hidden = dim * 2
        self.ff1 = nn.Linear(dim, hidden)
        self.ff2 = nn.Linear(hidden, dim)
        self.sdpa = AtallaSdpa(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x = self.patch_embed(x) + self.pos_embed
        xa = self.ln1(x)
        q = self.q_proj(xa)
        k = self.k_proj(xa)
        v = self.v_proj(xa)
        mixed = self.sdpa(q, k, v)
        x = x + self.attn_proj(mixed)
        xb = self.ln2(x)
        h = self.ff1(xb)
        h = F.gelu(h, approximate="tanh")
        x = x + self.ff2(h)
        return x
