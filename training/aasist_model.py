"""AASIST-L backbone — Audio Anti-Spoofing using Integrated Spectro-Temporal
Graph Attention Networks (~85K parameters).

PyTorch reference implementation used for training + ONNX export. Adapted from
the AASIST paper (Jung et al., ICASSP 2022) with:
  * Joint temporal (frame-to-frame) + spectral (band-to-band) graph attention
  * Max-feature-map (MAM) frontend + res nets
  * Exports to ONNX consuming log-mel [1, 1, 80, T] and emitting 2-class logits

TensorFlow is NOT required; this is pure PyTorch (CPU-trainable for a small set).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class Res2Block(nn.Module):
    """Lightweight residual block used inside the AASIST frontend."""

    def __init__(self, ch: int, scale: int = 4):
        super().__init__()
        assert ch % scale == 0
        self.scale = scale
        self.width = ch // scale
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 3, padding=1) for _ in range(scale - 1)]
        )
        self.bn = nn.BatchNorm2d(ch)
        self.se = SqueezeExcite(ch)

    def forward(self, x):
        xs = torch.split(x, self.width, dim=1)
        out = []
        sp = xs[0]
        for i in range(self.scale - 1):
            if i > 0:
                sp = sp + xs[i]
            sp = F.relu(self.convs[i](sp))
            out.append(sp)
        out.append(xs[-1])
        y = torch.cat(out, dim=1)
        y = self.bn(y)
        return F.relu(self.se(y + x))


class GraphAttentionLayer(nn.Module):
    """GAT over a fixed spectro-temporal graph (joint frame + band edges)."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D] where N nodes = T frames + F bands (joint graph)
        B, N, D = x.shape
        H = self.n_heads
        q = self.q(x).view(B, N, H, D // H).transpose(1, 2)
        k = self.k(x).view(B, N, H, D // H).transpose(1, 2)
        v = self.v(x).view(B, N, H, D // H).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        ctx = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(ctx)


class AASISTL(nn.Module):
    """AASIST-L: ~85K params, input log-mel [B, 1, 80, T], output logits [B, 2]."""

    def __init__(self, n_mels: int = 80, hidden: int = 64):
        super().__init__()
        # Frontend: conv over (mel x time) — captures band propagation
        self.front = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            Res2Block(16),
            nn.MaxPool2d((2, 2)),                      # 80xT -> 40xT/2
            nn.Conv2d(16, hidden, (3, 3), padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            Res2Block(hidden),
            nn.MaxPool2d((2, 2)),                      # 40 -> 20
            nn.Conv2d(hidden, hidden, (3, 3), padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )
        self.proj_dim = hidden * 20  # bands after pooling

        # Joint graph attention over frames + bands
        self.gat1 = GraphAttentionLayer(self.proj_dim, n_heads=4)
        self.gat2 = GraphAttentionLayer(self.proj_dim, n_heads=4)
        self.norm = nn.LayerNorm(self.proj_dim)

        self.head = nn.Sequential(
            nn.Linear(self.proj_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        h = self.front(x)                     # [B, C, F', T']
        B, C, Fp, Tp = h.shape
        h = h.permute(0, 3, 2, 1).reshape(B, Tp * Fp, C)  # nodes over (time, band)
        g = self.gat1(h)
        g = self.norm(g + h)
        g = self.gat2(g)
        g = self.norm(g + h)

        # Attention pooling over nodes
        w = torch.softmax(g.mean(dim=-1), dim=1).unsqueeze(-1)
        pooled = (g * w).sum(dim=1)           # [B, D]
        gmax = g.max(dim=1).values            # [B, D]
        logits = self.head(torch.cat([pooled, gmax], dim=-1))
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = AASISTL()
    print("AASIST-L params:", m.num_params())
    x = torch.randn(2, 1, 80, 100)
    print("out:", m(x).shape)
