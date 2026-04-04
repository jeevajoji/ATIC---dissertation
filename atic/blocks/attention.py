"""
attention.py — ATIC: Spatial Attention Gate (SAG) + CBAM Channel Attention
============================================================================
Two modules injected after each Swin stage in the encoder and decoder.

SAG (Spatial Attention Gate):
    Learns which spatial regions are visually complex.
    Output: H×W map of scalar weights in [0,1].
    HoneyBee wings → ~0.9  (complex, preserve)
    Smooth sky     → ~0.1  (simple, can coarsen)
    This map is ALSO passed downstream to the adaptive quantizer (Block 3).

CBAM Channel Attention:
    Learns which feature channels are most informative.
    Squeezes spatial dims → shared MLP → per-channel weights in [0,1].
    Amplifies discriminative channels, suppresses irrelevant ones.

Both are gating mechanisms: they re-weight features rather than transform
them, which keeps gradients stable and the modules lightweight.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SpatialAttentionGate(nn.Module):
    """
    Produces a spatial importance map in [0,1] for each pixel location.

    Architecture:
        1×1 conv  →  reduce channels (C → C//8)   cheap channel compression
        7×7 conv  →  large receptive field         captures spatial context
        sigmoid   →  squash to [0,1]

    Returns both the gated feature map AND the raw attention map, because
    the attention map is reused by the adaptive quantizer (Block 3).

    Args:
        in_channels : number of input feature channels
        reduction   : channel compression ratio (default 8)
    """

    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()

        mid = max(in_channels // reduction, 8)  # at least 8 channels

        # 1×1 conv: cheap channel compression before the expensive 7×7
        self.compress = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.bn1      = nn.BatchNorm2d(mid)

        # 7×7 conv: padding=3 preserves spatial size
        self.attend   = nn.Conv2d(mid, 1, kernel_size=7, padding=3, bias=False)
        self.bn2      = nn.BatchNorm2d(1)

        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            gated : (B, C, H, W)  — input scaled by spatial attention
            attn  : (B, 1, H, W)  — spatial importance map in [0,1]
                    passed to the adaptive quantizer
        """
        attn = self.relu(self.bn1(self.compress(x)))   # (B, C//8, H, W)
        attn = self.sigmoid(self.bn2(self.attend(attn)))  # (B, 1, H, W)
        return x * attn, attn


class ChannelAttentionCBAM(nn.Module):
    """
    CBAM channel attention module.

    Squeezes spatial dims using average-pooling AND max-pooling (two
    complementary statistics), passes each through a shared MLP, sums,
    and applies sigmoid → per-channel weights in [0,1].

    Avg pooling captures smooth context; max pooling captures sharp
    salient features. Using both is the key CBAM contribution.

    Args:
        in_channels : number of input feature channels
        reduction   : MLP bottleneck ratio (default 16)
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        bottleneck = max(in_channels // reduction, 8)

        # Shared MLP applied to both avg-pool and max-pool features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W)  — channels re-weighted by learned importance
        """
        B, C, H, W = x.shape

        avg_feat = x.mean(dim=[2, 3])       # (B, C) — average over H, W
        max_feat = x.amax(dim=[2, 3])       # (B, C) — max over H, W

        avg_out  = self.mlp(avg_feat)       # (B, C)
        max_out  = self.mlp(max_feat)       # (B, C)

        scale = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        return x * scale


class AttentionStack(nn.Module):
    """
    The 3-layer attention stack injected after each Swin stage.

    Order: Swin window attention (done inside SwinStage) → SAG → CBAM
    This module covers SAG + CBAM only.

    Ablation flags allow toggling each component independently:
        use_sag=False, use_cbam=False  → A1/A2 (no attention stack)
        use_sag=True,  use_cbam=False  → A3
        use_sag=True,  use_cbam=True   → A4 / full ATIC

    Args:
        in_channels : channel dimension at this stage
        use_sag     : enable Spatial Attention Gate
        use_cbam    : enable CBAM channel attention
    """

    def __init__(
        self,
        in_channels: int,
        use_sag: bool  = True,
        use_cbam: bool = True,
    ):
        super().__init__()

        self.use_sag  = use_sag
        self.use_cbam = use_cbam
        self.sag      = SpatialAttentionGate(in_channels)  if use_sag  else None
        self.cbam     = ChannelAttentionCBAM(in_channels)  if use_cbam else None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            x    : (B, C, H, W)  — attended feature map
            attn : (B, 1, H, W) or None  — spatial attention map for quantizer
        """
        attn = None

        if self.use_sag and self.sag is not None:
            x, attn = self.sag(x)

        if self.use_cbam and self.cbam is not None:
            x = self.cbam(x)

        return x, attn