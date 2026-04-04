"""
decoder.py — ATIC Block 6: Mirror Swin Transformer Decoder
===========================================================
Exact mirror of the encoder. Progressively upsamples the latent tensor
back to the full token grid resolution for the patch reconstructor.

Stage shapes (mirror of encoder):
    Stage 1:  16×30,  1024ch  →  33×60,   512ch
    Stage 2:  33×60,   512ch  →  67×120,  256ch
    Stage 3:  67×120,  256ch  → 135×240,  128ch
    Stage 4: 135×240,  128ch  → 135×240,  128ch  (no upsampling)

PatchExpanding2D is the inverse of PatchMerging2D:
    Linear expand: 2C → 4C, then reshape to (B, C, 2H, 2W)
    This is the standard Swin-Unet upsampling strategy.

Skip connections are supported optionally.
In ablation stages A1–A5 set use_skip=False.
In the full model you can pass encoder skip features for better fidelity.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

try:
    from .attention import AttentionStack
    from .encoder   import SwinStage
except ImportError:
    from attention import AttentionStack
    from encoder   import SwinStage


# ---------------------------------------------------------------------------
# Patch Expanding — inverse of PatchMerging2D
# ---------------------------------------------------------------------------

class PatchExpanding2D(nn.Module):
    """
    Swin-Unet style upsampling: (H, W, 2C) → (2H, 2W, C).

    Steps:
        1. Linear: 2C → 4C          expand channel budget
        2. LayerNorm over 4C
        3. Reshape: pixel-shuffle   fold extra channels into spatial dims
           (B, H, W, 4C) → (B, 2H, 2W, C)

    Args:
        dim : input channel dimension (the 2C side)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)   # applied after spatial expand → C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:    x: (B, C_in, H, W)   where C_in = 2C
        Returns:    (B, C_in//2, 2H, 2W)
        """
        B, C, H, W = x.shape

        # channel-last for linear layers
        x = x.permute(0, 2, 3, 1)              # (B, H, W, C)
        x = self.expand(x)                      # (B, H, W, 2C)

        # pixel-shuffle: fold channel pairs into spatial positions
        # (B, H, W, 2C) → (B, 2H, 2W, C//2)
        x = x.view(B, H, W, 2, 2, C // 2)      # split 2C → 2×2 × C//2
        x = x.permute(0, 1, 3, 2, 4, 5)        # (B, H, 2, W, 2, C//2)
        x = x.contiguous().view(B, 2*H, 2*W, C // 2)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)              # (B, C//2, 2H, 2W)
        return x


# ---------------------------------------------------------------------------
# One full decoder stage: optional skip → PatchExpanding → Swin → AttentionStack
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    One full decoder stage with dynamic padding matching exact Swin window size constraints
    and handling feature expansions strictly aligned with encoder skips.
    """
    def __init__(
        self,
        dim: int,
        input_resolution: tuple, 
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        use_sag: bool = True,
        use_cbam: bool = True,
        upsample: bool = True,
        use_skip: bool = False,
    ):
        super().__init__()

        self.upsample  = upsample
        self.use_skip  = use_skip
        
        # If upsampling: input is 2*dim, expand outputs dim
        # If not upsampling (last stage): input is dim, output is dim
        in_dim         = dim * 2 if upsample else dim
        self.target_resolution = input_resolution

        if use_skip:
            self.skip_fuse = __import__('torch').nn.Sequential(
                __import__('torch').nn.Conv2d(in_dim + dim, in_dim, kernel_size=1, bias=False),     
                __import__('torch').nn.LayerNorm([in_dim, 1, 1]),   
            )
        else:
            self.skip_fuse = None

        self.expand = PatchExpanding2D(in_dim) if upsample else None
        out_dim     = dim

        # Calculate padded resolution required for Swin
        pad_H = (window_size - self.target_resolution[0] % window_size) % window_size
        pad_W = (window_size - self.target_resolution[1] % window_size) % window_size
        self.padded_resolution = (self.target_resolution[0] + pad_H, self.target_resolution[1] + pad_W)

        self.swin    = SwinStage(out_dim, self.padded_resolution, depth, num_heads, window_size)
        self.attn    = AttentionStack(out_dim, use_sag=use_sag, use_cbam=use_cbam)

    def forward(
        self,
        x,
        skip = None,
    ):
        import torch.nn.functional as F
        import torch

        # 1. upsample (expands dims and divides channels)
        if self.expand is not None:
            x = self.expand(x)

        # 2. Crop x to exactly match target_resolution and skip resolution
        H, W = x.shape[2:]
        TH, TW = self.target_resolution
        if H > TH or W > TW:
            x = x[:, :, :TH, :TW]

        # 3. optional skip fusion before swin 
        if self.use_skip and skip is not None and self.skip_fuse is not None:   
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fuse(x)

        # 4. Pad for SwinStage
        pad_h = self.padded_resolution[0] - TH
        pad_w = self.padded_resolution[1] - TW
        
        x_pad = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x

        x_swin = self.swin(x_pad)

        # 5. Crop back to target_resolution
        x = x_swin[:, :, :TH, :TW] if (pad_h > 0 or pad_w > 0) else x_swin

        x, attn = self.attn(x)
        return x, attn


# ---------------------------------------------------------------------------
# Full 4-stage Swin Decoder
# ---------------------------------------------------------------------------

class SwinDecoder(nn.Module):
    """
    ATIC Block 6: 4-stage Mirror Swin Transformer Decoder.

    Reads the quantised latent (B, 1024, 16, 30) and progressively
    upsamples to (B, 128, 135, 240) — matching the tokenizer output shape
    that the patch reconstructor expects.

    Args:
        embed_dim   : base channel dim (must match encoder, 128)
        token_H     : token grid height at full resolution (135)
        token_W     : token grid width  at full resolution (240)
        depths      : Swin blocks per stage
        num_heads   : attention heads per stage
        window_size : Swin local window size
        use_sag     : enable SAG  (ablation toggle)
        use_cbam    : enable CBAM (ablation toggle)
        use_skip    : enable encoder skip connections
    """

    def __init__(
        self,
        embed_dim: int = 128,
        token_H: int = 136,
        token_W: int = 240,
        depths: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [32, 16, 8, 4],   # reversed vs encoder
        window_size: int = 8,
        use_sag: bool = True,
        use_cbam: bool = True,
        use_skip: bool = False,
    ):
        super().__init__()

        # Channel dims: encoder was [128,256,512,1024] — decoder reverses
        # Each DecoderBlock takes 2C in, produces C out
        dims = [embed_dim * (2 ** i) for i in range(4)]
        # dims = [128, 256, 512, 1024]
        # decoder stages process: 1024→512, 512→256, 256→128, 128→128

        # Output resolutions after upsampling at each stage
        resolutions = []
        H, W = token_H, token_W
        spatial = [(H, W)]
        for _ in range(3):
            H_even, W_even = H + (H % 2), W + (W % 2)
            H, W = H_even // 2, W_even // 2
            spatial.append((H, W))
        spatial.reverse()
        # spatial = [(16,30), (33,60), (67,120), (135,240)]

        # upsampled-to resolutions for each decoder stage
        # stage 0: 16×30 → 33×60   (output_res = spatial[1])
        # stage 1: 33×60 → 67×120  (output_res = spatial[2])
        # stage 2: 67×120→135×240  (output_res = spatial[3])
        # stage 3: 135×240→135×240 (no upsample)
        output_resolutions = [spatial[1], spatial[2], spatial[3], spatial[3]]
        out_dims           = [dims[2], dims[1], dims[0], dims[0]]
        # out_dims = [512, 256, 128, 128]

        self.stages = nn.ModuleList([
            DecoderBlock(
                dim=out_dims[i],
                input_resolution=output_resolutions[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                use_sag=use_sag,
                use_cbam=use_cbam,
                upsample=(i < 3),
                use_skip=use_skip,
            )
            for i in range(4)
        ])

        # Final norm before handing off to patch reconstructor
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        latent: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            latent : (B, 1024, 16, 30)   quantised latent from entropy decoder
            skips  : list of 4 encoder feature maps, ordered coarse→fine
                     [stage4_out, stage3_out, stage2_out, stage1_out]
                     Pass None if use_skip=False (default)

        Returns:
            (B, 128, 135, 240)  — token grid for patch reconstructor
        """
        x = latent

        for i, stage in enumerate(self.stages):
            skip = skips[i] if (skips is not None) else None
            x, _ = stage(x, skip=skip)

        # Final LayerNorm (channel-last)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        return x


# ---------------------------------------------------------------------------
# Smoke test — python decoder.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    decoder = SwinDecoder(
        embed_dim=128,
        token_H=135,
        token_W=240,
        use_sag=True,
        use_cbam=True,
        use_skip=False,
    )

    latent = torch.randn(1, 1024, 16, 30)
    print(f"Latent in:   {list(latent.shape)}")

    out = decoder(latent)
    print(f"Decoder out: {list(out.shape)}")
    print(f"Expected:    [1, 128, 135, 240]")
    print(f"Shape match: {'PASS' if list(out.shape) == [1, 128, 135, 240] else 'FAIL'}")
    print(f"Parameters:  {sum(p.numel() for p in decoder.parameters()):,}")