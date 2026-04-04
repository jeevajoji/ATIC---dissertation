"""
encoder.py — ATIC Block 2: Swin Transformer Encoder
=====================================================
4-stage hierarchical encoder. Each stage:
    1. SwinStage      — shifted window self-attention (via timm)
    2. AttentionStack — SAG + CBAM injection
    3. PatchMerging2D — halves spatial resolution, doubles channels

Spatial dimensions for 1920×1080 input (tokenizer: patch=16, stride=8, pad=4):
    Token grid in  :  135 × 240,  128ch
    After stage 1  :   67 × 120,  256ch
    After stage 2  :   33 ×  60,  512ch
    After stage 3  :   16 ×  30, 1024ch
    Stage 4 output :   16 ×  30, 1024ch  (no spatial downsampling)

Latent shape: (B, 1024, 16, 30)

The spatial attention map from Stage 4 (most semantically meaningful)
is returned and passed to the adaptive quantizer (Block 3).

Ablation flags:
    use_sag, use_cbam — toggle attention modules for stages A3 and A4
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

try:
    from .attention import AttentionStack
except ImportError:
    from attention import AttentionStack


# ---------------------------------------------------------------------------
# Swin Stage wrapper
# Wraps timm's BasicLayer and handles format conversion:
#   our (B, C, H, W)  ↔  timm expects (B, H*W, C)
# ---------------------------------------------------------------------------

class SwinStage(nn.Module):
    """
    One Swin Transformer stage: N blocks of W-MSA + SW-MSA alternating.
    No downsampling — handled separately by PatchMerging2D.

    Args:
        dim              : channel dimension at this stage
        input_resolution : (H, W) of the token grid entering this stage
        depth            : number of Swin blocks (must be even)
        num_heads        : number of self-attention heads
        window_size      : local window size (default 7, standard Swin)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 5,
    ):
        super().__init__()

        from timm.models.swin_transformer import BasicLayer

        assert depth % 2 == 0, f"depth must be even (W-MSA + SW-MSA pairs), got {depth}"

        self.H, self.W = input_resolution

        self.layer = BasicLayer(
            dim=dim,
            out_dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            downsample=None,    # downsampling done separately
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:    x: (B, C, H, W)
        Returns:    (B, C, H, W)  — same spatial size, same channels
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, L, C)
        x = self.layer(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)    # (B, C, H, W)
        return x


# ---------------------------------------------------------------------------
# Patch Merging — written independently of timm for format compatibility
# ---------------------------------------------------------------------------

class PatchMerging2D(nn.Module):
    """
    Swin-style downsampling: (H, W) → (H//2, W//2), channels: C → 2C.

    Concatenates 4 spatially-adjacent tokens along the channel axis,
    then applies LayerNorm + linear reduction: 4C → 2C.

    Args:
        dim : input channel dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:    x: (B, C, H, W)
        Returns:    (B, 2C, H//2, W//2)
        """
        x0 = x[:, :, 0::2, 0::2]   # top-left
        x1 = x[:, :, 1::2, 0::2]   # bottom-left
        x2 = x[:, :, 0::2, 1::2]   # top-right
        x3 = x[:, :, 1::2, 1::2]   # bottom-right

        x = torch.cat([x0, x1, x2, x3], dim=1)   # (B, 4C, H//2, W//2)

        B, C4, H2, W2 = x.shape
        x = x.permute(0, 2, 3, 1)    # (B, H//2, W//2, 4C)
        x = self.norm(x)
        x = self.reduction(x)         # (B, H//2, W//2, 2C)
        x = x.permute(0, 3, 1, 2)    # (B, 2C, H//2, W//2)
        return x


# ---------------------------------------------------------------------------
# One full encoder stage: Swin + AttentionStack + optional PatchMerging
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    One full encoder stage with dynamic padding for window size and merge constraints.
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
        downsample: bool = True,
    ):
        super().__init__()
        
        self.H, self.W = input_resolution
        
        # Calculate how much padding is needed to be cleanly divisible by window_size
        pad_H = (window_size - self.H % window_size) % window_size
        pad_W = (window_size - self.W % window_size) % window_size
        self.padded_resolution = (self.H + pad_H, self.W + pad_W)

        self.swin    = SwinStage(dim, self.padded_resolution, depth, num_heads, window_size)
        self.attn    = AttentionStack(dim, use_sag=use_sag, use_cbam=use_cbam)  
        self.merge   = PatchMerging2D(dim) if downsample else None
        self.out_dim = dim * 2 if downsample else dim

    def forward(
        self, x
    ):
        import torch.nn.functional as F
        B, C, H, W = x.shape
        
        # 1. Swin padding
        pad_h = self.padded_resolution[0] - H
        pad_w = self.padded_resolution[1] - W
        
        x_pad = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else x

        # 2. Swin stage
        x_swin = self.swin(x_pad)

        # 3. Crop back
        x = x_swin[:, :, :H, :W] if (pad_h > 0 or pad_w > 0) else x_swin

        # 4. Attention
        x, attn = self.attn(x)

        # 5. Patch Merging (requires even padding)
        if self.merge is not None:
            pad_m_h = (2 - H % 2) % 2
            pad_m_w = (2 - W % 2) % 2
            x = F.pad(x, (0, pad_m_w, 0, pad_m_h)) if (pad_m_h > 0 or pad_m_w > 0) else x
            x = self.merge(x)
            
        return x, attn


# ---------------------------------------------------------------------------
# Full 4-stage Swin Encoder
# ---------------------------------------------------------------------------

class SwinEncoder(nn.Module):
    """
    ATIC Block 2: 4-stage Swin Transformer Encoder.

    Stage shapes (1920×1080 input):
        Stage 1: 135×240, 128ch  →  67×120, 256ch
        Stage 2:  67×120, 256ch  →  33×60,  512ch
        Stage 3:  33×60,  512ch  →  16×30, 1024ch
        Stage 4:  16×30, 1024ch  →  16×30, 1024ch  (no downsampling)

    Latent output: (B, 1024, 16, 30)

    The spatial attention map from Stage 4 is returned to the quantizer.

    Args:
        embed_dim   : channel dim from tokenizer output (128)
        token_H     : token grid height  (135 for 1080p with our tokenizer)
        token_W     : token grid width   (240 for 1080p)
        depths      : Swin blocks per stage
        num_heads   : attention heads per stage
        window_size : Swin local window size
        use_sag     : enable SAG  (ablation toggle)
        use_cbam    : enable CBAM (ablation toggle)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        token_H: int = 136,
        token_W: int = 240,
        depths: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [4, 8, 16, 32],
        window_size: int = 8,
        use_sag: bool = True,
        use_cbam: bool = True,
    ):
        super().__init__()

        # Channel dims double every stage: [128, 256, 512, 1024]
        dims = [embed_dim * (2 ** i) for i in range(4)]

        # Spatial resolutions halve every stage
        resolutions = []
        H, W = token_H, token_W
        for _ in range(4):
            resolutions.append((H, W))
            # PatchMerging dynamically pads to an even number before dividing by 2
            H_even, W_even = H + (H % 2), W + (W % 2)
            H, W = H_even // 2, W_even // 2

        self.stages = nn.ModuleList([
            EncoderBlock(
                dim=dims[i],
                input_resolution=resolutions[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                use_sag=use_sag,
                use_cbam=use_cbam,
                downsample=(i < 3),   # no downsampling after stage 4
            )
            for i in range(4)
        ])

        self.latent_dim = dims[3]   # 1024
        self.norm = nn.LayerNorm(self.latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, embed_dim, H_t, W_t)  — token grid from tokenizer

        Returns:
            latent   : (B, 1024, 16, 30)  — encoder output
            attn_map : (B, 1, H4, W4) or None  — deepest spatial attn map
                       passed to the adaptive quantizer
        """
        attn_map = None

        for stage in self.stages:
            x, attn = stage(x)
            if attn is not None:
                attn_map = attn   # keep the deepest (most semantic) map

        # Final LayerNorm (channel-last)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        return x, attn_map


# ---------------------------------------------------------------------------
# Smoke test — python encoder.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    encoder = SwinEncoder(
        embed_dim=128,
        token_H=135,
        token_W=240,
        use_sag=True,
        use_cbam=True,
    )

    tokens = torch.randn(1, 128, 135, 240)
    print(f"Input:       {list(tokens.shape)}")

    latent, attn_map = encoder(tokens)
    print(f"Latent:      {list(latent.shape)}")
    print(f"Attn map:    {list(attn_map.shape) if attn_map is not None else 'None'}")
    print(f"Parameters:  {sum(p.numel() for p in encoder.parameters()):,}")