"""
tokenizer.py  —  ATIC Block 1: Overlapping Patch Tokenizer
============================================================
Converts a full-resolution input frame into a sequence of
overlapping patch tokens ready for the Swin Transformer encoder.

Design choices:
    - kernel_size=16, stride=8  →  50% overlap between adjacent patches
    - Overlap catches features that straddle patch boundaries
        (e.g. bee wing edges, water ripple peaks)
    - Linear projection (via Conv2d) maps each 16x16xC patch → 128-dim token
    - Output is a 2D spatial token grid, NOT a flattened sequence,
    because Swin Transformer operates on spatial grids

Input shape:  (B, 3, H, W)   — batch of RGB frames
Output shape: (B, embed_dim, H_t, W_t)
    where H_t = floor((H + 2*pad - kernel) / stride) + 1
        W_t = floor((W + 2*pad - kernel) / stride) + 1

For H=1080, W=1920, kernel=16, stride=8, pad=4:
    H_t = floor((1080 + 8 - 16) / 8) + 1 = 135
    W_t = floor((1920 + 8 - 16) / 8) + 1 = 240
    Total tokens: 135 × 240 = 32,400
"""

import torch
import torch.nn as nn


class OverlappingPatchTokenizer(nn.Module):
    """
    Embeds an image into a grid of overlapping patch tokens.

    Args:
        in_channels  : number of input image channels (3 for RGB, 1 for Y-luma)
        embed_dim    : output token dimension (128 in ATIC)
        patch_size   : spatial size of each patch in pixels (16 in ATIC)
        stride       : step between consecutive patches (8 = 50% overlap)
        padding      : reflect-pad the image before patching so border
                        pixels are covered by at least one patch
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_size: int = 16,
        stride: int = 8,
        padding: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.embed_dim = embed_dim

        # A single Conv2d does both patching and projection in one step:
        #   kernel_size=patch_size  →  each receptive field is one patch
        #   stride=stride           →  controls overlap
        #   out_channels=embed_dim  →  the linear projection
        # Equivalent to: extract patch → flatten → linear(patch_size²×C, embed_dim)
        # but Conv2d is faster and keeps gradients clean.
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=0,          # padding handled separately (reflect mode)
            bias=True,
        )

        # LayerNorm over the channel (embed) dimension.
        # Applied after projection to stabilise training.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  — a batch of images

        Returns:
            tokens: (B, embed_dim, H_t, W_t)  — spatial token grid
        """
        # 1. Reflect-pad so border regions get full patch coverage.
        #    'reflect' continues image content rather than padding with zeros,
        #    avoiding artificial edges that confuse attention at boundaries.
        if self.padding > 0:
            x = nn.functional.pad(x, (self.padding,) * 4, mode="reflect")

        # 2. Patch + project in one Conv2d pass
        tokens = self.projection(x)
        # Shape: (B, embed_dim, H_t, W_t)

        # 3. LayerNorm — expects normalised dim to be last, so permute → norm → permute back
        B, C, H_t, W_t = tokens.shape
        tokens = tokens.permute(0, 2, 3, 1)   # (B, H_t, W_t, embed_dim)
        tokens = self.norm(tokens)
        tokens = tokens.permute(0, 3, 1, 2)   # (B, embed_dim, H_t, W_t)

        return tokens

    def output_spatial_size(self, H: int, W: int) -> tuple[int, int]:
        """Returns token grid size (H_t, W_t) for a given input (H, W)."""
        H_padded = H + 2 * self.padding
        W_padded = W + 2 * self.padding
        H_t = (H_padded - self.patch_size) // self.stride + 1
        W_t = (W_padded - self.patch_size) // self.stride + 1
        return H_t, W_t


if __name__ == "__main__":
    tokenizer = OverlappingPatchTokenizer(
        in_channels=3, embed_dim=128, patch_size=16, stride=8, padding=4
    )
    dummy = torch.randn(1, 3, 1080, 1920)
    print(f"Input shape:  {list(dummy.shape)}")
    tokens = tokenizer(dummy)
    print(f"Token shape:  {list(tokens.shape)}")
    H_t, W_t = tokenizer.output_spatial_size(1080, 1920)
    print(f"Token grid:   {H_t} x {W_t} = {H_t * W_t} tokens")
    print(f"Parameters:   {sum(p.numel() for p in tokenizer.parameters()):,}")