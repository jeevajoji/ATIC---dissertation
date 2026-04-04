"""
reconstructor.py  —  ATIC Block 7: Overlapping Patch Reconstructor
====================================================================
The exact inverse of the tokenizer. Converts the decoded spatial
token grid back into a full-resolution pixel image.

Because patches overlap (stride < patch_size), multiple decoded patches
cover the same output pixel. We average those contributions using fold:

    output_pixel = sum(decoded_patch_values) / count(overlapping_patches)

This averaging prevents visible seam artefacts at patch boundaries.

Input shape:  (B, embed_dim, H_t, W_t)  — spatial token grid from decoder
Output shape: (B, out_channels, H, W)   — reconstructed frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlappingPatchReconstructor(nn.Module):
    """
    Converts a spatial token grid back to a full-resolution image.

    Args:
        embed_dim    : token channel dimension — must match encoder (128)
        out_channels : output image channels (3 for RGB, 1 for luma)
        patch_size   : must match tokenizer (16)
        stride       : must match tokenizer (8)
        padding      : must match tokenizer (4) — cropped at the end
        output_H     : expected output height (1080)
        output_W     : expected output width  (1920)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        out_channels: int = 3,
        patch_size: int = 16,
        stride: int = 8,
        padding: int = 4,
        output_H: int = 1080,
        output_W: int = 1920,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.output_H = output_H
        self.output_W = output_W

        self.norm = nn.LayerNorm(embed_dim)

        # Projects each token back to a flat patch vector.
        # Inverse of the Conv2d in the tokenizer.
        self.projection = nn.Linear(
            embed_dim,
            patch_size * patch_size * out_channels,
            bias=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, embed_dim, H_t, W_t)

        Returns:
            image: (B, out_channels, output_H, output_W)
        """
        B, C, H_t, W_t = tokens.shape

        # 1. LayerNorm (channel-last)
        tokens = tokens.permute(0, 2, 3, 1)       # (B, H_t, W_t, embed_dim)
        tokens = self.norm(tokens)

        # 2. Project each token → flat patch vector
        patches = self.projection(tokens)
        # Shape: (B, H_t, W_t, patch_size² × out_channels)

        # 3. Reshape for fold: needs (B, C*kH*kW, L) where L = H_t * W_t
        patches = patches.permute(0, 3, 1, 2)
        patches = patches.contiguous().view(
            B,
            self.out_channels * self.patch_size * self.patch_size,
            H_t * W_t,
        )

        # 4. Compute padded output size
        H_padded = self.output_H + 2 * self.padding
        W_padded = self.output_W + 2 * self.padding

        fold_params = dict(
            output_size=(H_padded, W_padded),
            kernel_size=(self.patch_size, self.patch_size),
            dilation=(1, 1),
            padding=(0, 0),
            stride=(self.stride, self.stride),
        )

        # 5. Fold: accumulate all patch contributions into image space
        folded = F.fold(patches, **fold_params)
        # Shape: (B, out_channels, H_padded, W_padded)

        # 6. Count how many patches overlap each pixel
        #    Fold a ones-tensor with the same params → per-pixel overlap count
        count_map = F.fold(torch.ones_like(patches), **fold_params)
        count_map = count_map.clamp(min=1.0)   # guard against division by zero

        # 7. Average overlapping contributions → no seam artefacts
        averaged = folded / count_map

        # 8. Remove the reflect-padding added by the tokenizer
        p = self.padding
        output = averaged[:, :, p : p + self.output_H, p : p + self.output_W]

        return output


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from tokenizer import OverlappingPatchTokenizer

    H, W = 1080, 1920
    tokenizer = OverlappingPatchTokenizer(
        in_channels=3, embed_dim=128, patch_size=16, stride=8, padding=4
    )
    reconstructor = OverlappingPatchReconstructor(
        embed_dim=128, out_channels=3, patch_size=16, stride=8, padding=4,
        output_H=H, output_W=W,
    )

    dummy_frame = torch.randn(1, 3, H, W)
    tokens = tokenizer(dummy_frame)
    decoded_tokens = torch.randn_like(tokens)
    reconstructed = reconstructor(decoded_tokens)

    print(f"Input:         {list(dummy_frame.shape)}")
    print(f"After tokenizer: {list(tokens.shape)}")
    print(f"Reconstructed: {list(reconstructed.shape)}")
    print(f"Shape match:   {'PASS' if reconstructed.shape == dummy_frame.shape else 'FAIL'}")
    print(f"Parameters:    {sum(p.numel() for p in reconstructor.parameters()):,}")