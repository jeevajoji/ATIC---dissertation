"""Subprocess helper used to prove source-image-free ATIC decoding."""

import argparse

import torch

from atic.config import ArchitectureConfig
from atic.model import ATICModel


def small_config():
    return ArchitectureConfig(
        patch_size=8,
        token_dim=8,
        use_overlapping_patches=True,
        swin_stages=4,
        depths=[2, 2, 2, 2],
        num_heads_enc=[1, 2, 4, 8],
        window_size=2,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    model = ATICModel(small_config(), H=args.height, W=args.width)
    model.load_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    reconstructed = model.decompress(args.bitstream)
    torch.save(reconstructed, args.output)


if __name__ == "__main__":
    main()
