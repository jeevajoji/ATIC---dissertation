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


def entropy_buffer_snapshot(model):
    suffixes = (
        "_quantized_cdf",
        "_offset",
        "_cdf_length",
        "scale_table",
        "scale_bound",
    )
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith("entropy.")
        and name.endswith(suffixes)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    model = ATICModel(small_config(), H=args.height, W=args.width)
    model.load_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    entropy_buffers_before_decode = entropy_buffer_snapshot(model)
    with torch.no_grad():
        decoded = model.decompress(
            args.bitstream,
            return_info=True,
            return_diagnostics=True,
        )
    torch.save(
        {
            "x_hat": decoded["x_hat"].cpu(),
            "entropy_diagnostics": {
                name: tensor.cpu()
                for name, tensor in decoded["entropy_diagnostics"].items()
            },
            "entropy_buffers_before_decode": entropy_buffers_before_decode,
            "entropy_buffers_after_decode": entropy_buffer_snapshot(model),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
