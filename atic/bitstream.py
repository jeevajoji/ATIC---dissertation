"""
bitstream.py - Utilities for real byte-level bitrate accounting.

This module serializes quantized latents into a compressed payload and
measures true bytes-per-pixel from the resulting bitstream size.
"""
import io
import os
from typing import Dict, Optional

import numpy as np
import torch


def _quantized_int_array(tensor: torch.Tensor) -> np.ndarray:
    """Convert quantized tensor values to integer symbols for coding."""
    arr = torch.round(tensor.detach().cpu()).to(torch.int32).numpy()
    return arr


def encode_latent_payload(
    z_hat: torch.Tensor,
    h_hat: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Encode quantized latents into a compressed npz payload and return byte stats.

    If save_path is provided, writes the bitstream to disk.
    """
    payload = {"z_hat": _quantized_int_array(z_hat)}
    if h_hat is not None:
        payload["h_hat"] = _quantized_int_array(h_hat)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    bitstream = buffer.getvalue()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(bitstream)

    return {
        "num_bytes": float(len(bitstream)),
        "num_bits": float(len(bitstream) * 8),
    }


def bpp_from_num_bytes(num_bytes: float, batch: torch.Tensor) -> float:
    """Compute bits-per-pixel from actual payload bytes."""
    n, _, h, w = batch.shape
    num_pixels = n * h * w
    if num_pixels == 0:
        return 0.0
    return float((num_bytes * 8.0) / num_pixels)
