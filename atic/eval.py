"""Evaluation helpers for estimated and real ATIC bitrates."""

from __future__ import annotations

import os
from typing import Optional

import torch

from atic.metrics import ATICMetrics


def _extract_images(batch):
    """Support datasets returning either images or ``(images, paths)``."""

    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _bpp_from_likelihoods(likelihoods, batch, eps: float = 1e-9) -> float:
    """Calculate differentiable-model BPP from entropy likelihoods."""

    batch_size, _, height, width = batch.shape
    num_pixels = batch_size * height * width
    return float(_bits_from_likelihoods(likelihoods, eps=eps) / num_pixels)


def _bits_from_likelihoods(likelihoods, eps: float = 1e-9) -> float:
    """Return the total estimated coding bits represented by likelihoods."""

    if likelihoods is None:
        return 0.0

    if isinstance(likelihoods, dict):
        bits = 0.0
        for likelihood in likelihoods.values():
            if likelihood is not None:
                bits += -torch.log2(likelihood.clamp(min=eps)).sum().item()
        return float(bits)
    elif isinstance(likelihoods, torch.Tensor):
        return float(
            -torch.log2(likelihoods.clamp(min=eps)).sum().item()
        )
    raise TypeError(
        f"Unsupported likelihoods type: {type(likelihoods)}. "
        "Expected dict, Tensor, or None."
    )


def _image_bpp_from_likelihoods(
    likelihoods,
    image_index: int,
    height: int,
    width: int,
    eps: float = 1e-9,
) -> float:
    """Calculate one image's likelihood BPP from a batched model output."""

    if likelihoods is None:
        return 0.0
    if isinstance(likelihoods, dict):
        selected = {
            key: (
                value[image_index : image_index + 1]
                if value is not None
                else None
            )
            for key, value in likelihoods.items()
        }
    elif isinstance(likelihoods, torch.Tensor):
        selected = likelihoods[image_index : image_index + 1]
    else:
        raise TypeError(
            f"Unsupported likelihoods type: {type(likelihoods)}. "
            "Expected dict, Tensor, or None."
        )
    return float(_bits_from_likelihoods(selected, eps=eps) / (height * width))


def eval_single(
    model,
    dataloader,
    device="cuda",
    bitstream_dir=None,
    use_actual_bitstream: Optional[bool] = None,
):
    """Evaluate one model on a validation or test loader.

    When ``bitstream_dir`` is supplied (as it is by ``ablation.py``), the
    headline ``BPP`` is measured from complete ``.atic`` files and quality
    metrics use images returned by ``decompress()``.  Model-likelihood rate is
    retained as ``BPP_estimated``.

    With no bitstream directory, callers can still request real coding through
    ``use_actual_bitstream=True``; streams are then held only in memory.
    """

    if dataloader is None:
        return {}

    model.to(device)
    model.eval()
    if use_actual_bitstream is None:
        use_actual_bitstream = bitstream_dir is not None
    if use_actual_bitstream and not (
        hasattr(model, "compress") and hasattr(model, "decompress")
    ):
        raise TypeError("Actual-bitstream evaluation requires codec methods")
    if bitstream_dir is not None:
        os.makedirs(bitstream_dir, exist_ok=True)

    if hasattr(model, "update"):
        model.update(force=True)

    metric_calculator = ATICMetrics(device=device)
    totals = {
        "PSNR": 0.0,
        "SSIM": 0.0,
        "MS-SSIM": 0.0,
        "LPIPS": 0.0,
        "DISTS": 0.0,
        "MSE": 0.0,
    }
    counts = {key: 0 for key in totals}

    total_pixels = 0
    total_estimated_bits = 0.0
    total_actual_bytes = 0
    total_payload_bytes = 0
    total_images = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _extract_images(batch).to(device, non_blocking=True)
            outputs = model(batch)
            likelihoods = outputs.get("likelihoods")
            batch_pixels = int(batch.size(0) * batch.size(2) * batch.size(3))
            total_pixels += batch_pixels
            total_estimated_bits += _bits_from_likelihoods(likelihoods)
            image_height = int(batch.size(2))
            image_width = int(batch.size(3))

            if use_actual_bitstream:
                reconstructions = []
                image_metric_bpps = []
                for image_index in range(batch.size(0)):
                    global_index = total_images + image_index
                    output_path = None
                    if bitstream_dir is not None:
                        output_path = os.path.join(
                            bitstream_dir,
                            f"sample_{global_index:06d}.atic",
                        )

                    encoded = model.compress(
                        batch[image_index : image_index + 1],
                        output_path=output_path,
                    )
                    reconstructions.append(
                        model.decompress(encoded["bitstream"])
                    )
                    encoded_bytes = int(encoded["num_bytes"])
                    total_actual_bytes += encoded_bytes
                    total_payload_bytes += int(encoded["payload_bytes"])
                    image_metric_bpps.append(
                        float(
                            encoded_bytes
                            * 8.0
                            / (image_height * image_width)
                        )
                    )

                x_hat = torch.cat(reconstructions, dim=0)
            else:
                x_hat = outputs["x_hat"]
                image_metric_bpps = [
                    _image_bpp_from_likelihoods(
                        likelihoods,
                        image_index,
                        image_height,
                        image_width,
                    )
                    for image_index in range(batch.size(0))
                ]
            total_images += int(batch.size(0))

            for image_index, metric_bpp in enumerate(image_metric_bpps):
                image_metrics = metric_calculator.compute_all(
                    x_hat=x_hat[image_index : image_index + 1],
                    x=batch[image_index : image_index + 1],
                    bpp=metric_bpp,
                )
                for key, value in image_metrics.items():
                    if key in totals and value is not None:
                        totals[key] += float(value)
                        counts[key] += 1

    averaged = {
        key: totals[key] / counts[key]
        for key in totals
        if counts[key] > 0
    }
    if total_pixels == 0:
        return averaged

    estimated_bpp = float(total_estimated_bits / total_pixels)
    averaged["BPP_estimated"] = estimated_bpp
    averaged["num_images"] = total_images
    averaged["num_pixels"] = total_pixels

    if use_actual_bitstream:
        actual_bpp = float((total_actual_bytes * 8.0) / total_pixels)
        payload_bpp = float((total_payload_bytes * 8.0) / total_pixels)
        averaged["BPP"] = actual_bpp
        averaged["BPP_actual"] = actual_bpp
        averaged["BPP_payload"] = payload_bpp
        averaged["bitstream_bytes"] = total_actual_bytes
        averaged["payload_bytes"] = total_payload_bytes
    else:
        averaged["BPP"] = estimated_bpp

    return averaged


def eval_loop(
    model,
    variant_name,
    dataloader,
    device="cuda",
    bitstream_dir=None,
    use_actual_bitstream: Optional[bool] = None,
):
    """Backward-compatible reporting wrapper around :func:`eval_single`."""

    if dataloader is None:
        print(f"[{variant_name}] Dataloader not found. Skipping eval.")
        return None

    print(f"\n{'=' * 45}")
    print(f"[{variant_name}] Evaluating...")
    print(f"{'=' * 45}")

    point = eval_single(
        model=model,
        dataloader=dataloader,
        device=device,
        bitstream_dir=bitstream_dir,
        use_actual_bitstream=use_actual_bitstream,
    )

    print(f"  BPP:     {point.get('BPP', 0):.4f}")
    if "BPP_estimated" in point:
        print(f"  BPP est: {point['BPP_estimated']:.4f}")
    print(f"  PSNR:    {point.get('PSNR', 0):.4f}")
    print(f"  SSIM:    {point.get('SSIM', 0):.4f}")
    print(f"  MS-SSIM: {point.get('MS-SSIM', 0):.4f}")
    print(f"  LPIPS:   {point.get('LPIPS', 0):.4f}")
    print(f"  DISTS:   {point.get('DISTS', 0):.4f}")
    print(f"  MSE:     {point.get('MSE', 0):.6f}")

    bpp_key = round(point.get("BPP", 0.0), 4)
    return {
        "variant": variant_name,
        "bpp_levels": {
            bpp_key: point,
        },
    }
