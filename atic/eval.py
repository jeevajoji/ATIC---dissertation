"""Evaluation helpers for estimated and real ATIC bitrates."""

from __future__ import annotations

import math
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

    if not math.isfinite(eps) or not 0.0 < eps <= 1.0:
        raise ValueError("eps must be finite and in the interval (0, 1]")

    if likelihoods is None:
        return 0.0

    def tensor_bits(likelihood) -> float:
        if not isinstance(likelihood, torch.Tensor):
            raise TypeError(
                "Likelihood entries must be tensors or None; "
                f"received {type(likelihood)}"
            )
        if not likelihood.is_floating_point():
            raise TypeError("Likelihood tensors must use a floating-point dtype")
        if not bool(torch.isfinite(likelihood).all()):
            raise ValueError("Likelihood tensor contains NaN or infinity")
        if bool((likelihood < 0).any()):
            raise ValueError("Likelihood tensor contains a negative probability")
        probability = likelihood.clamp(min=eps, max=1.0)
        return float(-torch.log2(probability).sum().item())

    if isinstance(likelihoods, dict):
        bits = 0.0
        for likelihood in likelihoods.values():
            if likelihood is not None:
                bits += tensor_bits(likelihood)
        return float(bits)
    elif isinstance(likelihoods, torch.Tensor):
        return tensor_bits(likelihoods)
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


def _encoded_byte_breakdown(encoded):
    """Validate and return the byte components reported by ``compress()``."""

    num_bytes = int(encoded["num_bytes"])
    payload_bytes = int(encoded["payload_bytes"])
    y_bytes = int(encoded["y_bytes"])
    z_bytes = int(encoded["z_bytes"])
    header_bytes = int(encoded.get("header_bytes", num_bytes - payload_bytes))

    components = {
        "num_bytes": num_bytes,
        "payload_bytes": payload_bytes,
        "y_bytes": y_bytes,
        "z_bytes": z_bytes,
        "header_bytes": header_bytes,
    }
    if any(value < 0 for value in components.values()):
        raise ValueError(f"Codec returned a negative byte count: {components}")
    if y_bytes + z_bytes != payload_bytes:
        raise ValueError(
            "Codec byte accounting is inconsistent: "
            f"y_bytes + z_bytes = {y_bytes + z_bytes}, "
            f"payload_bytes = {payload_bytes}"
        )
    if header_bytes + payload_bytes != num_bytes:
        raise ValueError(
            "Codec byte accounting is inconsistent: "
            f"header_bytes + payload_bytes = {header_bytes + payload_bytes}, "
            f"num_bytes = {num_bytes}"
        )
    return components


def _format_optional_metric(point, name: str, precision: int) -> str:
    """Render unavailable optional metrics explicitly instead of as zero."""

    value = point.get(name)
    return "unavailable" if value is None else f"{float(value):.{precision}f}"


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
    retained as ``BPP_estimated``. Actual mode also separates y, z, and header
    rates; ``BPP_payload_minus_estimated`` is the entropy-coder calibration
    gap, while ``BPP_actual_minus_estimated`` additionally includes the fixed
    container header.

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
        # Training force-refreshes the tables before saving the final
        # checkpoint. Preserve those exact tables here so streams remain
        # decodable by a fresh receiver; empty legacy tables are still built.
        model.update(force=False)

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
    total_y_bytes = 0
    total_z_bytes = 0
    total_header_bytes = 0
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
                    byte_counts = _encoded_byte_breakdown(encoded)
                    encoded_bytes = byte_counts["num_bytes"]
                    total_actual_bytes += encoded_bytes
                    total_payload_bytes += byte_counts["payload_bytes"]
                    total_y_bytes += byte_counts["y_bytes"]
                    total_z_bytes += byte_counts["z_bytes"]
                    total_header_bytes += byte_counts["header_bytes"]
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
        y_bpp = float((total_y_bytes * 8.0) / total_pixels)
        z_bpp = float((total_z_bytes * 8.0) / total_pixels)
        header_bpp = float((total_header_bytes * 8.0) / total_pixels)
        averaged["BPP"] = actual_bpp
        averaged["BPP_actual"] = actual_bpp
        averaged["BPP_payload"] = payload_bpp
        averaged["y_bpp_actual"] = y_bpp
        averaged["z_bpp_actual"] = z_bpp
        averaged["header_bpp"] = header_bpp
        averaged["z_fraction"] = (
            float(total_z_bytes / total_actual_bytes)
            if total_actual_bytes > 0
            else 0.0
        )
        averaged["BPP_payload_minus_estimated"] = (
            payload_bpp - estimated_bpp
        )
        averaged["BPP_actual_minus_estimated"] = actual_bpp - estimated_bpp
        averaged["bitstream_bytes"] = total_actual_bytes
        averaged["payload_bytes"] = total_payload_bytes
        averaged["y_bytes"] = total_y_bytes
        averaged["z_bytes"] = total_z_bytes
        averaged["header_bytes"] = total_header_bytes
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
    if "y_bpp_actual" in point:
        print(
            "  BPP parts: "
            f"y={point['y_bpp_actual']:.4f}, "
            f"z={point['z_bpp_actual']:.4f}, "
            f"header={point['header_bpp']:.4f}"
        )
        print(
            "  z/stream: "
            f"{100.0 * point['z_fraction']:.2f}% | "
            "payload-estimated: "
            f"{point['BPP_payload_minus_estimated']:+.4f} BPP | "
            "file-estimated: "
            f"{point['BPP_actual_minus_estimated']:+.4f} BPP"
        )
    print(f"  PSNR:    {_format_optional_metric(point, 'PSNR', 4)}")
    print(f"  SSIM:    {_format_optional_metric(point, 'SSIM', 4)}")
    print(f"  MS-SSIM: {_format_optional_metric(point, 'MS-SSIM', 4)}")
    print(f"  LPIPS:   {_format_optional_metric(point, 'LPIPS', 4)}")
    print(f"  DISTS:   {_format_optional_metric(point, 'DISTS', 4)}")
    print(f"  MSE:     {_format_optional_metric(point, 'MSE', 6)}")

    bpp_key = round(point.get("BPP", 0.0), 4)
    return {
        "variant": variant_name,
        "bpp_levels": {
            bpp_key: point,
        },
    }
