import torch

from atic.metrics import ATICMetrics


def _extract_images(batch):
    """
    Supports both:
        Dataset returning image only:
            batch = tensor

        Dataset returning image and path:
            batch = (tensor, path)
    """
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _bpp_from_likelihoods(likelihoods, batch, eps: float = 1e-9) -> float:
    """
    Computes likelihood-based BPP for evaluation.

    BPP = sum(-log2(likelihoods)) / number_of_pixels
    """
    if likelihoods is None:
        return 0.0

    N, _, H, W = batch.shape
    num_pixels = N * H * W

    bpp = 0.0

    if isinstance(likelihoods, dict):
        for likelihood in likelihoods.values():
            if likelihood is None:
                continue
            bpp += (
                -torch.log2(likelihood.clamp(min=eps)).sum().item()
                / num_pixels
            )

    elif isinstance(likelihoods, torch.Tensor):
        bpp = (
            -torch.log2(likelihoods.clamp(min=eps)).sum().item()
            / num_pixels
        )

    else:
        raise TypeError(
            f"Unsupported likelihoods type: {type(likelihoods)}. "
            "Expected dict, Tensor, or None."
        )

    return float(bpp)


def eval_single(model, dataloader, device="cuda", bitstream_dir=None):
    """
    Evaluates a single trained model on the validation/test set.

    Important:
        This version uses ONE headline BPP:
            BPP = likelihood-based BPP from CompressAI entropy models.

        The old npz payload byte BPP has been removed to avoid confusing
        the scientific rate-distortion reporting.

    Args:
        model:
            ATICModel.
        dataloader:
            Validation/test dataloader.
        device:
            cuda or cpu.
        bitstream_dir:
            Kept only for backward compatibility with older ablation.py.
            It is not used.

    Returns:
        Averaged metrics dictionary:
            {
                "BPP": ...,
                "PSNR": ...,
                "SSIM": ...,
                "MS-SSIM": ...,
                "LPIPS": ...,
                "DISTS": ...,
                "MSE": ...
            }
    """
    if dataloader is None:
        return {}

    model.to(device)
    model.eval()

    # Update entropy CDF tables if available.
    # For likelihood evaluation this is not always required, but it is safe.
    if hasattr(model, "update"):
        try:
            model.update(force=True)
        except Exception as e:
            print(f"Warning: model.update(force=True) failed during eval: {e}")

    metric_calculator = ATICMetrics(device=device)

    totals = {
        "BPP": 0.0,
        "PSNR": 0.0,
        "SSIM": 0.0,
        "MS-SSIM": 0.0,
        "LPIPS": 0.0,
        "DISTS": 0.0,
        "MSE": 0.0,
    }

    counts = {k: 0 for k in totals}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _extract_images(batch).to(device, non_blocking=True)

            outputs = model(batch)

            x_hat = outputs["x_hat"]
            likelihoods = outputs.get("likelihoods", None)

            bpp = _bpp_from_likelihoods(likelihoods, batch)

            batch_metrics = metric_calculator.compute_all(
                x_hat=x_hat,
                x=batch,
                bpp=bpp,
            )

            # Ensure BPP is always present and is the likelihood-based BPP.
            batch_metrics["BPP"] = bpp

            for key, value in batch_metrics.items():
                if key in totals and value is not None:
                    totals[key] += float(value)
                    counts[key] += 1

    averaged = {
        key: totals[key] / counts[key]
        for key in totals
        if counts[key] > 0
    }

    return averaged


def eval_loop(model, variant_name, dataloader, device="cuda", bitstream_dir=None):
    """
    Thin wrapper kept for backward compatibility with ablation.py.

    Returns:
        {
            "variant": variant_name,
            "bpp_levels": {
                bpp_value: metrics_dict
            }
        }
    """
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
    )

    print(f"  BPP:     {point.get('BPP', 0):.4f}")
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