import torch
import math
from atic.metrics import ATICMetrics


def eval_single(model, dataloader, device="cuda"):
    """
    Evaluates a single trained model on the validation set.
    Returns one dict of averaged metrics including real BPP from likelihoods.
    This is called once per (variant, lambda) pair.
    """
    model.eval()
    model.to(device)
    metric_calculator = ATICMetrics(device=device)

    totals = {"PSNR": 0, "SSIM": 0, "MS-SSIM": 0,
              "LPIPS": 0, "DISTS": 0, "MSE": 0, "BPP": 0}
    counts = {k: 0 for k in totals}

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)

            x_hat      = outputs["x_hat"]
            likelihoods = outputs["likelihoods"]

            # Compute REAL BPP from entropy likelihoods
            N, _, H, W = batch.shape
            num_pixels = N * H * W
            bpp = 0.0
            if likelihoods is not None:
                for lik in likelihoods.values():
                    bpp += -torch.log2(
                        lik.clamp(min=1e-9)
                    ).sum().item() / num_pixels

            batch_metrics = metric_calculator.compute_all(x_hat, batch, bpp=bpp)
            batch_metrics["BPP"] = bpp

            for k, v in batch_metrics.items():
                if k in totals:
                    totals[k] += v
                    counts[k] += 1

    return {k: totals[k] / counts[k] for k in totals if counts[k] > 0}


def eval_loop(model, variant_name, dataloader, device="cuda"):
    """
    Thin wrapper kept for backward compatibility with ablation.py.
    Calls eval_single and wraps result in the format ablation.py expects.
    """
    if dataloader is None:
        print(f"[{variant_name}] Dataloader not found. Skipping eval.")
        return None

    print(f"\n{'='*45}")
    print(f"[{variant_name}] Evaluating...")
    print(f"{'='*45}")

    point = eval_single(model, dataloader, device=device)

    print(f"  BPP:     {point.get('BPP',  0):.4f}")
    print(f"  PSNR:    {point.get('PSNR', 0):.4f}")
    print(f"  SSIM:    {point.get('SSIM', 0):.4f}")
    print(f"  MS-SSIM: {point.get('MS-SSIM', 0):.4f}")
    print(f"  LPIPS:   {point.get('LPIPS', 0):.4f}")
    print(f"  DISTS:   {point.get('DISTS', 0):.4f}")

    # Package in the shape ablation.py reads: {bpp_value: metrics_dict}
    bpp_key = round(point.get("BPP", 0.0), 4)
    return {
        "variant"   : variant_name,
        "bpp_levels": {bpp_key: point},
    }