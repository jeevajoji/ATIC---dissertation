import torch
import os
from atic.metrics import ATICMetrics
from atic.bitstream import encode_latent_payload, bpp_from_num_bytes


def _proxy_bpp_from_likelihoods(likelihoods, num_pixels):
    bpp = 0.0
    if likelihoods is None:
        return bpp
    for lik in likelihoods.values():
        bpp += -torch.log2(lik.clamp(min=1e-9)).sum().item() / num_pixels
    return bpp


def eval_single(model, dataloader, device="cuda", bitstream_dir=None):
    """
    Evaluates a single trained model on the validation set.
    Returns one dict of averaged metrics including REAL BPP from encoded bytes,
    plus likelihood-proxy BPP for analysis.
    This is called once per (variant, lambda) pair.
    """
    model.eval()
    model.to(device)
    metric_calculator = ATICMetrics(device=device)

    totals = {"PSNR": 0, "SSIM": 0, "MS-SSIM": 0,
              "LPIPS": 0, "DISTS": 0, "MSE": 0,
              "BPP": 0, "BPP_REAL": 0, "BPP_PROXY": 0,
              "BITSTREAM_BYTES": 0}
    counts = {k: 0 for k in totals}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            outputs = model(batch)

            x_hat      = outputs["x_hat"]
            likelihoods = outputs["likelihoods"]
            z_hat = outputs.get("z_hat")
            h_hat = outputs.get("hyper_hat")

            # 1) Proxy BPP from likelihoods (training objective view)
            N, _, H, W = batch.shape
            num_pixels = N * H * W
            proxy_bpp = _proxy_bpp_from_likelihoods(likelihoods, num_pixels)

            # 2) REAL BPP from encoded payload bytes (headline metric)
            bitstream_path = None
            if bitstream_dir is not None:
                os.makedirs(bitstream_dir, exist_ok=True)
                bitstream_path = os.path.join(bitstream_dir, f"batch_{batch_idx:05d}.npz")

            if z_hat is not None:
                payload_stats = encode_latent_payload(
                    z_hat=z_hat,
                    h_hat=h_hat,
                    save_path=bitstream_path,
                )
                real_bpp = bpp_from_num_bytes(payload_stats["num_bytes"], batch)
                bitstream_bytes = payload_stats["num_bytes"]
            else:
                # Fallback for backward compatibility if model doesn't expose z_hat.
                real_bpp = proxy_bpp
                bitstream_bytes = 0.0

            batch_metrics = metric_calculator.compute_all(x_hat, batch, bpp=real_bpp)
            batch_metrics["BPP"] = real_bpp
            batch_metrics["BPP_REAL"] = real_bpp
            batch_metrics["BPP_PROXY"] = proxy_bpp
            batch_metrics["BITSTREAM_BYTES"] = bitstream_bytes

            for k, v in batch_metrics.items():
                if k in totals:
                    totals[k] += v
                    counts[k] += 1

    return {k: totals[k] / counts[k] for k in totals if counts[k] > 0}


def eval_loop(model, variant_name, dataloader, device="cuda", bitstream_dir=None):
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

    point = eval_single(model, dataloader, device=device, bitstream_dir=bitstream_dir)

    print(f"  BPP:     {point.get('BPP',  0):.4f}")
    print(f"  BPP_REAL:{point.get('BPP_REAL', 0):.4f}")
    print(f"  BPP_PROXY:{point.get('BPP_PROXY', 0):.4f}")
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