"""
ablation.py — ATIC Ablation Study Runner
Each variant is trained at multiple lambda_rate values to produce
a real rate-distortion curve (one point per lambda, not mocked).
"""
import os
import torch
import matplotlib.pyplot as plt

from atic.config  import ArchitectureConfig
from atic.model   import ATICModel
from atic.train   import train_loop
from atic.eval    import eval_single
from atic.dataset import get_video_dataloaders
from atic.metrics import plot_rate_distortion_curves


# ---------------------------------------------------------------------------
# Ablation variant definitions  (A1 = true baseline, A6 = full ATIC)
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = {
    "A1_Baseline": ArchitectureConfig(
        use_overlapping_patches=False,
        use_sag=False,
        use_cbam=False,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    "A2_Overlap": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=False,
        use_cbam=False,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    "A3_SAG": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=False,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    "A4_CBAM": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    "A5_AdaptiveQuant": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=False,
    ),
    "A6_FullATIC": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=True,
    ),
}

# Each lambda produces one point on the RD curve.
# Lower lambda  → model uses more bits → higher quality (high BPP, high PSNR)
# Higher lambda → model uses fewer bits → lower quality (low BPP, low PSNR)
LAMBDA_RATES = [0.001, 0.005, 0.01, 0.05, 0.1]


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------
def visualise_reconstruction(model, val_loader, variant_name, lam, device):
    try:
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader)).to(device)
            x_hat = model(batch)["x_hat"]

        x_orig  = batch[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        x_recon = x_hat[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(x_orig);  axes[0].set_title("Original");         axes[0].axis("off")
        axes[1].imshow(x_recon); axes[1].set_title(f"{variant_name} λ={lam}"); axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(f"ablation_results/{variant_name}_lam{lam}_recon.png", dpi=150)
        plt.show()
    except Exception as e:
        print(f"Visualisation skipped: {e}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_ablation_study(
    video_path: str  = "/kaggle/input/datasets/jeevajoji/uvg-honeybee",
    epochs: int      = 10,
    device: str      = "cuda",
    # Set to a list of variant names to run only those, e.g. ["A1_Baseline", "A6_FullATIC"]
    run_variants     = None,
):
    os.makedirs("ablation_results/plots", exist_ok=True)

    torch.manual_seed(42)
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader = get_video_dataloaders(
        video_dir=video_path, batch_size=1
    )
    if train_loader is None:
        print("No frames found. Check video_path.")
        return

    # all_results[variant_name][bpp_value] = metrics_dict
    # This is the format plot_rate_distortion_curves expects.
    all_results = {}

    variants_to_run = {
        k: v for k, v in ABLATION_VARIANTS.items()
        if run_variants is None or k in run_variants
    }

    for variant_name, config in variants_to_run.items():
        print(f"\n{'='*55}")
        print(f"Variant: {variant_name}")
        print(f"{'='*55}")

        all_results[variant_name] = {}

        for lam in LAMBDA_RATES:
            print(f"\n  --- lambda = {lam} ---")

            # Fresh model for every (variant, lambda) combination.
            # This is essential — each point on the RD curve is a separately
            # trained model optimised for a different rate-distortion tradeoff.
            model = ATICModel(config, H=1088, W=1920).to(device)

            train_loop(
                model,
                variant_name=f"{variant_name}_lam{lam}",
                dataloader=train_loader,
                epochs=epochs,
                device=device,
                lambda_rate=lam,
            )

            # eval_single returns real BPP computed from likelihoods
            point = eval_single(model, val_loader, device=device)
            bpp_key = round(point.get("BPP", lam), 4)
            all_results[variant_name][bpp_key] = point

            print(f"  BPP={bpp_key:.4f} | "
                  f"PSNR={point.get('PSNR',0):.2f} | "
                  f"SSIM={point.get('SSIM',0):.4f} | "
                  f"LPIPS={point.get('LPIPS',0):.4f}")

            # Show reconstruction for the middle lambda only (saves time)
            if lam == LAMBDA_RATES[len(LAMBDA_RATES) // 2]:
                visualise_reconstruction(model, val_loader, variant_name, lam, device)

            # Incremental RD plot after each lambda
            try:
                plot_rate_distortion_curves(all_results)
            except Exception as e:
                print(f"Incremental plot skipped: {e}")

            del model
            torch.cuda.empty_cache()

    print("\nAll variants complete. RD curves saved to ablation_results/plots/")
    return all_results


if __name__ == "__main__":
    # Quick test: run only A1 and A6 with 3 lambdas to validate pipeline
    # then swap run_variants=None for the full study
    run_ablation_study(
        epochs=10,
        run_variants=["A1_Baseline", "A6_FullATIC"],
    )