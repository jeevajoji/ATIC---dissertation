"""
ablation.py — ATIC Ablation Study Runner
Correct A1→A6 ladder: each stage adds exactly one component.
"""
import os
import torch
from atic.config import ArchitectureConfig
from atic.model  import ATICModel
from atic.train  import train_loop
from atic.eval   import eval_loop
from atic.dataset import get_video_dataloaders


ABLATION_VARIANTS = {
    # A1: non-overlapping patches, no attention, uniform quant, no hyperprior
    # This is the true baseline — vanilla Swin codec
    "A1_Baseline": ArchitectureConfig(
        use_overlapping_patches=False,
        use_sag=False,
        use_cbam=False,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    # A2: add overlapping patches only
    "A2_Overlap": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=False,
        use_cbam=False,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    # A3: + Spatial Attention Gate
    "A3_SAG": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=False,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    # A4: + CBAM channel attention
    "A4_CBAM": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=False,
        use_hyperprior=False,
    ),
    # A5: + attention-guided adaptive quantizer
    "A5_AdaptiveQuant": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=False,
    ),
    # A6: + hyperprior entropy model — full ATIC
    "A6_FullATIC": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=True,
    ),
}


def run_ablation_study(
    video_path: str = "UVG frames/honeybee",
    epochs: int = 10,
    lambda_rate: float = 0.01,
    device: str = "cuda",
):
    os.makedirs("ablation_results", exist_ok=True)
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader = get_video_dataloaders(
        video_dir=video_path, batch_size=1
    )

    all_results = {}

    for variant_name, config in ABLATION_VARIANTS.items():
        print(f"\n{'='*55}")
        print(f"  Variant: {variant_name}")
        print(f"{'='*55}")

        model = ATICModel(config, H=1088, W=1920)

        train_loop(
            model, variant_name, train_loader,
            epochs=epochs, device=device, lambda_rate=lambda_rate,
        )

        results = eval_loop(
            model, variant_name, val_loader, device=device
        )

        if results:
            all_results[variant_name] = results.get("bpp_levels", {})

    # Plot RD curves across all variants
    try:
        from atic.metrics import plot_rate_distortion_curves
        plot_rate_distortion_curves(all_results)
        print("\nRD curves saved to ablation_results/plots/")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    return all_results


if __name__ == "__main__":
    run_ablation_study()