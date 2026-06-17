"""
ablation.py — ATIC Ablation Study Runner
Each variant is trained at multiple lambda_rate values to produce
a real rate-distortion curve (one point per lambda, not mocked).

Kaggle-friendly usage (no code edits required):
    python ablation.py --epochs 10 --batch-size 4 --lambdas 0.001,0.01,0.1

Environment variable equivalents are also supported, e.g.:
    ATIC_EPOCHS=10
    ATIC_BATCH_SIZE=4
    ATIC_LAMBDAS=0.001,0.01,0.1
"""
import argparse
import csv
import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt

from atic.config  import ArchitectureConfig
from atic.model   import ATICModel
from atic.train   import train_loop
from atic.eval    import eval_single
from atic.dataset import build_and_save_split_manifests, get_video_dataloaders
from atic.metrics import plot_rate_distortion_curves
from atic.repro import (
    get_environment_snapshot,
    set_global_determinism,
    utc_timestamp,
    write_json,
)


# ---------------------------------------------------------------------------
# Ablation variant definitions  (A1 = true baseline, A6 = full ATIC)
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = {
    "Baseline": ArchitectureConfig(
        use_overlapping_patches=False,
        use_sag=False,
        use_cbam=False,
        use_adaptive_quant=False,
        use_hyperprior=True,
    ),
    "No_Overlap": ArchitectureConfig(
        use_overlapping_patches=False,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=True,
    ),
    "No_CBAM": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=False,
        use_adaptive_quant=True,
        use_hyperprior=True,
    ),
    "No_AdaptiveQuant": ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=False,
        use_hyperprior=True,
    ),
    "Full_ATIC": ArchitectureConfig(
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
LAMBDA_RATES = [0.001, 0.01, 0.1]


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value


def _parse_csv_str_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values if values else None


def _parse_csv_int_list(raw: Optional[str], fallback: List[int]) -> List[int]:
    values = _parse_csv_str_list(raw)
    if values is None:
        return fallback
    return [int(v) for v in values]


def _parse_csv_float_list(raw: Optional[str], fallback: List[float]) -> List[float]:
    values = _parse_csv_str_list(raw)
    if values is None:
        return fallback
    return [float(v) for v in values]


def _parse_bool(raw: str) -> bool:
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {raw}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ATIC ablation with CLI/env configurable hyperparameters.",
    )
    parser.add_argument(
        "--video-path",
        default=_env_or_default("ATIC_VIDEO_PATH", "/kaggle/input/datasets/jeevajoji/uvg-honeybee"),
        help="Directory containing PNG frames.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(_env_or_default("ATIC_EPOCHS", "2")),
        help="Training epochs per variant/lambda/seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(_env_or_default("ATIC_BATCH_SIZE", "1")),
        help="Training batch size.",
    )
    parser.add_argument(
        "--device",
        default=_env_or_default("ATIC_DEVICE", "cuda"),
        help="Target device (cuda or cpu).",
    )
    parser.add_argument(
        "--variants",
        default=_env_or_default("ATIC_VARIANTS", ""),
        help="Comma-separated variant names (empty = all).",
    )
    parser.add_argument(
        "--seeds",
        default=_env_or_default("ATIC_SEEDS", "42"),
        help="Comma-separated seeds, e.g. 42,123,999.",
    )
    parser.add_argument(
        "--lambdas",
        default=_env_or_default("ATIC_LAMBDAS", ",".join(str(x) for x in LAMBDA_RATES)),
        help="Comma-separated lambda rates, e.g. 0.001,0.01,0.1.",
    )
    parser.add_argument(
        "--output-root",
        default=_env_or_default("ATIC_OUTPUT_ROOT", "ablation_results/runs"),
        help="Root directory for study outputs.",
    )
    parser.add_argument(
        "--study-name",
        default=_env_or_default("ATIC_STUDY_NAME", "atic_ablation"),
        help="Study name prefix for output folder.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=int(_env_or_default("ATIC_VAL_EVERY", "10")),
        help="Use every Nth frame for validation split.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(_env_or_default("ATIC_NUM_WORKERS", "2")),
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--pin-memory",
        default=_env_or_default("ATIC_PIN_MEMORY", "true"),
        help="Pin DataLoader memory (true/false).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(_env_or_default("ATIC_HEIGHT", "512")),
        help="Training/evaluation image height.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=int(_env_or_default("ATIC_WIDTH", "512")),
        help="Training/evaluation image width.",
    )
    return parser


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------
def visualise_reconstruction(
    model,
    val_loader,
    variant_name,
    lam,
    seed,
    device,
    save_path,
    show=False,
):
    try:
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader)).to(device)
            x_hat = model(batch)["x_hat"]

        x_orig  = batch[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        x_recon = x_hat[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(x_orig);  axes[0].set_title("Original");         axes[0].axis("off")
        axes[1].imshow(x_recon); axes[1].set_title(f"{variant_name} lam={lam} seed={seed}"); axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"Visualisation skipped: {e}")


def _write_summary_csv(csv_path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_points_by_variant_lambda(
    points_by_variant_lambda: Dict[Tuple[str, float], List[Dict]],
) -> Dict[str, Dict[float, Dict]]:
    """Aggregate repeated seed runs into mean RD points for plotting."""
    payload: Dict[str, Dict[float, Dict]] = {}

    for (variant_name, lam), points in points_by_variant_lambda.items():
        if not points:
            continue

        mean_point: Dict[str, float] = {}
        metric_keys = set().union(*(p.keys() for p in points))
        for key in metric_keys:
            vals = [p[key] for p in points if key in p]
            if vals:
                mean_point[key] = float(sum(vals) / len(vals))

        bpp_key = round(mean_point.get("BPP", lam), 4)
        payload.setdefault(variant_name, {})
        while bpp_key in payload[variant_name]:
            bpp_key = round(bpp_key + 1e-4, 4)
        payload[variant_name][bpp_key] = mean_point

    return payload


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_ablation_study(
    video_path: str  = "/kaggle/input/datasets/jeevajoji/uvg-honeybee",
    epochs: int      = 2,
    batch_size: int  = 1,
    device: str      = "cuda",
    # Set to a list of variant names to run only those, e.g. ["A1_Baseline", "A6_FullATIC"]
    run_variants     = None,
    lambda_rates: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    output_root: str = "ablation_results/runs",
    study_name: str = "atic_ablation",
    val_every: int = 10,
    num_workers: int = 2,
    pin_memory: bool = True,
    height: int = 512,
    width: int = 512,

):
    if seeds is None:
        seeds = [42]
    if lambda_rates is None:
        lambda_rates = LAMBDA_RATES

    device = device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    variants_to_run = {
        k: v for k, v in ABLATION_VARIANTS.items()
        if run_variants is None or k in run_variants
    }
    if run_variants is not None:
        missing = [v for v in run_variants if v not in ABLATION_VARIANTS]
        if missing:
            raise ValueError(
                f"Unknown variant(s): {missing}. Available: {list(ABLATION_VARIANTS.keys())}"
            )

    study_dir = os.path.join(output_root, f"{study_name}_{utc_timestamp()}")
    runs_dir = os.path.join(study_dir, "runs")
    plots_dir = os.path.join(study_dir, "plots")
    manifests_dir = os.path.join(study_dir, "manifests")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    train_manifest, val_manifest = build_and_save_split_manifests(
        video_dir=video_path,
        manifest_dir=manifests_dir,
        val_every=val_every,
    )
    if train_manifest is None or val_manifest is None:
        print("No frames found. Check video_path.")
        return

    write_json(
        os.path.join(study_dir, "study_config.json"),
        {
            "study_name": study_name,
            "video_path": os.path.abspath(video_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "seeds": seeds,
            "val_every": val_every,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "lambdas": lambda_rates,
            "variants": {
                variant_name: asdict(config)
                for variant_name, config in variants_to_run.items()
            },
            "manifests": {
                "train": train_manifest,
                "val": val_manifest,
            },
            "environment": get_environment_snapshot(device=device, repo_dir=os.getcwd()),
        },
    )

    summary_rows: List[Dict] = []
    points_by_variant_lambda: Dict[Tuple[str, float], List[Dict]] = {}
    summary_csv_path = os.path.join(study_dir, "summary_metrics.csv")
    summary_json_path = os.path.join(study_dir, "summary_metrics.json")

    for seed in seeds:
        set_global_determinism(seed=seed, deterministic=True)

        train_loader, val_loader = get_video_dataloaders(
            video_dir=video_path,
            batch_size=batch_size,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        if train_loader is None or val_loader is None:
            print("Could not build dataloaders from manifests.")
            return

        for variant_name, config in variants_to_run.items():
            print(f"\n{'='*55}")
            print(f"Seed: {seed} | Variant: {variant_name}")
            print(f"{'='*55}")

            for lam in lambda_rates:
                print(f"\n  --- lambda = {lam} ---")

                run_dir = os.path.join(
                    runs_dir,
                    variant_name,
                    f"lam_{lam}",
                    f"seed_{seed}",
                )
                os.makedirs(run_dir, exist_ok=True)
                write_json(
                    os.path.join(run_dir, "run_config.json"),
                    {
                        "variant": variant_name,
                        "lambda_rate": lam,
                        "seed": seed,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "architecture": asdict(config),
                        "device": device,
                        "manifest_paths": {
                            "train": train_manifest,
                            "val": val_manifest,
                        },
                    },
                )

                write_json(
                    os.path.join(run_dir, "environment.json"),
                    get_environment_snapshot(device=device, repo_dir=os.getcwd()),
                )

                # Fresh model for every (variant, lambda, seed) combination.
                model = ATICModel(config, H=height, W=width).to(device)

                train_artifacts = train_loop(
                    model,
                    variant_name=f"{variant_name}_lam{lam}_seed{seed}",
                    dataloader=train_loader,\
                    val_loader=val_loader,
                    epochs=epochs,
                    device=device,
                    lambda_rate=lam,
                    checkpoint_path=os.path.join(run_dir, "model.pth"),
                    train_log_path=os.path.join(run_dir, "train_log.jsonl"),
                )

                point = eval_single(
                    model,
                    val_loader,
                    device=device,
                    bitstream_dir=os.path.join(run_dir, "bitstreams"),
                )
                write_json(os.path.join(run_dir, "eval_metrics.json"), point)

                bpp_key = round(point.get("BPP", lam), 4)
                points_by_variant_lambda.setdefault((variant_name, lam), []).append(point)

                print(
                    f"  BPP={bpp_key:.4f} | "
                    f"PSNR={point.get('PSNR', 0):.2f} | "
                    f"SSIM={point.get('SSIM', 0):.4f} | "
                    f"LPIPS={point.get('LPIPS', 0):.4f}"
                )

                visualise_reconstruction(
                    model=model,
                    val_loader=val_loader,
                    variant_name=variant_name,
                    lam=lam,
                    seed=seed,
                    device=device,
                    save_path=os.path.join(run_dir, "reconstruction.png"),
                    show=False,
                )

                summary_row = {
                    "variant": variant_name,
                    "seed": seed,
                    "lambda_rate": lam,
                    "checkpoint_path": train_artifacts.get("checkpoint_path"),
                }
                summary_row.update(point)
                summary_rows.append(summary_row)

                _write_summary_csv(summary_csv_path, summary_rows)
                write_json(summary_json_path, {"rows": summary_rows})

                rd_payload = _aggregate_points_by_variant_lambda(points_by_variant_lambda)
                try:
                    plot_rate_distortion_curves(rd_payload, save_dir=plots_dir)
                except Exception as e:
                    print(f"Incremental plot skipped: {e}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    final_payload = _aggregate_points_by_variant_lambda(points_by_variant_lambda)
    write_json(os.path.join(study_dir, "rd_aggregate.json"), final_payload)
    print(f"\nAll variants complete. Study artifacts saved to {study_dir}")
    return {
        "study_dir": study_dir,
        "summary_rows": summary_rows,
        "rd_payload": final_payload,
    }


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    variants = _parse_csv_str_list(args.variants)
    seeds = _parse_csv_int_list(args.seeds, fallback=[42])
    lambdas = _parse_csv_float_list(args.lambdas, fallback=LAMBDA_RATES)
    pin_memory = _parse_bool(args.pin_memory)

    run_ablation_study(
        video_path=args.video_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        run_variants=variants,
        lambda_rates=lambdas,
        seeds=seeds,
        output_root=args.output_root,
        study_name=args.study_name,
        val_every=args.val_every,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        height=args.height,
        width=args.width
    )