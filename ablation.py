"""
ablation.py — ATIC Ablation Study Runner
Each variant is trained at multiple rate-distortion lambda values to produce
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
import math
import os
import shutil
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch

from atic.config  import ArchitectureConfig


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------
def _full_atic_config() -> ArchitectureConfig:
    """Return the shared architecture for the causal DSAD comparison."""

    return ArchitectureConfig(
        use_overlapping_patches=True,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=True,
    )


# These names are retained so archived experiments and old commands remain
# readable. They are architecture ablations, not the controlled DSAD claim.
HISTORICAL_VARIANTS = (
    "Baseline",
    "No_Overlap",
    "No_CBAM",
    "No_AdaptiveQuant",
    "Full_ATIC",
)

# This is the causal comparison for the paper: architecture, data, seed,
# lambda, and schedule are identical. Only the maximum DSAD beta differs.
CAUSAL_DSAD_VARIANTS = (
    "Full_ATIC_NoDSAD",
    "Full_ATIC_DSAD",
)

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
    "Full_ATIC": _full_atic_config(),
    "Full_ATIC_NoDSAD": _full_atic_config(),
    "Full_ATIC_DSAD": _full_atic_config(),
}

# Each lambda produces one point on the RD curve.
# Higher lambda gives distortion more weight, normally producing higher
# quality at a higher bitrate under the standard BPP + lambda*distortion loss.
LAMBDA_RATES = [0.001, 0.01, 0.1]


def _variant_family(variant_name: str) -> str:
    if variant_name in CAUSAL_DSAD_VARIANTS:
        return "causal_dsad_ablation"
    return "historical_architecture_ablation"


def _validate_dsad_hyperparameters(
    beta_max: float,
    warmup_fraction: float,
    ramp_fraction: float,
) -> None:
    values = {
        "dsad_beta_max": beta_max,
        "dsad_warmup_fraction": warmup_fraction,
        "dsad_ramp_fraction": ramp_fraction,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if beta_max < 0:
        raise ValueError("dsad_beta_max must be non-negative")
    if not 0.0 <= warmup_fraction <= 1.0:
        raise ValueError("dsad_warmup_fraction must be between 0 and 1")
    if not 0.0 <= ramp_fraction <= 1.0:
        raise ValueError("dsad_ramp_fraction must be between 0 and 1")
    if warmup_fraction + ramp_fraction > 1.0:
        raise ValueError(
            "dsad_warmup_fraction + dsad_ramp_fraction must not exceed 1"
        )


def _validate_study_hyperparameters(
    *,
    epochs: int,
    batch_size: int,
    height: int,
    width: int,
    val_every: int,
    num_workers: int,
    seeds: List[int],
    lambda_rates: List[float],
    run_variants: Optional[List[str]],
) -> None:
    positive_integers = {
        "epochs": epochs,
        "batch_size": batch_size,
        "height": height,
        "width": width,
    }
    for name, value in positive_integers.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(val_every, int) or isinstance(val_every, bool):
        raise ValueError("val_every must be an integer")
    if val_every < 2:
        raise ValueError(
            "val_every must be at least 2 so the training split is non-empty"
        )
    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")

    if not seeds:
        raise ValueError("At least one seed is required")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("Every seed must be an integer")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique to avoid overwriting run folders")

    if not lambda_rates:
        raise ValueError("At least one lambda_rd value is required")
    for value in lambda_rates:
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "Every lambda_rd value must be finite and non-negative"
            )
    if len(lambda_rates) != len(set(lambda_rates)):
        raise ValueError(
            "lambda_rd values must be unique to avoid overwriting run folders"
        )

    if run_variants is not None:
        if not run_variants:
            raise ValueError(
                "run_variants must contain at least one variant or be None"
            )
        if len(run_variants) != len(set(run_variants)):
            raise ValueError(
                "Variant names must be unique to avoid ambiguous repeated requests"
            )


def _manifest_entry_count(manifest_path: str) -> int:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _format_optional_metric(
    point: Dict[str, object],
    name: str,
    precision: int,
) -> str:
    """Format a metric without turning an unavailable dependency into zero."""

    value = point.get(name)
    return "unavailable" if value is None else f"{float(value):.{precision}f}"


def _dsad_settings_for_variant(
    variant_name: str,
    beta_max: float,
    warmup_fraction: float,
    ramp_fraction: float,
) -> Dict[str, object]:
    """Return the controlled training settings for one named variant."""

    _validate_dsad_hyperparameters(
        beta_max,
        warmup_fraction,
        ramp_fraction,
    )
    is_dsad_arm = variant_name == "Full_ATIC_DSAD"
    effective_beta = beta_max if is_dsad_arm else 0.0
    return {
        "comparison_role": (
            "distilled_student"
            if is_dsad_arm
            else (
                "identical_backbone_control"
                if variant_name == "Full_ATIC_NoDSAD"
                else "not_a_causal_dsad_arm"
            )
        ),
        "beta_max": effective_beta,
        "warmup_fraction": warmup_fraction,
        "ramp_fraction": ramp_fraction,
    }


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


def _validate_dataset_protocol(
    dataset_root: Optional[str],
    frozen_split_dir: Optional[str],
    evaluate_test: bool,
) -> bool:
    """Validate the legacy-pilot versus frozen-publication data contract."""

    has_root = bool(dataset_root and dataset_root.strip())
    has_bundle = bool(frozen_split_dir and frozen_split_dir.strip())
    if has_root != has_bundle:
        raise ValueError(
            "--dataset-root and --frozen-split-dir must be supplied together"
        )
    if evaluate_test and not has_bundle:
        raise ValueError(
            "--evaluate-test is allowed only with a verified frozen split bundle"
        )
    return has_bundle


def _evaluation_plan(evaluate_test: bool) -> Tuple[str, ...]:
    """Return the only dataset splits permitted to reach final evaluation."""

    return ("val", "test") if evaluate_test else ("val",)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ATIC ablation with CLI/env configurable hyperparameters.",
    )
    parser.add_argument(
        "--video-path",
        default=_env_or_default("ATIC_VIDEO_PATH", "/kaggle/input/datasets/jeevajoji/uvg-honeybee"),
        help=(
            "Legacy pilot directory containing PNG frames. This interleaved "
            "mode is not publication-eligible."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("ATIC_DATASET_ROOT"),
        help=(
            "Dataset root used by a frozen sequence-disjoint split bundle. "
            "Must be paired with --frozen-split-dir."
        ),
    )
    parser.add_argument(
        "--frozen-split-dir",
        default=os.getenv("ATIC_FROZEN_SPLIT_DIR"),
        help=(
            "Verified bundle created by `python -m atic.split_cli create`. "
            "Must be paired with --dataset-root."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        default=False,
        help=(
            "Explicitly unlock the frozen independent test split. This flag "
            "has no environment-variable shortcut; leave it disabled during "
            "model and hyperparameter selection."
        ),
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
        default=_env_or_default(
            "ATIC_VARIANTS",
            ",".join(CAUSAL_DSAD_VARIANTS),
        ),
        help=(
            "Comma-separated variant names. Defaults to the controlled "
            "Full_ATIC_NoDSAD,Full_ATIC_DSAD pair; pass an empty value to "
            "include every registered variant, including historical "
            "architecture ablations."
        ),
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(_env_or_default("ATIC_LEARNING_RATE", "0.0001")),
        help="Initial main-optimizer learning rate.",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=float(
            _env_or_default("ATIC_AUX_LEARNING_RATE", "0.001")
        ),
        help="Initial entropy-bottleneck auxiliary learning rate.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "cosine"),
        default=_env_or_default("ATIC_LR_SCHEDULE", "cosine"),
        help=(
            "Deterministic learning-rate schedule. Publication runs default "
            "to cosine decay; use none only for legacy reproduction."
        ),
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=float(
            _env_or_default("ATIC_MIN_LEARNING_RATE", "0.000001")
        ),
        help="Minimum main LR reached by cosine decay.",
    )
    parser.add_argument(
        "--min-aux-learning-rate",
        type=float,
        default=float(
            _env_or_default("ATIC_MIN_AUX_LEARNING_RATE", "0.00001")
        ),
        help="Minimum auxiliary LR reached by cosine decay.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("final", "best_val_rd"),
        default=_env_or_default(
            "ATIC_CHECKPOINT_SELECTION",
            "best_val_rd",
        ),
        help=(
            "Checkpoint used for bitstream evaluation. best_val_rd is "
            "selected only from the validation split."
        ),
    )
    parser.add_argument(
        "--dsad-beta-max",
        type=float,
        default=float(_env_or_default("ATIC_DSAD_BETA_MAX", "0.05")),
        help=(
            "Maximum distillation weight for Full_ATIC_DSAD. The matched "
            "Full_ATIC_NoDSAD control always uses zero."
        ),
    )
    parser.add_argument(
        "--dsad-warmup-fraction",
        type=float,
        default=float(
            _env_or_default("ATIC_DSAD_WARMUP_FRACTION", "0.20")
        ),
        help="Fraction of epochs with beta held at zero (default: 0.20).",
    )
    parser.add_argument(
        "--dsad-ramp-fraction",
        type=float,
        default=float(
            _env_or_default("ATIC_DSAD_RAMP_FRACTION", "0.10")
        ),
        help="Fraction of epochs used to ramp beta to its maximum (default: 0.10).",
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
        import matplotlib.pyplot as plt

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
    learning_rate: float = 1e-4,
    aux_learning_rate: float = 1e-3,
    lr_schedule: str = "cosine",
    min_learning_rate: float = 1e-6,
    min_aux_learning_rate: float = 1e-5,
    checkpoint_selection: str = "best_val_rd",
    dsad_beta_max: float = 0.05,
    dsad_warmup_fraction: float = 0.20,
    dsad_ramp_fraction: float = 0.10,
    dataset_root: Optional[str] = None,
    frozen_split_dir: Optional[str] = None,
    evaluate_test: bool = False,
):
    # Keep plotting/training-only dependencies lazy so configuration inspection
    # and ``--help`` work in lightweight codec environments.
    from atic.dataset import (
        build_and_save_split_manifests,
        get_frozen_split_dataloaders,
        get_video_dataloaders,
        load_and_verify_frozen_split_bundle,
    )
    from atic.eval import eval_single
    from atic.metrics import plot_rate_distortion_curves
    from atic.model import ATICModel
    from atic.repro import (
        get_environment_snapshot,
        hash_model_state,
        set_global_determinism,
        utc_timestamp,
        write_json,
    )
    from atic.train import (
        _validate_training_controls,
        dsad_beta_for_epoch,
        train_loop,
    )

    if seeds is None:
        seeds = [42]
    if lambda_rates is None:
        lambda_rates = LAMBDA_RATES
    _validate_dsad_hyperparameters(
        dsad_beta_max,
        dsad_warmup_fraction,
        dsad_ramp_fraction,
    )
    _validate_study_hyperparameters(
        epochs=epochs,
        batch_size=batch_size,
        height=height,
        width=width,
        val_every=val_every,
        num_workers=num_workers,
        seeds=seeds,
        lambda_rates=lambda_rates,
        run_variants=run_variants,
    )
    _validate_training_controls(
        learning_rate=learning_rate,
        aux_learning_rate=aux_learning_rate,
        lr_schedule=lr_schedule,
        min_learning_rate=min_learning_rate,
        min_aux_learning_rate=min_aux_learning_rate,
        checkpoint_selection=checkpoint_selection,
        checkpoint_selection_start_epoch=1,
        epochs=epochs,
    )
    use_frozen_protocol = _validate_dataset_protocol(
        dataset_root,
        frozen_split_dir,
        evaluate_test,
    )
    evaluation_plan = _evaluation_plan(evaluate_test)
    checkpoint_selection_start_epoch = 1
    if checkpoint_selection == "best_val_rd" and dsad_beta_max > 0:
        checkpoint_selection_start_epoch = next(
            epoch + 1
            for epoch in range(epochs)
            if math.isclose(
                dsad_beta_for_epoch(
                    epoch,
                    epochs,
                    dsad_beta_max,
                    dsad_warmup_fraction,
                    dsad_ramp_fraction,
                ),
                dsad_beta_max,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device}) but CUDA is unavailable; "
            "refusing a silent CPU fallback."
        )
    print(f"Device: {device}")
    if evaluate_test:
        print("!" * 72)
        print(
            "INDEPENDENT TEST SPLIT UNLOCKED: do not use these results for "
            "hyperparameter or checkpoint selection."
        )
        print("!" * 72)

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

    frozen_bundle = None
    if use_frozen_protocol:
        assert dataset_root is not None
        assert frozen_split_dir is not None
        frozen_bundle = load_and_verify_frozen_split_bundle(
            split_dir=frozen_split_dir,
            dataset_root=dataset_root,
            expected_size=(width, height),
        )
        shutil.copytree(frozen_bundle.split_dir, manifests_dir)
        train_manifest = os.path.join(
            manifests_dir,
            os.path.basename(frozen_bundle.splits["train"].manifest_path),
        )
        val_manifest = os.path.join(
            manifests_dir,
            os.path.basename(frozen_bundle.splits["val"].manifest_path),
        )
        test_manifest = os.path.join(
            manifests_dir,
            os.path.basename(frozen_bundle.splits["test"].manifest_path),
        )
        train_image_count = len(frozen_bundle.splits["train"].image_paths)
        val_image_count = len(frozen_bundle.splits["val"].image_paths)
        test_image_count = len(frozen_bundle.splits["test"].image_paths)
        data_protocol = "frozen_declared_sequence_groups_v1"
        manifest_config = {
            "bundle_id": frozen_bundle.bundle_id,
            "dataset_id": frozen_bundle.dataset_id,
            "dataset_root": frozen_bundle.dataset_root,
            "train": train_manifest,
            "val": val_manifest,
            "test": test_manifest,
            "hashes": {
                split_name: {
                    "manifest_sha256": split.manifest_sha256,
                    "file_sha256": split.file_sha256,
                    "content_sha256": split.content_sha256,
                    "sequences": list(split.sequences),
                }
                for split_name, split in frozen_bundle.splits.items()
            },
        }
    else:
        train_manifest, val_manifest = build_and_save_split_manifests(
            video_dir=video_path,
            manifest_dir=manifests_dir,
            val_every=val_every,
        )
        if train_manifest is None or val_manifest is None:
            raise FileNotFoundError(f"No PNG frames found in {video_path}")
        train_image_count = _manifest_entry_count(train_manifest)
        val_image_count = _manifest_entry_count(val_manifest)
        test_manifest = None
        test_image_count = 0
        if train_image_count == 0 or val_image_count == 0:
            raise ValueError(
                "The pilot requires non-empty train and validation splits; "
                f"found train={train_image_count}, validation={val_image_count}. "
                "Provide more frames or choose a smaller --val-every value."
            )
        data_protocol = "legacy_frame_interleaved_nonpublication"
        manifest_config = {
            "train": train_manifest,
            "val": val_manifest,
            "test": None,
        }

    write_json(
        os.path.join(study_dir, "study_config.json"),
        {
            "study_name": study_name,
            "data_protocol": data_protocol,
            "frozen_split_verified": use_frozen_protocol,
            "manual_sequence_audit_required": use_frozen_protocol,
            "test_evaluation_enabled": evaluate_test,
            "evaluation_split": "test" if evaluate_test else "val",
            "video_path": (
                None if use_frozen_protocol else os.path.abspath(video_path)
            ),
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "training_controls": {
                "learning_rate": learning_rate,
                "aux_learning_rate": aux_learning_rate,
                "lr_schedule": lr_schedule,
                "min_learning_rate": min_learning_rate,
                "min_aux_learning_rate": min_aux_learning_rate,
                "checkpoint_selection": checkpoint_selection,
                "causal_checkpoint_selection_start_epoch": (
                    checkpoint_selection_start_epoch
                ),
                "checkpoint_selection_metric": (
                    "val_rd_loss"
                    if checkpoint_selection == "best_val_rd"
                    else None
                ),
            },
            "seeds": seeds,
            "val_every": val_every,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "lambdas": lambda_rates,
            "split_counts": {
                "train": train_image_count,
                "val": val_image_count,
                "test": test_image_count,
            },
            "dsad_schedule": {
                "requested_beta_max": dsad_beta_max,
                "warmup_fraction": dsad_warmup_fraction,
                "ramp_fraction": dsad_ramp_fraction,
            },
            "variants": {
                variant_name: {
                    "experiment_family": _variant_family(variant_name),
                    "architecture": asdict(config),
                    "dsad": _dsad_settings_for_variant(
                        variant_name,
                        dsad_beta_max,
                        dsad_warmup_fraction,
                        dsad_ramp_fraction,
                    ),
                }
                for variant_name, config in variants_to_run.items()
            },
            "manifests": manifest_config,
            "environment": get_environment_snapshot(device=device, repo_dir=os.getcwd()),
        },
    )

    summary_rows: List[Dict] = []
    points_by_variant_lambda: Dict[Tuple[str, float], List[Dict]] = {}
    summary_csv_path = os.path.join(study_dir, "summary_metrics.csv")
    summary_json_path = os.path.join(study_dir, "summary_metrics.json")

    for seed in seeds:
        set_global_determinism(seed=seed, deterministic=True)

        if frozen_bundle is not None:
            loaders = get_frozen_split_dataloaders(
                frozen_bundle,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                seed=seed,
            )
            train_loader = loaders.train
            val_loader = loaders.val
            test_loader = loaders.test
        else:
            train_loader, val_loader = get_video_dataloaders(
                video_dir=video_path,
                batch_size=batch_size,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                num_workers=num_workers,
                pin_memory=pin_memory,
                seed=seed,
            )
            test_loader = None
        if train_loader is None or val_loader is None:
            raise RuntimeError("Could not build dataloaders from manifests")
        if len(train_loader.dataset) != train_image_count:
            raise RuntimeError("Training DataLoader does not match its manifest")
        if len(val_loader.dataset) != val_image_count:
            raise RuntimeError("Validation DataLoader does not match its manifest")
        if (
            test_loader is not None
            and len(test_loader.dataset) != test_image_count
        ):
            raise RuntimeError("Test DataLoader does not match its manifest")
        if evaluate_test and test_loader is None:
            raise RuntimeError("The requested test split is unavailable")

        for variant_name, config in variants_to_run.items():
            dsad_settings = _dsad_settings_for_variant(
                variant_name,
                dsad_beta_max,
                dsad_warmup_fraction,
                dsad_ramp_fraction,
            )
            print(f"\n{'='*55}")
            print(f"Seed: {seed} | Variant: {variant_name}")
            print(
                "Experiment family: "
                f"{_variant_family(variant_name)} | "
                f"DSAD beta max: {dsad_settings['beta_max']}"
            )
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
                        "experiment_family": _variant_family(variant_name),
                        "lambda_rd": lam,
                        "seed": seed,
                        "paired_initialization_seed": seed,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "height": height,
                        "width": width,
                        "architecture": asdict(config),
                        "dsad": dsad_settings,
                        "training_controls": {
                            "learning_rate": learning_rate,
                            "aux_learning_rate": aux_learning_rate,
                            "lr_schedule": lr_schedule,
                            "min_learning_rate": min_learning_rate,
                            "min_aux_learning_rate": min_aux_learning_rate,
                            "checkpoint_selection": checkpoint_selection,
                            "checkpoint_selection_start_epoch": (
                                checkpoint_selection_start_epoch
                                if variant_name in CAUSAL_DSAD_VARIANTS
                                else 1
                            ),
                        },
                        "device": device,
                        "data_protocol": data_protocol,
                        "evaluation_split": "test" if evaluate_test else "val",
                        "frozen_bundle_id": (
                            frozen_bundle.bundle_id
                            if frozen_bundle is not None
                            else None
                        ),
                        "manifest_paths": {
                            "train": train_manifest,
                            "val": val_manifest,
                            "test": test_manifest,
                        },
                    },
                )

                write_json(
                    os.path.join(run_dir, "environment.json"),
                    get_environment_snapshot(device=device, repo_dir=os.getcwd()),
                )

                # Fresh model for every (variant, lambda, seed) combination.
                # Reset immediately before construction so the causal DSAD
                # arms start from exactly the same parameter initialisation
                # and DataLoader worker-seed stream.
                set_global_determinism(seed=seed, deterministic=True)
                loaders_to_reset = [train_loader, val_loader]
                if test_loader is not None:
                    loaders_to_reset.append(test_loader)
                for loader_index, loader in enumerate(loaders_to_reset):
                    generator = getattr(loader, "generator", None)
                    if generator is not None:
                        generator.manual_seed(seed + loader_index)
                model = ATICModel(config, H=height, W=width).to(device)
                initial_state_sha256 = hash_model_state(model)
                write_json(
                    os.path.join(run_dir, "initial_state.json"),
                    {
                        "sha256": initial_state_sha256,
                        "seed": seed,
                        "variant": variant_name,
                    },
                )

                train_artifacts = train_loop(
                    model,
                    variant_name=f"{variant_name}_lam{lam}_seed{seed}",
                    dataloader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    device=device,
                    lambda_rd=lam,
                    dsad_beta_max=dsad_settings["beta_max"],
                    dsad_warmup_fraction=dsad_settings["warmup_fraction"],
                    dsad_ramp_fraction=dsad_settings["ramp_fraction"],
                    learning_rate=learning_rate,
                    aux_learning_rate=aux_learning_rate,
                    lr_schedule=lr_schedule,
                    min_learning_rate=min_learning_rate,
                    min_aux_learning_rate=min_aux_learning_rate,
                    checkpoint_selection=checkpoint_selection,
                    checkpoint_selection_start_epoch=(
                        checkpoint_selection_start_epoch
                        if variant_name in CAUSAL_DSAD_VARIANTS
                        else 1
                    ),
                    checkpoint_path=os.path.join(run_dir, "model.pth"),
                    train_log_path=os.path.join(run_dir, "train_log.jsonl"),
                )
                history = train_artifacts.get("history")
                if not isinstance(history, list) or len(history) != epochs:
                    observed_epochs = (
                        len(history) if isinstance(history, list) else None
                    )
                    raise RuntimeError(
                        "Training did not complete every requested epoch: "
                        f"expected {epochs}, observed {observed_epochs}"
                    )

                validation_point = eval_single(
                    model,
                    val_loader,
                    device=device,
                    bitstream_dir=os.path.join(run_dir, "bitstreams_val"),
                )
                write_json(
                    os.path.join(run_dir, "eval_val_metrics.json"),
                    validation_point,
                )
                if "test" in evaluation_plan:
                    assert test_loader is not None
                    point = eval_single(
                        model,
                        test_loader,
                        device=device,
                        bitstream_dir=os.path.join(
                            run_dir,
                            "bitstreams_test",
                        ),
                    )
                    evaluation_split = "test"
                    write_json(
                        os.path.join(run_dir, "eval_test_metrics.json"),
                        point,
                    )
                else:
                    point = validation_point
                    evaluation_split = "val"
                # Backward-compatible alias for existing result readers.
                write_json(os.path.join(run_dir, "eval_metrics.json"), point)

                bpp_key = round(point.get("BPP", lam), 4)
                points_by_variant_lambda.setdefault((variant_name, lam), []).append(point)

                print(
                    f"  BPP={bpp_key:.4f} | "
                    f"PSNR={_format_optional_metric(point, 'PSNR', 2)} | "
                    f"SSIM={_format_optional_metric(point, 'SSIM', 4)} | "
                    f"LPIPS={_format_optional_metric(point, 'LPIPS', 4)}"
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
                    "lambda_rd": lam,
                    "dsad_beta_max": dsad_settings["beta_max"],
                    "initial_state_sha256": initial_state_sha256,
                    "checkpoint_path": train_artifacts.get("checkpoint_path"),
                    "checkpoint_selection": checkpoint_selection,
                    "selected_epoch": train_artifacts.get("selected_epoch"),
                    "selected_val_rd_loss": train_artifacts.get(
                        "selected_val_rd_loss"
                    ),
                    "data_protocol": data_protocol,
                    "evaluation_split": evaluation_split,
                    "frozen_bundle_id": (
                        frozen_bundle.bundle_id
                        if frozen_bundle is not None
                        else None
                    ),
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

    if frozen_bundle is not None:
        assert dataset_root is not None
        verified_after_run = load_and_verify_frozen_split_bundle(
            split_dir=frozen_bundle.split_dir,
            dataset_root=dataset_root,
            expected_size=(width, height),
        )
        if verified_after_run.bundle_id != frozen_bundle.bundle_id:
            raise RuntimeError("Frozen dataset bundle changed during the study")
        write_json(
            os.path.join(study_dir, "post_run_data_verification.json"),
            {
                "bundle_id": verified_after_run.bundle_id,
                "verified_after_training": True,
                "limitation": (
                    "Exact paths, file bytes, and decoded pixels were checked. "
                    "A separate human/perceptual audit is still required for "
                    "near-duplicate or mislabeled source sequences."
                ),
            },
        )

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
        width=args.width,
        learning_rate=args.learning_rate,
        aux_learning_rate=args.aux_learning_rate,
        lr_schedule=args.lr_schedule,
        min_learning_rate=args.min_learning_rate,
        min_aux_learning_rate=args.min_aux_learning_rate,
        checkpoint_selection=args.checkpoint_selection,
        dsad_beta_max=args.dsad_beta_max,
        dsad_warmup_fraction=args.dsad_warmup_fraction,
        dsad_ramp_fraction=args.dsad_ramp_fraction,
        dataset_root=args.dataset_root,
        frozen_split_dir=args.frozen_split_dir,
        evaluate_test=args.evaluate_test,
    )
