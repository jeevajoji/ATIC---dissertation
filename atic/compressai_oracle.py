"""Validation-only pretrained CompressAI oracle for the frozen UVG screen.

This module answers a narrow diagnostic question: can an official pretrained
learned codec produce an ordered rate-distortion curve on the exact validation
data used by the ATIC screening protocol? It never evaluates the locked test
split and reports real entropy payload bytes from ``compress()``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atic.dataset import (
    FrozenSplitBundle,
    UVGVideoDataset,
    load_and_verify_frozen_split_bundle,
)
from atic.repro import (
    get_environment_snapshot,
    hash_model_state,
    make_torch_generator,
    seed_worker,
    set_global_determinism,
)
from atic.uvg_prep import PROTOCOL_NAME, TEST_SEQUENCES, VAL_SEQUENCES


MODEL_NAME = "mbt2018_mean"
MODEL_ARCHITECTURE = "MeanScaleHyperprior"
MODEL_METRIC = "mse"
VALID_QUALITIES = tuple(range(1, 9))
DEFAULT_QUALITIES = (1, 8)
DEFAULT_SEED = 42
DEFAULT_MIN_DELTA_BPP = 0.01
DEFAULT_MIN_DELTA_PSNR = 0.25
PSNR_MSE_FLOOR = 1e-12
EXPECTED_SOFTWARE = {
    "torch_version": "2.6.0+cu124",
    "cuda_version": "12.4",
    "dependency_versions": {
        "compressai": "1.2.8",
        "numpy": "1.26.4",
        "pillow": "12.2.0",
        "timm": "0.6.13",
        "torchvision": "0.21.0+cu124",
    },
}

# This is the content-bound identity printed by ``atic.uvg_prep`` for the
# dissertation's frozen UVG screening bundle. Keeping it as the CLI default
# prevents an accidentally regenerated or different split from being treated
# as the declared experiment.
EXPECTED_BUNDLE_ID = (
    "9af91138ebe5b31b75526b64a9f579fb85edac6e445008a5537486d2d652e5c6"
)


def _parse_qualities(raw: str) -> List[int]:
    try:
        qualities = [
            int(value.strip())
            for value in raw.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "qualities must be comma-separated integers"
        ) from exc
    if len(qualities) < 2:
        raise argparse.ArgumentTypeError(
            "at least two qualities are required for a monotonicity oracle"
        )
    if len(qualities) != len(set(qualities)):
        raise argparse.ArgumentTypeError("qualities must be unique")
    invalid = [
        quality for quality in qualities if quality not in VALID_QUALITIES
    ]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"{MODEL_NAME} qualities must be in {VALID_QUALITIES}; "
            f"received {invalid}"
        )
    return sorted(qualities)


def _entropy_stream_byte_sizes(strings: object) -> List[int]:
    """Return one byte total per top-level entropy stream."""

    if not isinstance(strings, (list, tuple)) or not strings:
        raise TypeError("codec output 'strings' must be a non-empty sequence")

    def payload_bytes(value: object) -> int:
        if isinstance(value, (bytes, bytearray, memoryview)):
            if len(value) == 0:
                raise ValueError("entropy byte strings must be non-empty")
            return len(value)
        if isinstance(value, (list, tuple)) and value:
            return sum(payload_bytes(item) for item in value)
        raise TypeError(
            "entropy strings must contain only non-empty nested sequences "
            "of byte strings"
        )

    return [payload_bytes(stream) for stream in strings]


def _load_pretrained_model(quality: int):
    """Load the official CompressAI MSE checkpoint for one quality."""

    from compressai.zoo import mbt2018_mean

    return mbt2018_mean(
        quality=quality,
        metric=MODEL_METRIC,
        pretrained=True,
        progress=True,
    )


def _validate_bundle(
    bundle: FrozenSplitBundle,
    *,
    expected_bundle_id: Optional[str],
) -> None:
    if bundle.dataset_id != PROTOCOL_NAME:
        raise ValueError(
            f"oracle requires dataset_id={PROTOCOL_NAME!r}; "
            f"received {bundle.dataset_id!r}"
        )
    if (bundle.image_width, bundle.image_height) != (512, 512):
        raise ValueError(
            "oracle requires the declared 512x512 frozen validation crops"
        )
    if expected_bundle_id is not None and bundle.bundle_id != expected_bundle_id:
        raise ValueError(
            "frozen bundle identity does not match the declared screening "
            f"bundle: {bundle.bundle_id}"
        )

    val_split = bundle.splits.get("val")
    test_split = bundle.splits.get("test")
    if val_split is None or test_split is None:
        raise ValueError("frozen bundle must retain separate val and test splits")
    if val_split.sequences != tuple(sorted(VAL_SEQUENCES)):
        raise ValueError(
            "validation split is not the declared ShakeNDry sequence group"
        )
    if test_split.sequences != tuple(sorted(TEST_SEQUENCES)):
        raise ValueError("locked test split is not the declared Beauty group")

    val_paths = set(val_split.image_paths)
    test_paths = set(test_split.image_paths)
    if val_paths & test_paths:
        raise ValueError("validation and locked-test image paths overlap")


def _make_validation_loader(
    bundle: FrozenSplitBundle,
    *,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    """Construct only the validation loader; no test DataLoader is created."""

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    return DataLoader(
        UVGVideoDataset(list(bundle.splits["val"].image_paths)),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=make_torch_generator(seed),
        worker_init_fn=seed_worker,
    )


@torch.no_grad()
def evaluate_quality(
    model,
    dataloader: Iterable[torch.Tensor],
    *,
    quality: int,
    device: str,
) -> Dict[str, object]:
    """Evaluate one pretrained quality using decoded entropy streams."""

    model = model.to(device)
    model.eval()
    if hasattr(model, "update"):
        model.update(force=False)

    total_payload_bytes = 0
    total_pixels = 0
    total_squared_error = 0.0
    total_values = 0
    total_psnr = 0.0
    total_images = 0
    perfect_reconstructions = 0
    stream_byte_totals: Optional[List[int]] = None

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if (
            not isinstance(batch, torch.Tensor)
            or batch.ndim != 4
            or batch.size(0) != 1
            or batch.size(1) != 3
        ):
            raise ValueError(
                "oracle validation loader must yield one RGB image at a time"
            )
        batch = batch.to(device, non_blocking=True)
        if not bool(torch.isfinite(batch).all()):
            raise ValueError("validation input contains NaN or infinity")

        encoded = model.compress(batch)
        if not isinstance(encoded, dict):
            raise TypeError("CompressAI compress() must return a dictionary")
        if "strings" not in encoded or "shape" not in encoded:
            raise KeyError("CompressAI output must contain strings and shape")
        stream_sizes = _entropy_stream_byte_sizes(encoded["strings"])
        if stream_byte_totals is None:
            stream_byte_totals = [0] * len(stream_sizes)
        if len(stream_sizes) != len(stream_byte_totals):
            raise RuntimeError("entropy stream count changed during evaluation")
        for index, num_bytes in enumerate(stream_sizes):
            stream_byte_totals[index] += num_bytes

        decoded = model.decompress(
            strings=encoded["strings"],
            shape=encoded["shape"],
        )
        if not isinstance(decoded, dict) or "x_hat" not in decoded:
            raise TypeError(
                "CompressAI decompress() must return a dictionary with x_hat"
            )
        x_hat = decoded["x_hat"]
        if x_hat.shape != batch.shape:
            raise ValueError(
                "decoded image shape changed: "
                f"{tuple(x_hat.shape)} versus {tuple(batch.shape)}"
            )
        if not bool(torch.isfinite(x_hat).all()):
            raise ValueError("decoded image contains NaN or infinity")
        x_hat = x_hat.clamp(0.0, 1.0)

        image_mse = float(F.mse_loss(x_hat, batch).item())
        if image_mse == 0.0:
            perfect_reconstructions += 1
        total_psnr += -10.0 * math.log10(max(image_mse, PSNR_MSE_FLOOR))
        total_squared_error += float(
            F.mse_loss(x_hat, batch, reduction="sum").item()
        )
        total_values += int(batch.numel())
        total_pixels += int(batch.size(2) * batch.size(3))
        total_payload_bytes += sum(stream_sizes)
        total_images += 1

    if total_images == 0 or total_pixels == 0 or total_values == 0:
        raise RuntimeError("validation split produced no images")
    assert stream_byte_totals is not None

    aggregate_mse = total_squared_error / total_values
    return {
        "quality": quality,
        "model_class": type(model).__name__,
        "BPP": float(8.0 * total_payload_bytes / total_pixels),
        "PSNR": float(total_psnr / total_images),
        "MSE": float(aggregate_mse),
        "aggregate_PSNR": float(
            -10.0 * math.log10(max(aggregate_mse, PSNR_MSE_FLOOR))
        ),
        "payload_bytes": total_payload_bytes,
        "stream_bytes": stream_byte_totals,
        "num_images": total_images,
        "num_pixels": total_pixels,
        "perfect_reconstruction_count": perfect_reconstructions,
        "psnr_mse_floor": PSNR_MSE_FLOOR,
    }


def _build_monotonicity_report(
    results: Sequence[Dict[str, object]],
    *,
    tolerance: float = 1e-10,
    min_delta_bpp: float = DEFAULT_MIN_DELTA_BPP,
    min_delta_psnr: float = DEFAULT_MIN_DELTA_PSNR,
) -> Dict[str, object]:
    if len(results) < 2:
        raise ValueError("monotonicity requires at least two quality points")
    if (
        not math.isfinite(tolerance)
        or tolerance < 0
        or not math.isfinite(min_delta_bpp)
        or min_delta_bpp < 0
        or not math.isfinite(min_delta_psnr)
        or min_delta_psnr < 0
    ):
        raise ValueError(
            "oracle tolerances and margins must be finite and non-negative"
        )
    ordered = sorted(results, key=lambda point: int(point["quality"]))
    bpps = [float(point["BPP"]) for point in ordered]
    psnrs = [float(point["PSNR"]) for point in ordered]
    finite = all(math.isfinite(value) for value in bpps + psnrs)
    delta_bpps = [
        current - previous
        for previous, current in zip(bpps, bpps[1:])
    ]
    delta_psnrs = [
        current - previous
        for previous, current in zip(psnrs, psnrs[1:])
    ]
    bpp_monotonic = finite and all(
        current + tolerance >= previous
        for previous, current in zip(bpps, bpps[1:])
    )
    psnr_monotonic = finite and all(
        current + tolerance >= previous
        for previous, current in zip(psnrs, psnrs[1:])
    )
    bpp_margin = finite and all(
        delta + tolerance >= min_delta_bpp for delta in delta_bpps
    )
    psnr_margin = finite and all(
        delta + tolerance >= min_delta_psnr for delta in delta_psnrs
    )
    return {
        "quality_order": [int(point["quality"]) for point in ordered],
        "BPP_non_decreasing": bpp_monotonic,
        "PSNR_non_decreasing": psnr_monotonic,
        "BPP_margin_passed": bpp_margin,
        "PSNR_margin_passed": psnr_margin,
        "adjacent_delta_BPP": delta_bpps,
        "adjacent_delta_PSNR": delta_psnrs,
        "minimum_delta_BPP": min_delta_bpp,
        "minimum_delta_PSNR": min_delta_psnr,
        "passed": (
            bpp_monotonic
            and psnr_monotonic
            and bpp_margin
            and psnr_margin
        ),
        "tolerance": tolerance,
    }


def run_oracle(
    *,
    dataset_root: str,
    frozen_split_dir: str,
    qualities: Sequence[int],
    output_path: str,
    device: str = "cuda",
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = DEFAULT_SEED,
    expected_bundle_id: Optional[str] = EXPECTED_BUNDLE_ID,
    min_delta_bpp: float = DEFAULT_MIN_DELTA_BPP,
    min_delta_psnr: float = DEFAULT_MIN_DELTA_PSNR,
    repo_dir: Optional[str] = None,
    expected_software: Optional[Dict[str, object]] = EXPECTED_SOFTWARE,
    model_factory: Optional[Callable[[int], object]] = None,
    bundle_loader: Callable[..., FrozenSplitBundle] = (
        load_and_verify_frozen_split_bundle
    ),
    environment_loader: Callable[..., Dict[str, object]] = (
        get_environment_snapshot
    ),
) -> Dict[str, object]:
    """Run and persist the validation-only oracle."""

    quality_values = sorted(int(value) for value in qualities)
    if len(quality_values) < 2 or len(quality_values) != len(set(quality_values)):
        raise ValueError("qualities must contain at least two unique values")
    if any(value not in VALID_QUALITIES for value in quality_values):
        raise ValueError(f"qualities must be in {VALID_QUALITIES}")
    if (
        not math.isfinite(min_delta_bpp)
        or min_delta_bpp < 0
        or not math.isfinite(min_delta_psnr)
        or min_delta_psnr < 0
    ):
        raise ValueError(
            "oracle pass margins must be finite and non-negative"
        )
    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}) but unavailable")

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    set_global_determinism(seed=seed, deterministic=True)
    repository = os.path.realpath(
        repo_dir
        if repo_dir is not None
        else str(Path(__file__).resolve().parents[1])
    )
    environment = environment_loader(device=device, repo_dir=repository)
    git = environment.get("git")
    if (
        not isinstance(git, dict)
        or not git.get("commit")
        or git.get("is_dirty") is not False
    ):
        raise RuntimeError(
            "CompressAI oracle requires a clean, recorded Git commit"
        )
    if environment.get("deterministic_algorithms_enabled") is not True:
        raise RuntimeError(
            "CompressAI oracle requires deterministic algorithms"
        )
    if expected_software is not None:
        observed_dependencies = environment.get("dependency_versions")
        expected_dependencies = expected_software.get("dependency_versions")
        if (
            environment.get("torch_version")
            != expected_software.get("torch_version")
            or environment.get("cuda_version")
            != expected_software.get("cuda_version")
            or not isinstance(observed_dependencies, dict)
            or not isinstance(expected_dependencies, dict)
            or any(
                observed_dependencies.get(name) != expected
                for name, expected in expected_dependencies.items()
            )
        ):
            raise RuntimeError(
                "CompressAI oracle software does not match the pinned "
                f"environment: observed={environment!r}"
            )

    bundle = bundle_loader(
        split_dir=frozen_split_dir,
        dataset_root=dataset_root,
        expected_size=(512, 512),
    )
    _validate_bundle(bundle, expected_bundle_id=expected_bundle_id)
    factory = model_factory or _load_pretrained_model
    results = []
    for quality in quality_values:
        # Recreate the loader so worker state cannot leak between qualities.
        quality_loader = _make_validation_loader(
            bundle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        model = factory(quality)
        model_state_sha256 = hash_model_state(model)
        point = evaluate_quality(
            model,
            quality_loader,
            quality=quality,
            device=device,
        )
        point["model_state_sha256"] = model_state_sha256
        results.append(point)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "oracle": {
            "model": MODEL_NAME,
            "architecture": MODEL_ARCHITECTURE,
            "metric": MODEL_METRIC,
            "pretrained": True,
            "qualities": quality_values,
            "rate_definition": (
                "sum of real CompressAI entropy-string payload bytes times "
                "eight, divided by source-image pixels"
            ),
            "quality_definition": (
                "mean per-image RGB PSNR after CompressAI decompress()"
            ),
            "expected_bundle_id": expected_bundle_id,
            "minimum_delta_BPP": min_delta_bpp,
            "minimum_delta_PSNR": min_delta_psnr,
            "expected_software": expected_software,
            "canonical_protocol": (
                quality_values == list(DEFAULT_QUALITIES)
                and expected_bundle_id == EXPECTED_BUNDLE_ID
                and min_delta_bpp == DEFAULT_MIN_DELTA_BPP
                and min_delta_psnr == DEFAULT_MIN_DELTA_PSNR
                and expected_software == EXPECTED_SOFTWARE
            ),
        },
        "protocol": {
            "dataset_id": bundle.dataset_id,
            "bundle_id": bundle.bundle_id,
            "dataset_root": os.path.realpath(dataset_root),
            "frozen_split_dir": os.path.realpath(frozen_split_dir),
            "evaluation_split": "val",
            "validation_sequences": list(bundle.splits["val"].sequences),
            "validation_image_count": len(
                bundle.splits["val"].image_paths
            ),
            "validation_manifest_sha256": (
                bundle.splits["val"].manifest_sha256
            ),
            "validation_file_sha256": bundle.splits["val"].file_sha256,
            "validation_content_sha256": (
                bundle.splits["val"].content_sha256
            ),
            "test_locked": True,
            "test_evaluated": False,
            "locked_test_sequences": list(
                bundle.splits["test"].sequences
            ),
            "seed": seed,
            "device": device,
        },
        "environment": environment,
        "results": results,
        "monotonicity": _build_monotonicity_report(
            results,
            min_delta_bpp=min_delta_bpp,
            min_delta_psnr=min_delta_psnr,
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate official pretrained CompressAI mbt2018_mean qualities "
            "on the frozen ShakeNDry validation split using real payload bytes. "
            "Beauty remains locked."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frozen-split-dir", required=True)
    parser.add_argument(
        "--qualities",
        type=_parse_qualities,
        default=list(DEFAULT_QUALITIES),
        help="Comma-separated official qualities in 1..8 (default: 1,8).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        choices=("true", "false"),
        default="true",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-delta-bpp",
        type=float,
        default=DEFAULT_MIN_DELTA_BPP,
    )
    parser.add_argument(
        "--min-delta-psnr",
        type=float,
        default=DEFAULT_MIN_DELTA_PSNR,
    )
    parser.add_argument(
        "--expected-bundle-id",
        default=EXPECTED_BUNDLE_ID,
        help="Content-bound frozen bundle identity required by the run.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_oracle(
        dataset_root=args.dataset_root,
        frozen_split_dir=args.frozen_split_dir,
        qualities=args.qualities,
        output_path=args.output,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory == "true",
        seed=args.seed,
        expected_bundle_id=args.expected_bundle_id,
        min_delta_bpp=args.min_delta_bpp,
        min_delta_psnr=args.min_delta_psnr,
    )

    print(
        f"Oracle: {MODEL_NAME} ({MODEL_ARCHITECTURE}), "
        f"split=val, bundle={report['protocol']['bundle_id']}"
    )
    print("quality    payload BPP    PSNR")
    for point in report["results"]:
        print(
            f"{int(point['quality']):<10d} "
            f"{float(point['BPP']):>11.6f} "
            f"{float(point['PSNR']):>8.3f}"
        )
    gate = report["monotonicity"]
    print(
        "oracle gate: "
        f"BPP={gate['BPP_non_decreasing']}, "
        f"PSNR={gate['PSNR_non_decreasing']}, "
        f"BPP margin={gate['BPP_margin_passed']}, "
        f"PSNR margin={gate['PSNR_margin_passed']} | "
        f"gate={'PASS' if gate['passed'] else 'FAIL'}"
    )
    print("Beauty test remained locked.")
    print(f"Saved: {Path(args.output).resolve()}")
    return 0 if bool(gate["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
