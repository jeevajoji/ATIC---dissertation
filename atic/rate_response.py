"""Evaluate a predeclared two-rate trainer sanity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Sequence

from atic.rd_stability import load_rows


DEFAULT_VARIANT = "Plain_Swin_Hyperprior"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two locked-validation studies and test whether the "
            "higher lambda produces meaningfully higher rate and quality."
        )
    )
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-lambda", type=float, default=0.0018)
    parser.add_argument("--high-lambda", type=float, default=0.013)
    parser.add_argument("--min-delta-bpp", type=float, default=0.01)
    parser.add_argument("--min-delta-psnr", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    return parser


def build_report(
    rows: Sequence[Dict[str, object]],
    *,
    variant: str,
    seed: int,
    low_lambda: float,
    high_lambda: float,
    min_delta_bpp: float,
    min_delta_psnr: float,
) -> Dict[str, object]:
    values = {
        "low_lambda": low_lambda,
        "high_lambda": high_lambda,
        "min_delta_bpp": min_delta_bpp,
        "min_delta_psnr": min_delta_psnr,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("lambda values and pass margins must be finite")
    if low_lambda < 0 or high_lambda <= low_lambda:
        raise ValueError("Require 0 <= low_lambda < high_lambda")
    if min_delta_bpp < 0 or min_delta_psnr < 0:
        raise ValueError("Pass margins must be non-negative")

    matching_rows = [
        dict(row)
        for row in rows
        if row.get("variant") == variant
    ]
    keys = [
        (int(row["seed"]), float(row["lambda_rd"]))
        for row in matching_rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate seed/lambda rows are not permitted")
    indexed = dict(zip(keys, matching_rows))
    expected_keys = {(seed, low_lambda), (seed, high_lambda)}
    if set(indexed) != expected_keys:
        raise RuntimeError(
            f"Expected exactly {sorted(expected_keys)}, found {sorted(indexed)}"
        )

    low = indexed[(seed, low_lambda)]
    high = indexed[(seed, high_lambda)]
    provenance_values = []
    for label, row in (("low", low), ("high", high)):
        provenance = row.get("provenance")
        provenance_sha256 = row.get("provenance_sha256")
        if not isinstance(provenance, dict) or not provenance_sha256:
            raise RuntimeError(
                f"{label}-lambda row has no verified provenance"
            )
        computed_sha256 = hashlib.sha256(
            json.dumps(
                provenance,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if str(provenance_sha256) != computed_sha256:
            raise RuntimeError(
                f"{label}-lambda provenance digest is invalid"
            )
        provenance_values.append(
            (provenance, str(provenance_sha256))
        )
    if provenance_values[0] != provenance_values[1]:
        raise RuntimeError(
            "Low/high studies have mismatched commits, data, architecture, "
            "initialisation, or training controls"
        )

    required_metrics = ("BPP_actual", "BPP_estimated", "PSNR")
    for label, row in (("low", low), ("high", high)):
        for metric in required_metrics:
            if metric not in row or not math.isfinite(float(row[metric])):
                raise RuntimeError(
                    f"{label}-lambda row has no finite {metric}"
                )

    delta_actual_bpp = float(high["BPP_actual"]) - float(low["BPP_actual"])
    delta_estimated_bpp = (
        float(high["BPP_estimated"]) - float(low["BPP_estimated"])
    )
    delta_psnr = float(high["PSNR"]) - float(low["PSNR"])
    checks = {
        "actual_bpp_margin": delta_actual_bpp >= min_delta_bpp,
        "estimated_bpp_order": delta_estimated_bpp > 0,
        "psnr_margin": delta_psnr >= min_delta_psnr,
    }

    return {
        "variant": variant,
        "seed": seed,
        "test_locked": True,
        "low_lambda": low_lambda,
        "high_lambda": high_lambda,
        "minimum_delta_bpp": min_delta_bpp,
        "minimum_delta_psnr": min_delta_psnr,
        "provenance": provenance_values[0][0],
        "provenance_sha256": provenance_values[0][1],
        "low": low,
        "high": high,
        "delta_BPP_actual": delta_actual_bpp,
        "delta_BPP_estimated": delta_estimated_bpp,
        "delta_PSNR": delta_psnr,
        "checks": checks,
        "passed": all(checks.values()),
    }


def print_report(report: Dict[str, object]) -> None:
    low = report["low"]
    high = report["high"]
    print(f"Variant: {report['variant']} | seed: {report['seed']}")
    print("lambda    estimated BPP    actual BPP    PSNR")
    for row in (low, high):
        print(
            f"{float(row['lambda_rd']):<9g} "
            f"{float(row['BPP_estimated']):>13.6f} "
            f"{float(row['BPP_actual']):>13.6f} "
            f"{float(row['PSNR']):>8.3f}"
        )
    print(
        "\nDeltas (high - low): "
        f"estimated BPP={float(report['delta_BPP_estimated']):+.6f}, "
        f"actual BPP={float(report['delta_BPP_actual']):+.6f}, "
        f"PSNR={float(report['delta_PSNR']):+.3f} dB"
    )
    for name, passed in report["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("Beauty test remained locked.")
    print(f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(
        args.studies,
        variant=args.variant,
        require_provenance=True,
    )
    report = build_report(
        rows,
        variant=args.variant,
        seed=args.seed,
        low_lambda=args.low_lambda,
        high_lambda=args.high_lambda,
        min_delta_bpp=args.min_delta_bpp,
        min_delta_psnr=args.min_delta_psnr,
    )
    print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
