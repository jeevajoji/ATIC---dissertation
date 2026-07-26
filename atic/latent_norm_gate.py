"""Evaluate the predeclared encoder-latent-LayerNorm diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Sequence, Tuple

from atic.rate_response import build_report as build_rate_response_report
from atic.rd_stability import load_rows


CONTROL_VARIANT = "Plain_Swin_Hyperprior"
INTERVENTION_VARIANT = "Plain_Swin_Hyperprior_NoLatentNorm"
SEED = 42
LOW_LAMBDA = 0.0018
HIGH_LAMBDA = 0.013
MIN_DELTA_BPP = 0.01
MIN_DELTA_PSNR = 0.25
LATENT_NORM_FIELD = "use_encoder_latent_norm"
LATENT_NORM_STATE_NAMES = [
    "encoder.norm.bias",
    "encoder.norm.weight",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two locked-validation studies for the predeclared "
            "encoder latent-LayerNorm diagnostic."
        )
    )
    parser.add_argument(
        "studies",
        nargs="+",
        type=Path,
        help=(
            "Exactly two study directories, each containing both fixed "
            "rates for one architecture variant."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Required destination for the strict JSON gate report.",
    )
    return parser


def _resolve_studies(studies: Sequence[Path]) -> Tuple[Path, ...]:
    if len(studies) != 2:
        raise RuntimeError(
            f"Expected exactly two study directories, found {len(studies)}"
        )
    resolved = tuple(path.resolve() for path in studies)
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("Duplicate study directories are not permitted")
    return resolved


def _rate_report(
    rows: Sequence[Dict[str, object]],
    *,
    variant: str,
) -> Dict[str, object]:
    return build_rate_response_report(
        rows,
        variant=variant,
        seed=SEED,
        low_lambda=LOW_LAMBDA,
        high_lambda=HIGH_LAMBDA,
        min_delta_bpp=MIN_DELTA_BPP,
        min_delta_psnr=MIN_DELTA_PSNR,
    )


def _normalise_cross_variant_provenance(
    report: Dict[str, object],
    *,
    expected_variant: str,
) -> Tuple[Dict[str, object], Dict[str, object], str, str]:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(
            f"{expected_variant} has no verified provenance"
        )
    run = provenance.get("run")
    if not isinstance(run, dict):
        raise RuntimeError(
            f"{expected_variant} provenance has no run controls"
        )
    if run.get("variant") != expected_variant:
        raise RuntimeError(
            f"{expected_variant} provenance names a different variant"
        )
    architecture = run.get("architecture")
    if not isinstance(architecture, dict):
        raise RuntimeError(
            f"{expected_variant} provenance has no architecture"
        )
    initial_state_sha256 = run.get("initial_state_sha256")
    if not isinstance(initial_state_sha256, str) or not initial_state_sha256:
        raise RuntimeError(
            f"{expected_variant} provenance has no initial-state hash"
        )
    common_initial_state_sha256 = run.get(
        "paired_common_initial_state_sha256"
    )
    if (
        not isinstance(common_initial_state_sha256, str)
        or not common_initial_state_sha256
    ):
        raise RuntimeError(
            f"{expected_variant} provenance has no paired-common "
            "initial-state hash"
        )
    if run.get("paired_common_excluded_names") != LATENT_NORM_STATE_NAMES:
        raise RuntimeError(
            f"{expected_variant} paired-common hash did not exclude exactly "
            "the encoder terminal LayerNorm state"
        )
    expected_present_names = (
        LATENT_NORM_STATE_NAMES
        if expected_variant == CONTROL_VARIANT
        else []
    )
    if (
        run.get("paired_common_excluded_present_names")
        != expected_present_names
    ):
        raise RuntimeError(
            f"{expected_variant} has unexpected encoder terminal "
            "LayerNorm state"
        )

    normalised = copy.deepcopy(provenance)
    normalised_run = normalised["run"]
    for field in (
        "variant",
        "architecture",
        "initial_state_sha256",
        "paired_common_initial_state_sha256",
        "paired_common_excluded_present_names",
    ):
        normalised_run.pop(field)
    return (
        normalised,
        dict(architecture),
        initial_state_sha256,
        common_initial_state_sha256,
    )


def _validate_cross_variant_provenance(
    control_report: Dict[str, object],
    intervention_report: Dict[str, object],
) -> Dict[str, object]:
    (
        control_normalised,
        control_architecture,
        control_initial_state,
        control_common_initial_state,
    ) = _normalise_cross_variant_provenance(
        control_report,
        expected_variant=CONTROL_VARIANT,
    )
    (
        intervention_normalised,
        intervention_architecture,
        intervention_initial_state,
        intervention_common_initial_state,
    ) = _normalise_cross_variant_provenance(
        intervention_report,
        expected_variant=INTERVENTION_VARIANT,
    )

    if control_normalised != intervention_normalised:
        raise RuntimeError(
            "Control/intervention provenance differs outside variant, "
            "architecture, or initial-state hash"
        )

    architecture_fields = (
        set(control_architecture) | set(intervention_architecture)
    )
    differences = {
        field
        for field in architecture_fields
        if control_architecture.get(field)
        != intervention_architecture.get(field)
    }
    if differences != {LATENT_NORM_FIELD}:
        raise RuntimeError(
            "Architectures must differ only in "
            f"{LATENT_NORM_FIELD}; found {sorted(differences)}"
        )
    if (
        control_architecture.get(LATENT_NORM_FIELD) is not True
        or intervention_architecture.get(LATENT_NORM_FIELD) is not False
    ):
        raise RuntimeError(
            f"{LATENT_NORM_FIELD} must change from true to false"
        )
    if control_initial_state == intervention_initial_state:
        raise RuntimeError(
            "Architecture-changing variants unexpectedly share an "
            "initial-state hash"
        )
    if control_common_initial_state != intervention_common_initial_state:
        raise RuntimeError(
            "Control/intervention common initial tensors are not identical"
        )

    return {
        "matched_except": [
            "run.variant",
            "run.architecture",
            "run.initial_state_sha256",
            "run.paired_common_excluded_present_names",
        ],
        "architecture_difference": {
            "field": LATENT_NORM_FIELD,
            "control": True,
            "intervention": False,
        },
        "paired_common_excluded_names": LATENT_NORM_STATE_NAMES,
        "control_initial_state_sha256": control_initial_state,
        "intervention_initial_state_sha256": intervention_initial_state,
        "paired_common_initial_state_sha256": (
            control_common_initial_state
        ),
        "passed": True,
    }


def _actual_rd_points(
    rate_report: Dict[str, object],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for label, expected_lambda in (
        ("low", LOW_LAMBDA),
        ("high", HIGH_LAMBDA),
    ):
        row = rate_report.get(label)
        if not isinstance(row, dict):
            raise RuntimeError(f"{label}-lambda result is missing")
        required = ("lambda_rd", "BPP_actual", "MSE")
        values: Dict[str, float] = {}
        for metric in required:
            if metric not in row:
                raise RuntimeError(
                    f"{label}-lambda row has no {metric}"
                )
            value = float(row[metric])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"{label}-lambda row has non-finite {metric}"
                )
            values[metric] = value
        if values["lambda_rd"] != expected_lambda:
            raise RuntimeError(
                f"{label}-lambda row has lambda_rd={values['lambda_rd']}, "
                f"expected {expected_lambda}"
            )
        if values["BPP_actual"] < 0:
            raise RuntimeError(
                f"{label}-lambda row has negative BPP_actual"
            )
        if values["MSE"] < 0:
            raise RuntimeError(f"{label}-lambda row has negative MSE")

        actual_rd = (
            values["BPP_actual"]
            + values["lambda_rd"] * (255.0**2) * values["MSE"]
        )
        if not math.isfinite(actual_rd):
            raise RuntimeError(
                f"{label}-lambda actual RD became non-finite"
            )
        result[label] = {
            **values,
            "actual_rd": actual_rd,
        }
    return result


def build_report(
    control_rows: Sequence[Dict[str, object]],
    intervention_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    control_rate = _rate_report(
        control_rows,
        variant=CONTROL_VARIANT,
    )
    intervention_rate = _rate_report(
        intervention_rows,
        variant=INTERVENTION_VARIANT,
    )
    provenance_check = _validate_cross_variant_provenance(
        control_rate,
        intervention_rate,
    )

    control_rd = _actual_rd_points(control_rate)
    intervention_rd = _actual_rd_points(intervention_rate)
    rd_checks = {
        f"intervention_actual_rd_lower_{label}": (
            intervention_rd[label]["actual_rd"]
            < control_rd[label]["actual_rd"]
        )
        for label in ("low", "high")
    }
    checks = {
        "intervention_rate_response": bool(intervention_rate["passed"]),
        **rd_checks,
    }

    return {
        "gate": "encoder_latent_norm_two_rate_validation_diagnostic_v1",
        "test_locked": True,
        "evaluation_split": "val",
        "beauty_evaluated": False,
        "seed": SEED,
        "low_lambda": LOW_LAMBDA,
        "high_lambda": HIGH_LAMBDA,
        "minimum_delta_bpp": MIN_DELTA_BPP,
        "minimum_delta_psnr": MIN_DELTA_PSNR,
        "control_variant": CONTROL_VARIANT,
        "intervention_variant": INTERVENTION_VARIANT,
        "cross_variant_provenance": provenance_check,
        "rate_response": {
            "control": control_rate,
            "intervention": intervention_rate,
        },
        "actual_rd": {
            "formula": "BPP_actual + lambda_rd * 255^2 * MSE",
            "control": control_rd,
            "intervention": intervention_rd,
            "intervention_minus_control": {
                label: (
                    intervention_rd[label]["actual_rd"]
                    - control_rd[label]["actual_rd"]
                )
                for label in ("low", "high")
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_study_contributions(
    report: Dict[str, object],
    studies: Sequence[Path],
) -> None:
    rate_response = report["rate_response"]
    sources = {}
    for variant_label in ("control", "intervention"):
        variant_report = rate_response[variant_label]
        variant_sources = set()
        for rate_label in ("low", "high"):
            study_dir = variant_report[rate_label].get("study_dir")
            if not study_dir:
                raise RuntimeError(
                    "A gate row has no source study directory"
                )
            variant_sources.add(Path(str(study_dir)).resolve())
        if len(variant_sources) != 1:
            raise RuntimeError(
                f"{variant_label} low/high rows must share one study"
            )
        sources[variant_label] = next(iter(variant_sources))
    if (
        sources["control"] == sources["intervention"]
        or set(sources.values()) != set(studies)
    ):
        raise RuntimeError(
            "Control and intervention must come from the two distinct "
            "supplied studies"
        )


def print_report(report: Dict[str, object]) -> None:
    for label in ("control", "intervention"):
        rate = report["rate_response"][label]
        rd = report["actual_rd"][label]
        print(f"\n{label}: {rate['variant']}")
        print("lambda    estimated BPP    actual BPP    PSNR      MSE       actual RD")
        for rate_label in ("low", "high"):
            row = rate[rate_label]
            point = rd[rate_label]
            print(
                f"{float(row['lambda_rd']):<9g} "
                f"{float(row['BPP_estimated']):>13.6f} "
                f"{float(row['BPP_actual']):>13.6f} "
                f"{float(row['PSNR']):>8.3f} "
                f"{float(row['MSE']):>9.6f} "
                f"{float(point['actual_rd']):>11.6f}"
            )
        print(
            "rate-response: "
            f"{'PASS' if rate['passed'] else 'FAIL'}"
        )

    print("\nPredeclared decisive checks:")
    for name, passed in report["checks"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print("Beauty test remained locked.")
    print(f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    studies = _resolve_studies(args.studies)
    control_rows = load_rows(
        studies,
        variant=CONTROL_VARIANT,
        require_provenance=True,
    )
    intervention_rows = load_rows(
        studies,
        variant=INTERVENTION_VARIANT,
        require_provenance=True,
    )
    report = build_report(control_rows, intervention_rows)
    _validate_study_contributions(report, studies)
    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
