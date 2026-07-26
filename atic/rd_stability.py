"""Audit selected validation checkpoints and enforce a monotonic RD gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_VARIANT = "Full_ATIC_NoDSAD"


def _csv_floats(value: str) -> List[float]:
    try:
        result = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from exc
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("values must be finite")
    return result


def _csv_ints(value: str) -> List[int]:
    try:
        result = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge validation-only ablation studies, audit selected-versus-"
            "final training histories using the checkpoint-eligibility "
            "window, and require non-decreasing actual BPP and PSNR as "
            "lambda_rd increases."
        )
    )
    parser.add_argument("studies", nargs="+", type=Path)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--expected-lambdas", type=_csv_floats)
    parser.add_argument("--expected-seeds", type=_csv_ints)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output", type=Path)
    return parser


def _read_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing required artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _load_history(run_dir: Path) -> List[Dict[str, object]]:
    path = run_dir / "train_log.jsonl"
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing training history: {path}") from exc
    if not records or any(not isinstance(record, dict) for record in records):
        raise RuntimeError(f"invalid or empty training history: {path}")
    return records


def _history_audit(
    records: Sequence[Dict[str, object]],
    *,
    selected_epoch: int | None = None,
    selection_strategy: str = "best_val_rd",
) -> Dict[str, object]:
    candidates = [
        record
        for record in records
        if record.get("val_rd_loss") is not None
        and math.isfinite(float(record["val_rd_loss"]))
    ]
    if not candidates:
        raise RuntimeError("training history has no finite val_rd_loss")
    best = min(candidates, key=lambda record: float(record["val_rd_loss"]))
    final = candidates[-1]
    eligibility_logged = any(
        "checkpoint_selection_eligible" in record
        for record in candidates
    )
    eligible = [
        record
        for record in candidates
        if record.get("checkpoint_selection_eligible") is True
    ]
    if not eligible and all(
        "checkpoint_selection_eligible" not in record
        for record in candidates
    ):
        # Compatibility with histories created before eligibility was logged.
        eligible = list(candidates)
    best_eligible = (
        min(eligible, key=lambda record: float(record["val_rd_loss"]))
        if eligible
        else None
    )
    if selected_epoch is None:
        selected = best_eligible if best_eligible is not None else best
    else:
        selected_matches = [
            record
            for record in candidates
            if int(record["epoch"]) == int(selected_epoch)
        ]
        if len(selected_matches) != 1:
            raise RuntimeError(
                "reported selected_epoch is absent or duplicated in "
                f"training history: {selected_epoch}"
            )
        selected = selected_matches[0]
        if (
            selection_strategy == "best_val_rd"
            and eligibility_logged
            and best_eligible is None
        ):
            raise RuntimeError(
                "best validation-RD strategy has no eligible checkpoint"
            )
        if selection_strategy == "best_val_rd" and best_eligible is not None:
            if (
                eligibility_logged
                and selected.get("checkpoint_selection_eligible") is not True
            ):
                raise RuntimeError(
                    f"reported selected_epoch is ineligible: {selected_epoch}"
                )
            if int(selected["epoch"]) != int(best_eligible["epoch"]):
                raise RuntimeError(
                    "reported selected_epoch is not the minimum-RD eligible "
                    f"checkpoint: reported {selected_epoch}, expected "
                    f"{int(best_eligible['epoch'])}"
                )
        elif (
            selection_strategy == "final"
            and int(selected["epoch"]) != int(final["epoch"])
        ):
            raise RuntimeError(
                "final checkpoint strategy did not select the final epoch"
            )
        elif selection_strategy not in {"best_val_rd", "final"}:
            raise RuntimeError(
                f"unknown checkpoint selection strategy: {selection_strategy}"
            )

    best_value = float(best["val_rd_loss"])
    selected_value = float(selected["val_rd_loss"])
    final_value = float(final["val_rd_loss"])
    return {
        "best_epoch_any": int(best["epoch"]),
        "best_val_rd_loss_any": best_value,
        "best_epoch_eligible": (
            None if best_eligible is None else int(best_eligible["epoch"])
        ),
        "best_val_rd_loss_eligible": (
            None
            if best_eligible is None
            else float(best_eligible["val_rd_loss"])
        ),
        "selected_epoch": int(selected["epoch"]),
        "selected_val_rd_loss": selected_value,
        "selected_val_total_bpp": (
            None
            if selected.get("val_total_bpp") is None
            else float(selected["val_total_bpp"])
        ),
        "selected_val_mse_loss": (
            None
            if selected.get("val_mse_loss") is None
            else float(selected["val_mse_loss"])
        ),
        "selected_val_psnr_from_mse": (
            None
            if selected.get("val_mse_loss") is None
            or float(selected["val_mse_loss"]) <= 0
            else -10.0 * math.log10(float(selected["val_mse_loss"]))
        ),
        "selected_main_grad_norm": (
            None
            if selected.get("main_grad_norm") is None
            else float(selected["main_grad_norm"])
        ),
        "selected_main_grad_clip_fraction": (
            None
            if selected.get("main_grad_clip_fraction") is None
            else float(selected["main_grad_clip_fraction"])
        ),
        "selected_aux_grad_norm": (
            None
            if selected.get("aux_grad_norm") is None
            else float(selected["aux_grad_norm"])
        ),
        "final_epoch": int(final["epoch"]),
        "final_val_rd_loss": final_value,
        "final_minus_best_val_rd_loss": final_value - best_value,
        "final_over_best_percent": (
            100.0 * (final_value / best_value - 1.0)
            if best_value != 0
            else None
        ),
        "final_minus_selected_val_rd_loss": final_value - selected_value,
        "final_over_selected_percent": (
            100.0 * (final_value / selected_value - 1.0)
            if selected_value != 0
            else None
        ),
        "final_aux_loss": (
            None
            if final.get("aux_loss") is None
            else float(final["aux_loss"])
        ),
    }


def _load_run_provenance(
    *,
    study_config: Dict[str, object],
    run_dir: Path,
    source: Dict[str, object],
    variant: str,
) -> Tuple[Dict[str, object], str]:
    """Load the causal controls needed to compare independently run rates."""

    run_config = _read_json(run_dir / "run_config.json")
    initial_state = _read_json(run_dir / "initial_state.json")
    run_environment = _read_json(run_dir / "environment.json")

    expected_run_fields = {
        "variant": variant,
        "seed": int(source["seed"]),
        "lambda_rd": float(source["lambda_rd"]),
        "evaluation_split": "val",
    }
    for name, expected in expected_run_fields.items():
        observed = run_config.get(name)
        if observed != expected:
            raise RuntimeError(
                f"{run_dir}: run_config {name}={observed!r}, "
                f"expected {expected!r}"
            )

    initial_sha256 = str(initial_state.get("sha256", ""))
    if (
        not initial_sha256
        or initial_sha256 != str(source.get("initial_state_sha256", ""))
        or int(initial_state.get("seed", -1)) != int(source["seed"])
        or initial_state.get("variant") != variant
    ):
        raise RuntimeError(f"{run_dir}: initial-state identity mismatch")
    common_initial_sha256 = str(
        initial_state.get("paired_common_sha256", "")
    )
    source_common_initial_sha256 = str(
        source.get("paired_common_initial_state_sha256", "")
    )
    if common_initial_sha256 != source_common_initial_sha256:
        raise RuntimeError(
            f"{run_dir}: paired-common initial-state identity mismatch"
        )
    common_excluded_names = initial_state.get(
        "paired_common_excluded_names"
    )
    common_excluded_present_names = initial_state.get(
        "paired_common_excluded_present_names"
    )
    if common_initial_sha256:
        if (
            not isinstance(common_excluded_names, list)
            or not all(
                isinstance(name, str) for name in common_excluded_names
            )
            or not isinstance(common_excluded_present_names, list)
            or not all(
                isinstance(name, str)
                for name in common_excluded_present_names
            )
        ):
            raise RuntimeError(
                f"{run_dir}: paired-common exclusion metadata is invalid"
            )

    git = run_environment.get("git")
    if (
        not isinstance(git, dict)
        or not git.get("commit")
        or git.get("is_dirty") is not False
    ):
        raise RuntimeError(
            f"{run_dir}: publication diagnostic requires a clean Git commit"
        )
    dependency_versions = run_environment.get("dependency_versions")
    if (
        not isinstance(dependency_versions, dict)
        or any(
            not dependency_versions.get(name)
            for name in ("compressai", "numpy", "pillow", "timm", "torchvision")
        )
    ):
        raise RuntimeError(
            f"{run_dir}: dependency-version provenance is incomplete"
        )
    if study_config.get("frozen_split_verified") is not True:
        raise RuntimeError(
            f"{run_dir}: rate-response diagnostic requires a frozen split"
        )
    if study_config.get("evaluation_split") != "val":
        raise RuntimeError(
            f"{run_dir}: rate-response diagnostic requires validation only"
        )

    variants = study_config.get("variants")
    variant_config = (
        variants.get(variant) if isinstance(variants, dict) else None
    )
    if not isinstance(variant_config, dict):
        raise RuntimeError(f"{run_dir}: missing study variant configuration")
    if run_config.get("architecture") != variant_config.get("architecture"):
        raise RuntimeError(f"{run_dir}: study/run architecture mismatch")
    if run_config.get("dsad") != variant_config.get("dsad"):
        raise RuntimeError(f"{run_dir}: study/run DSAD configuration mismatch")
    training_controls = run_config.get("training_controls")
    if (
        not isinstance(training_controls, dict)
        or training_controls.get("checkpoint_selection") != "best_val_rd"
        or training_controls.get("checkpoint_selection_start_epoch") != 1
        or source.get("checkpoint_selection")
        != training_controls.get("checkpoint_selection")
    ):
        raise RuntimeError(
            f"{run_dir}: sanity gate requires best validation-RD selection "
            "from epoch 1"
        )
    study_training_controls = study_config.get("training_controls")
    if (
        not isinstance(study_training_controls, dict)
        or any(
            study_training_controls.get(name) != value
            for name, value in training_controls.items()
        )
    ):
        raise RuntimeError(f"{run_dir}: study/run training controls mismatch")
    if (
        run_config.get("epochs") != study_config.get("epochs")
        or run_config.get("batch_size") != study_config.get("batch_size")
        or run_config.get("data_protocol")
        != study_config.get("data_protocol")
    ):
        raise RuntimeError(f"{run_dir}: study/run protocol mismatch")

    manifests = study_config.get("manifests")
    if not isinstance(manifests, dict) or not manifests.get("bundle_id"):
        raise RuntimeError(f"{run_dir}: missing frozen manifest identity")
    if (
        run_config.get("frozen_bundle_id") != manifests.get("bundle_id")
        or source.get("frozen_bundle_id") != manifests.get("bundle_id")
        or source.get("data_protocol") != study_config.get("data_protocol")
    ):
        raise RuntimeError(f"{run_dir}: frozen bundle identity mismatch")

    provenance: Dict[str, object] = {
        "git": {
            "commit": str(git["commit"]),
            "is_dirty": False,
        },
        "software": {
            name: run_environment.get(name)
            for name in (
                "python_version",
                "torch_version",
                "dependency_versions",
                "cuda_version",
                "cudnn_version",
                "deterministic_algorithms_enabled",
                "gpu_name",
            )
        },
        "data": {
            "data_protocol": study_config.get("data_protocol"),
            "frozen_split_verified": True,
            "bundle_id": manifests.get("bundle_id"),
            "dataset_id": manifests.get("dataset_id"),
            "hashes": manifests.get("hashes"),
            "split_counts": study_config.get("split_counts"),
        },
        "run": {
            "variant": variant,
            "architecture": run_config.get("architecture"),
            "initial_state_sha256": initial_sha256,
            "paired_common_initial_state_sha256": (
                common_initial_sha256 or None
            ),
            "paired_common_excluded_names": common_excluded_names,
            "paired_common_excluded_present_names": (
                common_excluded_present_names
            ),
            "seed": int(source["seed"]),
            "epochs": run_config.get("epochs"),
            "batch_size": run_config.get("batch_size"),
            "height": run_config.get("height"),
            "width": run_config.get("width"),
            "training_controls": run_config.get("training_controls"),
            "dsad": run_config.get("dsad"),
        },
    }
    canonical = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return provenance, hashlib.sha256(canonical).hexdigest()


def load_rows(
    study_dirs: Iterable[Path],
    *,
    variant: str,
    require_provenance: bool = False,
) -> List[Dict[str, object]]:
    indexed: Dict[Tuple[int, float], Dict[str, object]] = {}
    for study_dir in study_dirs:
        study_dir = study_dir.resolve()
        config = _read_json(study_dir / "study_config.json")
        if bool(config.get("test_evaluation_enabled")):
            raise RuntimeError(
                f"refusing unlocked test study in validation gate: {study_dir}"
            )
        summary = _read_json(study_dir / "summary_metrics.json")
        rows = summary.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"summary_metrics.json has no rows list: {study_dir}"
            )
        for source in rows:
            if not isinstance(source, dict) or source.get("variant") != variant:
                continue
            if source.get("evaluation_split") != "val":
                raise RuntimeError(
                    f"{study_dir} contains non-validation row for {variant}"
                )
            seed = int(source["seed"])
            lambda_rd = float(source["lambda_rd"])
            key = (seed, lambda_rd)
            if key in indexed:
                raise RuntimeError(
                    f"duplicate result for seed={seed}, lambda={lambda_rd}"
                )
            run_dir = (
                study_dir
                / "runs"
                / variant
                / f"lam_{lambda_rd:g}"
                / f"seed_{seed}"
            )
            checkpoint = source.get("checkpoint_path")
            if checkpoint:
                checkpoint_run_dir = Path(str(checkpoint)).parent
                if checkpoint_run_dir.exists():
                    run_dir = checkpoint_run_dir
            row = dict(source)
            row["study_dir"] = str(study_dir)
            reported_selected_epoch = source.get("selected_epoch")
            history = _history_audit(
                _load_history(run_dir),
                selected_epoch=(
                    None
                    if reported_selected_epoch is None
                    else int(reported_selected_epoch)
                ),
                selection_strategy=str(
                    source.get("checkpoint_selection", "best_val_rd")
                ),
            )
            reported_selected_loss = source.get("selected_val_rd_loss")
            if (
                reported_selected_loss is not None
                and not math.isclose(
                    float(reported_selected_loss),
                    float(history["selected_val_rd_loss"]),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    "summary selected_val_rd_loss does not match training "
                    f"history for seed={seed}, lambda={lambda_rd}"
                )
            row["history"] = history
            if require_provenance:
                provenance, provenance_sha256 = _load_run_provenance(
                    study_config=config,
                    run_dir=run_dir,
                    source=source,
                    variant=variant,
                )
                row["provenance"] = provenance
                row["provenance_sha256"] = provenance_sha256
            indexed[key] = row
    if not indexed:
        raise RuntimeError(f"no rows found for variant {variant!r}")
    return list(indexed.values())


def build_report(
    rows: Sequence[Dict[str, object]],
    *,
    variant: str,
    expected_lambdas: Sequence[float] | None,
    expected_seeds: Sequence[int] | None,
    tolerance: float,
) -> Dict[str, object]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    by_seed: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(dict(row))

    observed_seeds = sorted(by_seed)
    if expected_seeds is not None and observed_seeds != sorted(expected_seeds):
        raise RuntimeError(
            f"expected seeds {sorted(expected_seeds)}, found {observed_seeds}"
        )

    seed_reports = []
    all_passed = True
    for seed, seed_rows in sorted(by_seed.items()):
        seed_rows.sort(key=lambda row: float(row["lambda_rd"]))
        lambdas = [float(row["lambda_rd"]) for row in seed_rows]
        if (
            expected_lambdas is not None
            and lambdas != sorted(float(value) for value in expected_lambdas)
        ):
            raise RuntimeError(
                f"seed {seed}: expected lambdas "
                f"{sorted(expected_lambdas)}, found {lambdas}"
            )
        bpps = [float(row["BPP_actual"]) for row in seed_rows]
        psnrs = [float(row["PSNR"]) for row in seed_rows]
        finite = all(math.isfinite(value) for value in bpps + psnrs)
        bpp_monotonic = finite and all(
            current + tolerance >= previous
            for previous, current in zip(bpps, bpps[1:])
        )
        psnr_monotonic = finite and all(
            current + tolerance >= previous
            for previous, current in zip(psnrs, psnrs[1:])
        )
        passed = len(seed_rows) >= 4 and bpp_monotonic and psnr_monotonic
        all_passed = all_passed and passed
        seed_reports.append(
            {
                "seed": seed,
                "points": seed_rows,
                "bpp_monotonic": bpp_monotonic,
                "psnr_monotonic": psnr_monotonic,
                "point_count": len(seed_rows),
                "passed": passed,
            }
        )

    return {
        "variant": variant,
        "test_locked": True,
        "gate": (
            "At least four validation points per seed; actual BPP and PSNR "
            "must be non-decreasing with lambda_rd."
        ),
        "tolerance": tolerance,
        "seeds": seed_reports,
        "passed": all_passed,
    }


def print_report(report: Dict[str, object]) -> None:
    print(f"Variant: {report['variant']}")
    for seed_report in report["seeds"]:
        print(f"\n=== seed {seed_report['seed']} ===")
        print(
            "lambda    actual BPP/PSNR    selected/final epoch    "
            "final RD above selected"
        )
        for row in seed_report["points"]:
            history = row["history"]
            excess = history["final_over_selected_percent"]
            excess_text = (
                "n/a" if excess is None else f"{float(excess):+.2f}%"
            )
            print(
                f"{float(row['lambda_rd']):<9g} "
                f"{float(row['BPP_actual']):.6f}/"
                f"{float(row['PSNR']):.3f}       "
                f"{int(history['selected_epoch'])}/"
                f"{int(history['final_epoch'])}              "
                f"{excess_text}"
            )
        print(
            "monotonic: "
            f"BPP={seed_report['bpp_monotonic']}, "
            f"PSNR={seed_report['psnr_monotonic']} | "
            f"gate={'PASS' if seed_report['passed'] else 'FAIL'}"
        )
    print("\nBeauty test remained locked.")
    print(f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.studies, variant=args.variant)
    report = build_report(
        rows,
        variant=args.variant,
        expected_lambdas=args.expected_lambdas,
        expected_seeds=args.expected_seeds,
        tolerance=args.tolerance,
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
