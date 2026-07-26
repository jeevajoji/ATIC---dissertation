"""Audit validation RD histories and enforce a monotonic control-curve gate."""

from __future__ import annotations

import argparse
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
            "Merge validation-only ablation studies, audit best-versus-final "
            "training histories, and require non-decreasing actual BPP and "
            "PSNR as lambda_rd increases."
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
    best_value = float(best["val_rd_loss"])
    final_value = float(final["val_rd_loss"])
    return {
        "best_epoch_any": int(best["epoch"]),
        "best_val_rd_loss_any": best_value,
        "final_epoch": int(final["epoch"]),
        "final_val_rd_loss": final_value,
        "final_minus_best_val_rd_loss": final_value - best_value,
        "final_over_best_percent": (
            100.0 * (final_value / best_value - 1.0)
            if best_value != 0
            else None
        ),
        "final_aux_loss": (
            None
            if final.get("aux_loss") is None
            else float(final["aux_loss"])
        ),
    }


def load_rows(
    study_dirs: Iterable[Path],
    *,
    variant: str,
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
            row["history"] = _history_audit(_load_history(run_dir))
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
            "lambda    actual BPP/PSNR    best/final epoch    "
            "final RD above best"
        )
        for row in seed_report["points"]:
            history = row["history"]
            excess = history["final_over_best_percent"]
            excess_text = (
                "n/a" if excess is None else f"{float(excess):+.2f}%"
            )
            print(
                f"{float(row['lambda_rd']):<9g} "
                f"{float(row['BPP_actual']):.6f}/"
                f"{float(row['PSNR']):.3f}       "
                f"{int(history['best_epoch_any'])}/"
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
