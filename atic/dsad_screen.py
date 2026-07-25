"""Two-GPU, validation-only DSAD beta calibration.

This runner intentionally keeps the independent Beauty test split locked.  It
launches one isolated ``ablation.py`` process per arm, records exact commands
and GPU mappings, then merges the four validation results into one report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


EXPECTED_BUNDLE_ID = (
    "9af91138ebe5b31b75526b64a9f579fb85edac6e445008a5537486d2d652e5c6"
)
CONTROL_VARIANT = "Full_ATIC_NoDSAD"
DSAD_VARIANT = "Full_ATIC_DSAD"
_PRINT_LOCK = threading.Lock()
_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()


@dataclass(frozen=True)
class ScreenJob:
    beta: float

    @property
    def slug(self) -> str:
        beta_text = format(self.beta, ".17g").replace(".", "p")
        return f"paired_beta_{beta_text}"


def _csv_floats(value: str) -> List[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values or any(
        not math.isfinite(item) or item <= 0
        for item in values
    ):
        raise argparse.ArgumentTypeError(
            "DSAD betas must be finite and positive"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("DSAD betas must be unique")
    return values


def _csv_gpus(value: str) -> List[str]:
    raw_values = [item.strip() for item in value.split(",") if item.strip()]
    if len(raw_values) != 2 or any(
        not item.isdigit()
        for item in raw_values
    ):
        raise argparse.ArgumentTypeError(
            "exactly two non-negative integer GPU IDs are required"
        )
    values = [str(int(item)) for item in raw_values]
    if len(set(values)) != 2:
        raise argparse.ArgumentTypeError("exactly two distinct GPU IDs are required")
    return values


def build_parser() -> argparse.ArgumentParser:
    home = Path.home()
    data_base = home / "datasets" / "UVG_DSAD_screen_v1"
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked validation-only DSAD beta calibration on two GPUs."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=str(data_base / "dataset_512"),
    )
    parser.add_argument(
        "--frozen-split-dir",
        default=str(data_base / "frozen_split"),
    )
    parser.add_argument("--physical-gpus", type=_csv_gpus, default=["0", "1"])
    parser.add_argument(
        "--betas",
        type=_csv_floats,
        default=[0.5, 2.0, 8.0],
        help="Positive DSAD beta values (default: 0.5,2,8).",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-rd", type=float, default=0.0067)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--output-root",
        default="ablation_results/dsad_beta_calibration",
    )
    parser.add_argument("--required-branch", default="final")
    return parser


def calibration_jobs(betas: Sequence[float]) -> List[ScreenJob]:
    return [ScreenJob(float(beta)) for beta in betas]


def assign_jobs(
    jobs: Sequence[ScreenJob],
    physical_gpus: Sequence[str],
) -> Dict[str, List[ScreenJob]]:
    if len(physical_gpus) != 2:
        raise ValueError("exactly two GPUs are required")
    assignments = {gpu: [] for gpu in physical_gpus}
    for index, job in enumerate(jobs):
        assignments[physical_gpus[index % len(physical_gpus)]].append(job)
    return assignments


def build_ablation_command(
    *,
    python_executable: str,
    repo_root: Path,
    args: argparse.Namespace,
    job: ScreenJob,
    job_output_root: Path,
) -> List[str]:
    return [
        python_executable,
        str(repo_root / "ablation.py"),
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--frozen-split-dir",
        str(Path(args.frozen_split_dir).resolve()),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cuda:0",
        "--variants",
        f"{CONTROL_VARIANT},{DSAD_VARIANT}",
        "--seeds",
        str(args.seed),
        "--lambdas",
        format(args.lambda_rd, "g"),
        "--output-root",
        str(job_output_root),
        "--study-name",
        job.slug,
        "--num-workers",
        str(args.num_workers),
        "--pin-memory",
        "true",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--dsad-beta-max",
        format(job.beta, ".17g"),
        "--dsad-warmup-fraction",
        "0.20",
        "--dsad-ramp-fraction",
        "0.10",
    ]


def child_environment(
    base: Mapping[str, str],
    *,
    physical_gpu: str,
    seed: int,
) -> Dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": physical_gpu,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(seed),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment


def _print(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _git_value(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_repository(repo_root: Path, required_branch: str) -> Dict[str, object]:
    branch = _git_value(repo_root, "branch", "--show-current")
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    remote_commit = _git_value(repo_root, "rev-parse", f"origin/{required_branch}")
    dirty = _git_value(repo_root, "status", "--porcelain")
    if branch != required_branch:
        raise RuntimeError(
            f"screen requires branch {required_branch!r}, found {branch!r}"
        )
    if dirty:
        raise RuntimeError(
            "repository has uncommitted changes; refusing an unauditable run"
        )
    if commit != remote_commit:
        raise RuntimeError(
            f"local {required_branch} does not match origin/{required_branch}; "
            "fetch and pull before running"
        )
    return {
        "branch": branch,
        "commit": commit,
        "remote_commit": remote_commit,
        "is_dirty": False,
    }


def verify_bundle(args: argparse.Namespace) -> Dict[str, object]:
    from atic.dataset import load_and_verify_frozen_split_bundle

    bundle = load_and_verify_frozen_split_bundle(
        split_dir=args.frozen_split_dir,
        dataset_root=args.dataset_root,
        expected_size=(args.width, args.height),
    )
    if bundle.bundle_id != EXPECTED_BUNDLE_ID:
        raise RuntimeError(
            "wrong frozen bundle: "
            f"expected {EXPECTED_BUNDLE_ID}, found {bundle.bundle_id}"
        )
    expected_counts = {"train": 360, "val": 60, "test": 120}
    observed_counts = {
        name: len(bundle.splits[name].image_paths)
        for name in expected_counts
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"wrong frozen split counts: {observed_counts}"
        )
    return {
        "bundle_id": bundle.bundle_id,
        "dataset_id": bundle.dataset_id,
        "counts": observed_counts,
        "test_locked": True,
    }


def preflight_gpu(
    python_executable: str,
    *,
    physical_gpu: str,
    seed: int,
) -> Dict[str, object]:
    code = (
        "import json, torch\n"
        "assert torch.cuda.is_available(), 'CUDA unavailable'\n"
        "assert torch.cuda.device_count() == 1, "
        "'child must see exactly one GPU'\n"
        "x=torch.randn(256,256,device='cuda:0'); "
        "y=x@x; torch.cuda.synchronize()\n"
        "free,total=torch.cuda.mem_get_info(0)\n"
        "print(json.dumps({'name':torch.cuda.get_device_name(0),"
        "'free_bytes':free,'total_bytes':total,"
        "'mean':float(y.mean())}))\n"
    )
    result = subprocess.run(
        [python_executable, "-c", code],
        env=child_environment(
            os.environ,
            physical_gpu=physical_gpu,
            seed=seed,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"GPU {physical_gpu} preflight failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"GPU {physical_gpu} preflight returned invalid output"
        ) from exc
    payload["physical_gpu"] = physical_gpu
    return payload


def _stream_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name == "posix",
        )
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.add(process)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                _print(f"[{label}] {line.rstrip()}")
            return_code = process.wait()
        except BaseException:
            _terminate_process(process)
            raise
        finally:
            with _PROCESS_LOCK:
                _ACTIVE_PROCESSES.discard(process)
    if return_code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {return_code}; see {log_path}"
        )


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()


def terminate_active_processes() -> None:
    with _PROCESS_LOCK:
        active = list(_ACTIVE_PROCESSES)
    for process in active:
        _terminate_process(process)


def _single_path(paths: Iterable[Path], description: str) -> Path:
    found = list(paths)
    if len(found) != 1:
        raise RuntimeError(
            f"expected one {description}, found {len(found)}"
        )
    return found[0]


def _last_training_record(
    study_dir: Path,
    variant: str,
) -> Dict[str, object]:
    log_path = _single_path(
        study_dir.glob(f"runs/{variant}/**/train_log.jsonl"),
        f"training log for {variant}",
    )
    lines = [
        line.strip()
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"empty training log: {log_path}")
    return json.loads(lines[-1])


def load_job_rows(
    *,
    job: ScreenJob,
    job_output_root: Path,
    expected_bundle_id: str,
    seed: int,
    lambda_rd: float,
    physical_gpu: str,
) -> List[Dict[str, object]]:
    study_dir = _single_path(
        job_output_root.glob(f"{job.slug}_*"),
        f"study directory for {job.slug}",
    )
    summary_path = study_dir / "summary_metrics.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError(f"{summary_path} must contain exactly two rows")
    by_variant = {row.get("variant"): dict(row) for row in rows}
    if set(by_variant) != {CONTROL_VARIANT, DSAD_VARIANT}:
        raise RuntimeError(f"{summary_path} lacks the controlled DSAD pair")

    result: List[Dict[str, object]] = []
    for variant in (CONTROL_VARIANT, DSAD_VARIANT):
        source = by_variant[variant]
        expected = {
            "seed": seed,
            "evaluation_split": "val",
            "frozen_bundle_id": expected_bundle_id,
        }
        for key, value in expected.items():
            if source.get(key) != value:
                raise RuntimeError(
                    f"{job.slug}/{variant} has wrong {key}: "
                    f"{source.get(key)!r}"
                )
        if abs(float(source.get("lambda_rd")) - lambda_rd) > 1e-12:
            raise RuntimeError(f"{job.slug}/{variant} has wrong lambda")
        expected_beta = 0.0 if variant == CONTROL_VARIANT else job.beta
        if abs(float(source.get("dsad_beta_max")) - expected_beta) > 1e-12:
            raise RuntimeError(f"{job.slug}/{variant} has wrong beta")
        required_metrics = (
            "BPP_actual",
            "MSE",
            "PSNR",
            "SSIM",
            "MS-SSIM",
            "LPIPS",
            "DISTS",
        )
        for metric in required_metrics:
            value = source.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(
                    f"{job.slug}/{variant} has invalid {metric}: {value!r}"
                )
        if not source.get("initial_state_sha256"):
            raise RuntimeError(
                f"{job.slug}/{variant} lacks initial-state proof"
            )

        row: Dict[str, object] = {
            "arm": "control" if variant == CONTROL_VARIANT else "dsad",
            "pair_beta": job.beta,
            "beta": expected_beta,
            "variant": variant,
            "seed": seed,
            "lambda_rd": lambda_rd,
            "physical_gpu": physical_gpu,
            "study_dir": str(study_dir),
        }
        row.update(source)
        row["rd_objective_actual"] = (
            float(source["BPP_actual"])
            + lambda_rd * (255.0 ** 2) * float(source["MSE"])
        )

        final_train = _last_training_record(study_dir, variant)
        for key in (
            "val_dsad_loss",
            "val_dsad_mae",
            "val_dsad_cosine",
            "val_student_gain_min",
            "val_student_gain_max",
            "val_student_gain_std",
            "val_teacher_spatial_std",
        ):
            if key in final_train:
                if not math.isfinite(float(final_train[key])):
                    raise RuntimeError(
                        f"{job.slug}/{variant} has invalid final {key}"
                    )
                row[f"final_{key}"] = final_train[key]
        result.append(row)
    if (
        result[0]["initial_state_sha256"]
        != result[1]["initial_state_sha256"]
    ):
        raise RuntimeError(
            f"{job.slug} controlled arms did not share initial weights"
        )
    return result


def build_report(
    rows: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    repository: Mapping[str, object],
    bundle: Mapping[str, object],
) -> Dict[str, object]:
    delta_rules = {
        "BPP_actual": "lower",
        "PSNR": "higher",
        "SSIM": "higher",
        "MS-SSIM": "higher",
        "LPIPS": "lower",
        "DISTS": "lower",
        "MSE": "lower",
        "rd_objective_actual": "lower",
        "final_val_dsad_loss": "lower",
        "final_val_dsad_mae": "lower",
        "final_val_dsad_cosine": "higher",
    }
    report_rows: List[Dict[str, object]] = []
    pair_summaries: List[Dict[str, object]] = []
    pair_betas = sorted({float(row["pair_beta"]) for row in rows})
    for pair_beta in pair_betas:
        pair = [
            dict(row)
            for row in rows
            if float(row["pair_beta"]) == pair_beta
        ]
        controls = [row for row in pair if row["variant"] == CONTROL_VARIANT]
        dsad_rows = [row for row in pair if row["variant"] == DSAD_VARIANT]
        if len(controls) != 1 or len(dsad_rows) != 1:
            raise RuntimeError(
                f"beta {pair_beta:g} requires one same-job controlled pair"
            )
        control = controls[0]
        dsad = dsad_rows[0]
        for row in (control, dsad):
            for metric, direction in delta_rules.items():
                if metric in row and metric in control:
                    delta = float(row[metric]) - float(control[metric])
                    row[f"delta_{metric}"] = delta
                    row[f"{metric}_direction"] = (
                        "control"
                        if row["variant"] == CONTROL_VARIANT
                        else (
                            "better"
                            if (
                                delta < 0
                                if direction == "lower"
                                else delta > 0
                            )
                            else "worse_or_equal"
                        )
                    )
            gain_min = row.get("final_val_student_gain_min")
            gain_max = row.get("final_val_student_gain_max")
            row["gain_saturation_warning"] = bool(
                gain_min is not None
                and gain_max is not None
                and (
                    float(gain_min) <= 0.505
                    or float(gain_max) >= 1.98
                )
            )
            report_rows.append(row)
        pair_summaries.append(dsad)

    best = min(
        pair_summaries,
        key=lambda item: float(item["delta_rd_objective_actual"]),
    )
    control_rd_values = [
        float(row["rd_objective_actual"])
        for row in report_rows
        if row["variant"] == CONTROL_VARIANT
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "screen_type": "validation_only_beta_calibration",
        "repository": dict(repository),
        "bundle": dict(bundle),
        "test_evaluated": False,
        "test_locked": True,
        "rows": report_rows,
        "pair_summaries": pair_summaries,
        "lowest_validation_rd_delta_beta": best["pair_beta"],
        "control_rd_range": max(control_rd_values) - min(control_rd_values),
        "decision": (
            "Calibration only. Inspect alignment, gain saturation, actual "
            "rate-distortion, and perceptual metrics before selecting beta. "
            "Then confirm one frozen beta with paired seeds 42,123,999."
        ),
    }


def write_report(run_dir: Path, report: Mapping[str, object]) -> None:
    json_path = run_dir / "calibration_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = list(report["rows"])
    fieldnames = sorted({key for row in rows for key in row})
    with (run_dir / "calibration_report.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "batch_size", "num_workers", "height", "width"):
        value = getattr(args, name)
        minimum = 0 if name == "num_workers" else 1
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if not math.isfinite(args.lambda_rd) or args.lambda_rd <= 0:
        raise ValueError("lambda_rd must be positive")
    if any(not math.isfinite(beta) or beta <= 0 for beta in args.betas):
        raise ValueError("betas must be finite and positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested = ",".join(args.physical_gpus)
    if visible is not None and visible.replace(" ", "") != requested:
        raise RuntimeError(
            "parent CUDA_VISIBLE_DEVICES does not match --physical-gpus: "
            f"{visible!r} versus {requested!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    repo_root = Path(__file__).resolve().parents[1]
    repository = verify_repository(repo_root, args.required_branch)
    bundle = verify_bundle(args)

    preflights = [
        preflight_gpu(
            sys.executable,
            physical_gpu=gpu,
            seed=args.seed,
        )
        for gpu in args.physical_gpus
    ]
    for item in preflights:
        _print(
            f"GPU {item['physical_gpu']}: {item['name']} | "
            f"free {float(item['free_bytes']) / 2**30:.2f} GiB"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (repo_root / args.output_root / f"screen_{run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    jobs = calibration_jobs(args.betas)
    assignments = assign_jobs(jobs, args.physical_gpus)
    config = {
        "run_id": run_id,
        "repository": repository,
        "bundle": bundle,
        "arguments": vars(args),
        "jobs": [asdict(job) for job in jobs],
        "gpu_assignments": {
            gpu: [job.slug for job in assigned]
            for gpu, assigned in assignments.items()
        },
        "gpu_preflights": preflights,
        "test_evaluated": False,
    }
    (run_dir / "calibration_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def run_gpu_queue(gpu: str, assigned: Sequence[ScreenJob]) -> None:
        for job in assigned:
            job_root = run_dir / "studies" / job.slug
            command = build_ablation_command(
                python_executable=sys.executable,
                repo_root=repo_root,
                args=args,
                job=job,
                job_output_root=job_root,
            )
            command_path = run_dir / "commands" / f"{job.slug}.json"
            command_path.parent.mkdir(parents=True, exist_ok=True)
            command_path.write_text(
                json.dumps(
                    {
                        "physical_gpu": gpu,
                        "argv": command,
                        "environment": child_environment(
                            {},
                            physical_gpu=gpu,
                            seed=args.seed,
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            _print(f"START {job.slug} on physical GPU {gpu}")
            _stream_process(
                command,
                cwd=repo_root,
                environment=child_environment(
                    os.environ,
                    physical_gpu=gpu,
                    seed=args.seed,
                ),
                log_path=run_dir / "logs" / f"{job.slug}.log",
                label=f"GPU{gpu}:{job.slug}",
            )
            _print(f"DONE  {job.slug} on physical GPU {gpu}")

    try:
        executor = ThreadPoolExecutor(max_workers=2)
        futures = {}
        try:
            futures = {
                executor.submit(run_gpu_queue, gpu, assigned): gpu
                for gpu, assigned in assignments.items()
            }
            for future in as_completed(futures):
                future.result()
        except BaseException:
            terminate_active_processes()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        gpu_by_job = {
            job.slug: gpu
            for gpu, assigned in assignments.items()
            for job in assigned
        }
        rows = [
            row
            for job in jobs
            for row in load_job_rows(
                job=job,
                job_output_root=run_dir / "studies" / job.slug,
                expected_bundle_id=EXPECTED_BUNDLE_ID,
                seed=args.seed,
                lambda_rd=args.lambda_rd,
                physical_gpu=gpu_by_job[job.slug],
            )
        ]
        report = build_report(
            rows,
            run_id=run_id,
            repository=repository,
            bundle=bundle,
        )
        write_report(run_dir, report)
        (run_dir / "COMPLETE").write_text("ok\n", encoding="utf-8")
    except Exception as exc:
        terminate_active_processes()
        (run_dir / "FAILED").write_text(f"{exc}\n", encoding="utf-8")
        raise

    _print("")
    _print("VALIDATION-ONLY CALIBRATION COMPLETE")
    for row in report["pair_summaries"]:
        delta_psnr = row.get("delta_PSNR")
        _print(
            f"beta={float(row['pair_beta']):g} | "
            f"delta BPP={float(row['delta_BPP_actual']):+.6f} | "
            "delta PSNR="
            f"{'unavailable' if delta_psnr is None else f'{float(delta_psnr):+.4f}'}"
            " | "
            f"delta RD={float(row['delta_rd_objective_actual']):+.6f}"
        )
    _print(f"Report: {run_dir / 'calibration_report.json'}")
    _print("Beauty test remained locked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
