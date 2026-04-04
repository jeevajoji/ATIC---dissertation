"""
repro.py - Reproducibility helpers for ATIC experiments.

Provides deterministic setup, run metadata capture, and JSON utilities so
ablation runs can be reproduced from saved artifacts.
"""
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency in this project
    np = None


def utc_timestamp() -> str:
    """Return a stable UTC timestamp used in run folder names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def set_global_determinism(seed: int, deterministic: bool = True) -> None:
    """Set random seeds and deterministic backend flags for repeatable runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from the process-level torch seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded torch generator for DataLoader reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _run_git_command(repo_dir: str, args: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None


def get_git_snapshot(repo_dir: str) -> Dict[str, Any]:
    """Capture git commit and dirty state for experiment auditability."""
    commit = _run_git_command(repo_dir, ["rev-parse", "HEAD"])
    status = _run_git_command(repo_dir, ["status", "--porcelain"])

    return {
        "commit": commit,
        "is_dirty": bool(status) if status is not None else None,
    }


def get_environment_snapshot(device: str, repo_dir: str) -> Dict[str, Any]:
    """Capture Python, Torch, CUDA, and GPU context for exact reruns."""
    snapshot: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if torch.cuda.is_available():
        try:
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            snapshot["gpu_name"] = None
    else:
        snapshot["gpu_name"] = None

    snapshot["git"] = get_git_snapshot(repo_dir)
    return snapshot


def to_serializable(value: Any) -> Any:
    """Convert dataclasses and tensors into JSON-serializable structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_json(file_path: str, payload: Dict[str, Any]) -> None:
    """Write pretty JSON with deterministic key ordering."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, sort_keys=True)
