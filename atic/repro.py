"""
repro.py - Reproducibility helpers for ATIC experiments.

Provides deterministic setup, run metadata capture, and JSON utilities so
ablation runs can be reproduced from saved artifacts.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import AbstractSet, Any, Dict, Optional

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
        # Publication runs must fail instead of silently continuing through a
        # nondeterministic CUDA kernel.  Callers can explicitly opt out with
        # deterministic=False for exploratory work.
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


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


def hash_model_state(
    model: torch.nn.Module,
    *,
    exclude_names: Optional[AbstractSet[str]] = None,
) -> str:
    """Hash selected model tensors in stable name/dtype/shape/bytes order."""

    digest = hashlib.sha256()
    excluded = frozenset() if exclude_names is None else frozenset(exclude_names)

    def update_framed(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)

    for name, tensor in sorted(model.state_dict().items()):
        if name in excluded:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        update_framed(name.encode("utf-8"))
        update_framed(str(value.dtype).encode("ascii"))
        update_framed(json.dumps(list(value.shape)).encode("ascii"))
        # ``view(torch.uint8)`` rejects zero-dimensional tensors when the
        # element sizes differ. Flatten first so scalar integer buffers (for
        # example entropy-model counters) have a byte-addressable dimension.
        raw_bytes = (
            value.reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
        update_framed(raw_bytes)
    return digest.hexdigest()


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


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_environment_snapshot(device: str, repo_dir: str) -> Dict[str, Any]:
    """Capture Python, Torch, CUDA, and GPU context for exact reruns."""
    snapshot: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "dependency_versions": {
            distribution: _distribution_version(distribution)
            for distribution in (
                "compressai",
                "numpy",
                "pillow",
                "timm",
                "torchvision",
            )
        },
        "requested_device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "process_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_DEVICE_ORDER",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }

    if torch.cuda.is_available():
        try:
            snapshot["visible_gpu_names"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            snapshot["gpu_name"] = snapshot["visible_gpu_names"][0]
        except Exception:
            snapshot["visible_gpu_names"] = None
            snapshot["gpu_name"] = None
    else:
        snapshot["visible_gpu_names"] = []
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
