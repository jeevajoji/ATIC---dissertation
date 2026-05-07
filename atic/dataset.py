"""
dataset.py — UVG frame loader
Pads 1080p frames to 1088 (next multiple of 8) for clean Swin window divisions.
"""
import os
from glob import glob
from typing import List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from atic.repro import make_torch_generator, seed_worker


class UVGVideoDataset(Dataset):
    def __init__(self, image_paths: list):
        self.image_paths = image_paths
        self.transform = T.Compose([
            T.ToTensor(),
            # Padding removed; assuming images are already CenterCropped to 512x512
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def _read_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_manifest(manifest_path: str, image_paths: List[str]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(f"{path}\n")


def build_and_save_split_manifests(
    video_dir: str,
    manifest_dir: str,
    val_every: int = 10,
) -> Tuple[Optional[str], Optional[str]]:
    """Create deterministic train/val manifests from sorted frame paths."""
    all_frames = sorted(glob(os.path.join(video_dir, "*.png")))
    if not all_frames:
        return None, None

    train_paths = [f for i, f in enumerate(all_frames) if (i + 1) % val_every != 0]
    val_paths = [f for i, f in enumerate(all_frames) if (i + 1) % val_every == 0]

    train_manifest = os.path.join(manifest_dir, "train_manifest.txt")
    val_manifest = os.path.join(manifest_dir, "val_manifest.txt")

    _write_manifest(train_manifest, train_paths)
    _write_manifest(val_manifest, val_paths)
    return train_manifest, val_manifest


def get_video_dataloaders(
    video_dir: str,
    batch_size: int = 1,
    val_every: int = 10,
    train_manifest: Optional[str] = None,
    val_manifest: Optional[str] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: Optional[int] = None,
) -> tuple:
    if train_manifest and val_manifest:
        train_paths = _read_manifest(train_manifest)
        val_paths = _read_manifest(val_manifest)
    else:
        all_frames = sorted(glob(os.path.join(video_dir, "*.png")))

        if not all_frames:
            print(f"No .png frames found in {video_dir}")
            return None, None

        train_paths = [f for i, f in enumerate(all_frames) if (i + 1) % val_every != 0]
        val_paths = [f for i, f in enumerate(all_frames) if (i + 1) % val_every == 0]

    print(f"Frames — train: {len(train_paths)}, val: {len(val_paths)}")

    generator = make_torch_generator(seed) if seed is not None else None
    worker_init = seed_worker if seed is not None else None

    train_loader = DataLoader(
        UVGVideoDataset(train_paths),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        UVGVideoDataset(val_paths),
        batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init,
    )

    return train_loader, val_loader