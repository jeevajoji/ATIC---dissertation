"""
dataset.py — UVG frame loader
Pads 1080p frames to 1088 (next multiple of 8) for clean Swin window divisions.
"""
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class UVGVideoDataset(Dataset):
    def __init__(self, image_paths: list):
        self.image_paths = image_paths
        self.transform = T.Compose([
            T.ToTensor(),
            # Pad height: 1080 → 1088 (next multiple of 8)
            # Pad only the bottom edge — keeps spatial correspondence intact
            T.Pad((0, 0, 0, 8)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def get_video_dataloaders(
    video_dir: str,
    batch_size: int = 1,
    val_every: int = 10,
) -> tuple:
    all_frames = sorted(glob(os.path.join(video_dir, "*.png")))

    if not all_frames:
        print(f"No .png frames found in {video_dir}")
        return None, None

    train_paths = [f for i, f in enumerate(all_frames) if (i + 1) % val_every != 0]
    val_paths   = [f for i, f in enumerate(all_frames) if (i + 1) % val_every == 0]

    print(f"Frames — train: {len(train_paths)}, val: {len(val_paths)}")

    train_loader = DataLoader(
        UVGVideoDataset(train_paths),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        UVGVideoDataset(val_paths),
        batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader