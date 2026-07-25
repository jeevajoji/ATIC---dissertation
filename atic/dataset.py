"""
dataset.py — UVG frame loader
Pads 1080p frames to 1088 (next multiple of 8) for clean Swin window divisions.
"""
import hashlib
import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from PIL import Image


FROZEN_SPLIT_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class FrozenSplit:
    name: str
    sequences: Tuple[str, ...]
    relative_paths: Tuple[str, ...]
    image_paths: Tuple[str, ...]
    manifest_path: str
    manifest_sha256: str
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FrozenSplitBundle:
    dataset_id: str
    dataset_root: str
    split_dir: str
    bundle_id: str
    image_width: int
    image_height: int
    splits: Dict[str, FrozenSplit]


@dataclass(frozen=True)
class SplitLoaders:
    train: object
    val: object
    test: object


class UVGVideoDataset:
    def __init__(self, image_paths: list):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from torchvision.transforms.functional import to_tensor

        with Image.open(self.image_paths[idx]) as source:
            image = source.convert("RGB")
            return to_tensor(image)


def _read_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_manifest(manifest_path: str, image_paths: List[str]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(f"{path}\n")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_sequence_names(
    sequences: Sequence[str],
    split_name: str,
) -> Tuple[str, ...]:
    values = tuple(sorted(str(value).strip() for value in sequences))
    if not values or any(not value for value in values):
        raise ValueError(f"{split_name} must contain at least one sequence")
    if len(values) != len(set(values)):
        raise ValueError(f"{split_name} contains duplicate sequence names")
    for value in values:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
            raise ValueError(
                f"Sequence names must be single relative directory names: {value!r}"
            )
    return values


def _resolve_dataset_path(dataset_root: str, relative_path: str) -> str:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"Manifest entries must be dataset-root-relative: {relative_path!r}"
        )

    root = os.path.realpath(dataset_root)
    resolved = os.path.realpath(os.path.join(root, *relative.parts))
    try:
        inside_root = os.path.commonpath([root, resolved]) == root
    except ValueError:
        inside_root = False
    if not inside_root:
        raise ValueError(f"Manifest entry escapes dataset root: {relative_path!r}")
    return resolved


def _inspect_images(
    dataset_root: str,
    relative_paths: Sequence[str],
    *,
    expected_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[int, int], str, str, Dict[str, str]]:
    file_aggregate = hashlib.sha256()
    pixel_aggregate = hashlib.sha256()
    pixel_hashes: Dict[str, str] = {}
    observed_size: Optional[Tuple[int, int]] = expected_size

    for relative_path in relative_paths:
        absolute_path = _resolve_dataset_path(dataset_root, relative_path)
        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(
                f"Frozen manifest image does not exist: {absolute_path}"
            )
        if Path(relative_path).suffix.lower() != ".png":
            raise ValueError(
                f"Frozen manifests currently support PNG images only: {relative_path}"
            )

        with Image.open(absolute_path) as source:
            image = source.convert("RGB")
            size = tuple(int(value) for value in image.size)
            pixel_digest = hashlib.sha256()
            pixel_digest.update(f"{size[0]}x{size[1]}:RGB\0".encode("ascii"))
            pixel_digest.update(image.tobytes())
            image_hash = pixel_digest.hexdigest()
        if observed_size is None:
            observed_size = size
        elif size != observed_size:
            raise ValueError(
                "All frozen-split images must have one coded resolution; "
                f"expected {observed_size[0]}x{observed_size[1]}, "
                f"found {size[0]}x{size[1]} at {relative_path}"
            )

        file_hash = _sha256_file(absolute_path)
        pixel_hashes[relative_path] = image_hash
        for aggregate, digest in (
            (file_aggregate, file_hash),
            (pixel_aggregate, image_hash),
        ):
            aggregate.update(relative_path.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(bytes.fromhex(digest))
            aggregate.update(b"\0")

    if observed_size is None:
        raise ValueError("A frozen split cannot be empty")
    return (
        observed_size,
        file_aggregate.hexdigest(),
        pixel_aggregate.hexdigest(),
        pixel_hashes,
    )


def _manifest_payload(relative_paths: Sequence[str]) -> bytes:
    return "".join(f"{path}\n" for path in relative_paths).encode("utf-8")


def _bundle_identity_payload(metadata: Dict[str, object]) -> bytes:
    identity = {
        key: value
        for key, value in metadata.items()
        if key != "bundle_id"
    }
    return json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def create_frozen_sequence_split_bundle(
    dataset_root: str,
    output_dir: str,
    dataset_id: str,
    train_sequences: Sequence[str],
    val_sequences: Sequence[str],
    test_sequences: Sequence[str],
) -> FrozenSplitBundle:
    """Create an immutable, sequence-disjoint split bundle.

    ``dataset_root`` must contain one directory per named sequence. Every PNG
    below a sequence directory belongs to that sequence. Manifests store
    sorted, dataset-root-relative POSIX paths so the bundle can be moved to a
    different machine with the same dataset layout.
    """

    dataset_root = os.path.realpath(dataset_root)
    output_dir = os.path.abspath(output_dir)
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id must not be empty")
    if not os.path.isdir(dataset_root):
        raise NotADirectoryError(f"Dataset root does not exist: {dataset_root}")
    if os.path.exists(output_dir):
        raise FileExistsError(
            f"Refusing to overwrite frozen split bundle: {output_dir}"
        )

    split_sequences = {
        "train": _normalise_sequence_names(train_sequences, "train"),
        "val": _normalise_sequence_names(val_sequences, "val"),
        "test": _normalise_sequence_names(test_sequences, "test"),
    }
    sequence_owner: Dict[str, str] = {}
    for split_name, sequences in split_sequences.items():
        for sequence in sequences:
            previous = sequence_owner.setdefault(sequence, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Sequence {sequence!r} appears in both {previous} and "
                    f"{split_name}"
                )

    split_paths: Dict[str, Tuple[str, ...]] = {}
    path_owner: Dict[str, str] = {}
    for split_name, sequences in split_sequences.items():
        paths: List[str] = []
        for sequence in sequences:
            sequence_dir = os.path.join(dataset_root, sequence)
            if not os.path.isdir(sequence_dir):
                raise NotADirectoryError(
                    f"Sequence directory does not exist: {sequence_dir}"
                )
            for absolute_path in glob(
                os.path.join(sequence_dir, "**", "*.png"),
                recursive=True,
            ):
                relative = Path(
                    os.path.relpath(absolute_path, dataset_root)
                ).as_posix()
                paths.append(relative)
        relative_paths = tuple(sorted(paths))
        if not relative_paths:
            raise ValueError(f"{split_name} split contains no PNG images")
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError(f"{split_name} split contains duplicate paths")
        for relative_path in relative_paths:
            previous = path_owner.setdefault(relative_path, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Image path appears in both {previous} and {split_name}: "
                    f"{relative_path}"
                )
        split_paths[split_name] = relative_paths

    inspected: Dict[str, Dict[str, object]] = {}
    common_size: Optional[Tuple[int, int]] = None
    content_owner: Dict[str, Tuple[str, str]] = {}
    for split_name in SPLIT_NAMES:
        size, file_sha256, content_sha256, hashes = _inspect_images(
            dataset_root,
            split_paths[split_name],
            expected_size=common_size,
        )
        common_size = size
        for relative_path, file_hash in hashes.items():
            previous = content_owner.setdefault(
                file_hash,
                (split_name, relative_path),
            )
            if previous[0] != split_name:
                raise ValueError(
                    "Exact duplicate image content crosses frozen splits: "
                    f"{previous[0]}/{previous[1]} and "
                    f"{split_name}/{relative_path}"
                )
        payload = _manifest_payload(split_paths[split_name])
        inspected[split_name] = {
            "manifest_file": f"{split_name}_manifest.txt",
            "manifest_sha256": _sha256_bytes(payload),
            "file_sha256": file_sha256,
            "content_sha256": content_sha256,
            "image_count": len(split_paths[split_name]),
            "sequences": list(split_sequences[split_name]),
            "_payload": payload,
        }

    assert common_size is not None
    metadata: Dict[str, object] = {
        "schema_version": FROZEN_SPLIT_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "image_width": common_size[0],
        "image_height": common_size[1],
        "splits": {
            split_name: {
                key: value
                for key, value in inspected[split_name].items()
                if not key.startswith("_")
            }
            for split_name in SPLIT_NAMES
        },
    }
    metadata["bundle_id"] = _sha256_bytes(_bundle_identity_payload(metadata))

    os.makedirs(output_dir)
    for split_name in SPLIT_NAMES:
        manifest_path = os.path.join(
            output_dir,
            str(inspected[split_name]["manifest_file"]),
        )
        with open(manifest_path, "wb") as handle:
            handle.write(inspected[split_name]["_payload"])
    with open(
        os.path.join(output_dir, "split_metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    loaded_splits = {
        split_name: FrozenSplit(
            name=split_name,
            sequences=split_sequences[split_name],
            relative_paths=split_paths[split_name],
            image_paths=tuple(
                _resolve_dataset_path(dataset_root, path)
                for path in split_paths[split_name]
            ),
            manifest_path=os.path.join(
                output_dir,
                str(inspected[split_name]["manifest_file"]),
            ),
            manifest_sha256=str(
                inspected[split_name]["manifest_sha256"]
            ),
            file_sha256=str(inspected[split_name]["file_sha256"]),
            content_sha256=str(inspected[split_name]["content_sha256"]),
        )
        for split_name in SPLIT_NAMES
    }
    return FrozenSplitBundle(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        split_dir=output_dir,
        bundle_id=str(metadata["bundle_id"]),
        image_width=common_size[0],
        image_height=common_size[1],
        splits=loaded_splits,
    )


def load_and_verify_frozen_split_bundle(
    split_dir: str,
    dataset_root: str,
    expected_size: Optional[Tuple[int, int]] = None,
) -> FrozenSplitBundle:
    """Load a bundle and fail on leakage, tampering, or changed image bytes."""

    split_dir = os.path.realpath(split_dir)
    dataset_root = os.path.realpath(dataset_root)
    metadata_path = os.path.join(split_dir, "split_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != FROZEN_SPLIT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported frozen split schema version: "
            f"{metadata.get('schema_version')!r}"
        )
    stored_bundle_id = metadata.get("bundle_id")
    calculated_bundle_id = _sha256_bytes(_bundle_identity_payload(metadata))
    if stored_bundle_id != calculated_bundle_id:
        raise ValueError("Frozen split metadata or bundle ID was modified")

    metadata_size = (
        int(metadata.get("image_width", 0)),
        int(metadata.get("image_height", 0)),
    )
    if metadata_size[0] <= 0 or metadata_size[1] <= 0:
        raise ValueError("Frozen split metadata contains an invalid image size")
    if expected_size is not None and tuple(expected_size) != metadata_size:
        raise ValueError(
            f"Frozen split is {metadata_size[0]}x{metadata_size[1]}, "
            f"but the experiment expects {expected_size[0]}x{expected_size[1]}"
        )

    split_metadata = metadata.get("splits")
    if not isinstance(split_metadata, dict):
        raise ValueError("Frozen split metadata is missing split definitions")

    sequence_owner: Dict[str, str] = {}
    path_owner: Dict[str, str] = {}
    content_owner: Dict[str, Tuple[str, str]] = {}
    loaded: Dict[str, FrozenSplit] = {}

    for split_name in SPLIT_NAMES:
        details = split_metadata.get(split_name)
        if not isinstance(details, dict):
            raise ValueError(f"Frozen bundle is missing the {split_name} split")
        sequences = _normalise_sequence_names(
            details.get("sequences", []),
            split_name,
        )
        for sequence in sequences:
            previous = sequence_owner.setdefault(sequence, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Sequence {sequence!r} crosses {previous}/{split_name}"
                )

        manifest_file = details.get("manifest_file")
        if (
            not isinstance(manifest_file, str)
            or os.path.basename(manifest_file) != manifest_file
        ):
            raise ValueError(f"Invalid {split_name} manifest filename")
        manifest_path = os.path.join(split_dir, manifest_file)
        with open(manifest_path, "rb") as handle:
            payload = handle.read()
        if _sha256_bytes(payload) != details.get("manifest_sha256"):
            raise ValueError(f"{split_name} manifest hash does not match metadata")

        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{split_name} manifest is not UTF-8") from exc
        relative_paths = tuple(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )
        if not relative_paths:
            raise ValueError(f"{split_name} manifest is empty")
        if tuple(sorted(relative_paths)) != relative_paths:
            raise ValueError(f"{split_name} manifest must be sorted")
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError(f"{split_name} manifest contains duplicate paths")
        if len(relative_paths) != int(details.get("image_count", -1)):
            raise ValueError(f"{split_name} manifest count does not match metadata")

        for relative_path in relative_paths:
            parts = Path(relative_path).parts
            first_part = parts[0] if parts else ""
            if first_part not in sequences:
                raise ValueError(
                    f"{split_name} path is outside its declared sequences: "
                    f"{relative_path}"
                )
            previous = path_owner.setdefault(relative_path, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Image path crosses {previous}/{split_name}: {relative_path}"
                )

        size, file_sha256, content_sha256, hashes = _inspect_images(
            dataset_root,
            relative_paths,
            expected_size=metadata_size,
        )
        if size != metadata_size:
            raise ValueError(f"{split_name} image resolution changed")
        if file_sha256 != details.get("file_sha256"):
            raise ValueError(
                f"{split_name} image files no longer match the frozen bundle"
            )
        if content_sha256 != details.get("content_sha256"):
            raise ValueError(
                f"{split_name} image content no longer matches the frozen bundle"
            )
        for relative_path, file_hash in hashes.items():
            previous = content_owner.setdefault(
                file_hash,
                (split_name, relative_path),
            )
            if previous[0] != split_name:
                raise ValueError(
                    "Exact duplicate image content crosses frozen splits: "
                    f"{previous[0]}/{previous[1]} and "
                    f"{split_name}/{relative_path}"
                )

        image_paths = tuple(
            _resolve_dataset_path(dataset_root, path)
            for path in relative_paths
        )
        loaded[split_name] = FrozenSplit(
            name=split_name,
            sequences=sequences,
            relative_paths=relative_paths,
            image_paths=image_paths,
            manifest_path=manifest_path,
            manifest_sha256=str(details["manifest_sha256"]),
            file_sha256=str(details["file_sha256"]),
            content_sha256=str(details["content_sha256"]),
        )

    return FrozenSplitBundle(
        dataset_id=str(metadata["dataset_id"]),
        dataset_root=dataset_root,
        split_dir=split_dir,
        bundle_id=str(stored_bundle_id),
        image_width=metadata_size[0],
        image_height=metadata_size[1],
        splits=loaded,
    )


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
    from torch.utils.data import DataLoader

    from atic.repro import make_torch_generator, seed_worker

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


def _make_loader(
    image_paths: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    shuffle: bool,
) -> object:
    from torch.utils.data import DataLoader

    from atic.repro import make_torch_generator, seed_worker

    return DataLoader(
        UVGVideoDataset(list(image_paths)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=make_torch_generator(seed),
        worker_init_fn=seed_worker,
    )


def get_frozen_split_dataloaders(
    bundle: FrozenSplitBundle,
    batch_size: int = 1,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
) -> SplitLoaders:
    """Build stable train/validation/test loaders from a verified bundle."""

    return SplitLoaders(
        train=_make_loader(
            bundle.splits["train"].image_paths,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            shuffle=True,
        ),
        val=_make_loader(
            bundle.splits["val"].image_paths,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + 1,
            shuffle=False,
        ),
        test=_make_loader(
            bundle.splits["test"].image_paths,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + 2,
            shuffle=False,
        ),
    )
