"""Prepare the frozen UVG dataset used for causal-DSAD screening.

This module deliberately implements one fixed protocol rather than exposing
sampling choices that could be tuned after looking at validation or test
results:

* train: Bosphorus, HoneyBee, YachtRide
* validation: ShakeNDry
* locked test: Beauty
* source frames: 12 + 25k (zero based)
* crops: five non-overlapping 512x512 regions from each 1920x1080 frame

Beauty and ShakeNDry are downloaded from the official UVG site.  Existing
official Bosphorus, HoneyBee and YachtRide sources are read from sibling
directories under ``--source-root``.  No test metric is computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from atic.dataset import (
    FrozenSplitBundle,
    create_frozen_sequence_split_bundle,
    load_and_verify_frozen_split_bundle,
)


PROTOCOL_NAME = "uvg_dsad_screen_v1"
PREPARATION_SCHEMA_VERSION = 1
MINIMUM_FREE_BYTES = 6 * 1024**3
SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080
CROP_SIZE = 512
TRAIN_SEQUENCES = ("Bosphorus", "HoneyBee", "YachtRide")
VAL_SEQUENCES = ("ShakeNDry",)
TEST_SEQUENCES = ("Beauty",)
ALL_SEQUENCES = (*TRAIN_SEQUENCES, *VAL_SEQUENCES, *TEST_SEQUENCES)
EXPECTED_SOURCE_FRAMES = {
    "Bosphorus": 600,
    "HoneyBee": 600,
    "YachtRide": 600,
    "ShakeNDry": 300,
    "Beauty": 600,
}
LOCAL_SOURCE_DIRS = {
    "Bosphorus": "UVG_bosphorous",
    "HoneyBee": "UVG_Honeybee",
    "YachtRide": "UVG_yachtride",
}
CROP_POSITIONS = (
    ("top_left", 0, 0),
    ("top_right", 1408, 0),
    ("centre", 704, 284),
    ("bottom_left", 0, 568),
    ("bottom_right", 1408, 568),
)


@dataclass(frozen=True)
class OfficialArchive:
    sequence: str
    url: str
    filename: str
    expected_archive_bytes: int
    expected_raw_bytes: int
    frame_rate: int = 120
    width: int = SOURCE_WIDTH
    height: int = SOURCE_HEIGHT
    pixel_format: str = "yuv420p"


OFFICIAL_ARCHIVES: Mapping[str, OfficialArchive] = {
    "Beauty": OfficialArchive(
        sequence="Beauty",
        url=(
            "https://ultravideo.fi/video/"
            "Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z"
        ),
        filename="Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z",
        expected_archive_bytes=925_430_047,
        expected_raw_bytes=1_866_240_000,
    ),
    "ShakeNDry": OfficialArchive(
        sequence="ShakeNDry",
        url=(
            "https://ultravideo.fi/video/"
            "ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z"
        ),
        filename="ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z",
        expected_archive_bytes=460_046_003,
        expected_raw_bytes=933_120_000,
    ),
}


def _frame_indices(sequence: str) -> Tuple[int, ...]:
    """Return the protocol's fixed, midpoint-in-bin source-frame indices."""

    frame_count = EXPECTED_SOURCE_FRAMES[sequence]
    return tuple(range(12, frame_count, 25))


def _crop_boxes() -> Tuple[Tuple[str, Tuple[int, int, int, int]], ...]:
    boxes = []
    for label, x, y in CROP_POSITIONS:
        if x % 2 or y % 2:
            raise ValueError("YUV420 crop origins must be even")
        box = (x, y, x + CROP_SIZE, y + CROP_SIZE)
        if box[2] > SOURCE_WIDTH or box[3] > SOURCE_HEIGHT:
            raise ValueError(f"Crop {label} is outside the source frame")
        boxes.append((label, box))
    return tuple(boxes)


def _expected_crop_filenames(sequence: str) -> Tuple[str, ...]:
    return tuple(
        sorted(
            f"f{frame_index:06d}_x{x:04d}_y{y:04d}.png"
            for frame_index in _frame_indices(sequence)
            for _label, x, y in CROP_POSITIONS
        )
    )


def _expected_relative_paths(
    sequences: Sequence[str],
) -> Tuple[str, ...]:
    return tuple(
        sorted(
            f"{sequence}/{filename}"
            for sequence in sequences
            for filename in _expected_crop_filenames(sequence)
        )
    )


def _natural_key(path: Path) -> Tuple[object, ...]:
    parts = re.split(r"(\d+)", path.as_posix().casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_files(root: Path, suffix: str) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == suffix.casefold()
        ),
        key=_natural_key,
    )


def _require_tool(*names: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    joined = " or ".join(names)
    raise RuntimeError(f"Required command is missing: {joined}")


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True, shell=False)


def _download_archive(
    spec: OfficialArchive,
    downloads_dir: Path,
    wget: str,
) -> Path:
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive = downloads_dir / spec.filename
    partial = downloads_dir / f"{spec.filename}.part"

    if archive.exists():
        observed = archive.stat().st_size
        if observed != spec.expected_archive_bytes:
            raise ValueError(
                f"Existing archive has the wrong size: {archive} "
                f"(expected {spec.expected_archive_bytes}, observed {observed})"
            )
        print(f"[download] Reusing {archive}")
        return archive

    if partial.exists() and partial.stat().st_size > spec.expected_archive_bytes:
        raise ValueError(
            f"Partial archive is larger than expected; move it aside: {partial}"
        )

    print(f"[download] {spec.sequence}: {spec.url}")
    _run(
        [
            wget,
            "--continue",
            "--tries=5",
            "--timeout=30",
            "--output-document",
            str(partial),
            spec.url,
        ]
    )
    observed = partial.stat().st_size
    if observed != spec.expected_archive_bytes:
        raise ValueError(
            f"Download size mismatch for {spec.sequence}: expected "
            f"{spec.expected_archive_bytes}, observed {observed}. "
            f"The resumable partial file was retained at {partial}."
        )
    os.replace(partial, archive)
    return archive


def _choose_yuv(root: Path, sequence: str) -> Path:
    candidates = _find_files(root, ".yuv")
    if not candidates:
        raise FileNotFoundError(f"No YUV file found below {root}")

    preferred = [
        path
        for path in candidates
        if sequence.casefold() in path.name.casefold()
        and "1920x1080" in path.name.casefold()
        and "8bit" in path.name.casefold()
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Could not choose one 1920x1080 8-bit YUV source for {sequence}: "
        + ", ".join(str(path) for path in candidates)
    )


def _validate_raw_yuv(
    path: Path,
    sequence: str,
    *,
    expected_bytes: Optional[int] = None,
) -> int:
    name = path.name.casefold()
    if "10bit" in name or "3840x2160" in name:
        raise ValueError(
            f"{sequence} must use the official 1920x1080 8-bit source: {path}"
        )

    expected_frames = EXPECTED_SOURCE_FRAMES[sequence]
    bytes_per_frame = SOURCE_WIDTH * SOURCE_HEIGHT * 3 // 2
    observed_bytes = path.stat().st_size
    required_bytes = expected_frames * bytes_per_frame
    if expected_bytes is not None and expected_bytes != required_bytes:
        raise AssertionError("Official raw-byte catalog is internally inconsistent")
    if observed_bytes != required_bytes:
        raise ValueError(
            f"{sequence} raw YUV has the wrong size: expected {required_bytes} "
            f"bytes ({expected_frames} frames), observed {observed_bytes}: {path}"
        )
    return expected_frames


def _extract_archive(
    spec: OfficialArchive,
    archive: Path,
    raw_root: Path,
    seven_zip: str,
) -> Path:
    final_dir = raw_root / spec.sequence
    if final_dir.exists():
        raw_path = _choose_yuv(final_dir, spec.sequence)
        _validate_raw_yuv(
            raw_path,
            spec.sequence,
            expected_bytes=spec.expected_raw_bytes,
        )
        integrity_path = final_dir / "_source_integrity.json"
        if not integrity_path.is_file():
            raise FileNotFoundError(
                f"Refusing unverified extracted-data reuse: {integrity_path}"
            )
        with integrity_path.open("r", encoding="utf-8") as handle:
            integrity = json.load(handle)
        expected_identity = {
            "sequence": spec.sequence,
            "url": spec.url,
            "archive_filename": spec.filename,
            "archive_bytes": spec.expected_archive_bytes,
            "raw_bytes": spec.expected_raw_bytes,
        }
        for key, value in expected_identity.items():
            if integrity.get(key) != value:
                raise ValueError(
                    f"Stored {spec.sequence} source identity changed at "
                    f"{integrity_path}: {key}"
                )
        if integrity.get("archive_sha256") != _sha256_file(archive):
            raise ValueError(
                f"Stored {spec.sequence} archive hash no longer matches"
            )
        if integrity.get("raw_sha256") != _sha256_file(raw_path):
            raise ValueError(
                f"Stored {spec.sequence} raw YUV hash no longer matches"
            )
        print(f"[extract] Reusing {raw_path}")
        return raw_path

    raw_root.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{spec.sequence}-extract-",
            dir=str(raw_root),
        )
    )
    try:
        print(f"[extract] {spec.sequence}")
        _run([seven_zip, "t", str(archive)])
        _run([seven_zip, "x", "-y", f"-o{stage}", str(archive)])
        staged_yuv = _choose_yuv(stage, spec.sequence)
        _validate_raw_yuv(
            staged_yuv,
            spec.sequence,
            expected_bytes=spec.expected_raw_bytes,
        )
        integrity = {
            "sequence": spec.sequence,
            "url": spec.url,
            "archive_filename": spec.filename,
            "archive_bytes": spec.expected_archive_bytes,
            "archive_sha256": _sha256_file(archive),
            "raw_bytes": spec.expected_raw_bytes,
            "raw_sha256": _sha256_file(staged_yuv),
        }
        with (stage / "_source_integrity.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(integrity, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(stage, final_dir)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    return _choose_yuv(final_dir, spec.sequence)


def _ffmpeg_version(ffmpeg: str) -> str:
    completed = subprocess.run(
        [ffmpeg, "-version"],
        check=True,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.splitlines()[0].strip()


def _ffmpeg_command(
    ffmpeg: str,
    raw_yuv: Path,
    frame_indices: Sequence[int],
    output_pattern: Path,
) -> List[str]:
    select = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    filters = (
        f"select={select},"
        "scale=1920:1080:"
        "flags=bitexact+accurate_rnd+full_chroma_int:"
        "in_range=tv:out_range=pc:in_color_matrix=bt709,"
        "format=rgb24"
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv420p",
        "-video_size",
        "1920x1080",
        "-framerate",
        "120",
        "-i",
        str(raw_yuv),
        "-vf",
        filters,
        "-vsync",
        "0",
        "-frames:v",
        str(len(frame_indices)),
        str(output_pattern),
    ]


def _extract_selected_frames(
    ffmpeg: str,
    raw_yuv: Path,
    frame_indices: Sequence[int],
    temp_dir: Path,
) -> List[Path]:
    output_pattern = temp_dir / "selected_%03d.png"
    command = _ffmpeg_command(
        ffmpeg,
        raw_yuv,
        frame_indices,
        output_pattern,
    )
    _run(command)
    frames = _find_files(temp_dir, ".png")
    if len(frames) != len(frame_indices):
        raise RuntimeError(
            f"ffmpeg produced {len(frames)} frames; expected "
            f"{len(frame_indices)} from {raw_yuv}"
        )
    return frames


def _validate_png_source(
    sequence: str,
    source_dir: Path,
) -> Tuple[List[Path], str]:
    frames = _find_files(source_dir, ".png")
    expected = EXPECTED_SOURCE_FRAMES[sequence]
    if len(frames) < expected:
        raise ValueError(
            f"{sequence} must contain {expected} full 1920x1080 PNG frames "
            f"or one official raw YUV; found {len(frames)} PNG files "
            f"below {source_dir}"
        )

    seen_casefold: Dict[str, Path] = {}
    indexed_frames: Dict[int, List[Path]] = {}
    pixel_hashes: Dict[Path, str] = {}
    for frame_number, frame in enumerate(frames, start=1):
        if frame_number == 1 or frame_number % 100 == 0:
            print(
                f"[verify] {sequence} PNG {frame_number}/{len(frames)}"
            )
        relative = frame.relative_to(source_dir).as_posix()
        collision = seen_casefold.setdefault(relative.casefold(), frame)
        if collision != frame:
            raise ValueError(
                f"Case-folding filename collision: {collision} and {frame}"
            )
        with Image.open(frame) as image:
            rgb = image.convert("RGB")
            rgb.load()
            if rgb.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise ValueError(
                    f"{sequence} source frame is {rgb.size[0]}x{rgb.size[1]}; "
                    f"expected 1920x1080: {frame}"
                )
            digest = hashlib.sha256()
            digest.update(
                f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}:RGB\0".encode("ascii")
            )
            digest.update(rgb.tobytes())
            pixel_hashes[frame] = digest.hexdigest()

        numeric_tokens = re.findall(r"\d+", frame.stem)
        if numeric_tokens:
            indexed_frames.setdefault(int(numeric_tokens[-1]), []).append(frame)

    all_frames_are_indexed = sum(
        len(paths) for paths in indexed_frames.values()
    ) == len(frames)
    selected = frames
    if all_frames_are_indexed:
        if len(indexed_frames) != expected:
            raise ValueError(
                f"{sequence} has {len(frames)} PNG files but they do not "
                f"resolve to {expected} numeric source-frame indices"
            )
        numeric_indices = sorted(indexed_frames)
        contiguous_zero_based = numeric_indices == list(range(expected))
        contiguous_one_based = numeric_indices == list(range(1, expected + 1))
        if not contiguous_zero_based and not contiguous_one_based:
            raise ValueError(
                f"{sequence} PNG frame indices are not contiguous 0.."
                f"{expected - 1} or 1..{expected}"
            )

        selected = []
        for numeric_index in numeric_indices:
            aliases = sorted(indexed_frames[numeric_index], key=_natural_key)
            hashes = {pixel_hashes[path] for path in aliases}
            if len(hashes) != 1:
                raise ValueError(
                    f"{sequence} contains conflicting PNG aliases for frame "
                    f"{numeric_index}: "
                    + ", ".join(str(path) for path in aliases)
                )
            selected.append(aliases[0])
    elif len(frames) != expected:
        raise ValueError(
            f"{sequence} has {len(frames)} PNG files, but not every filename "
            "has a numeric frame index that can safely collapse aliases"
        )

    aggregate = hashlib.sha256()
    for path in selected:
        aggregate.update(path.relative_to(source_dir).as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(pixel_hashes[path]))
        aggregate.update(b"\0")
    return selected, aggregate.hexdigest()


def _write_crops(
    sequence: str,
    source_frames: Sequence[Path],
    frame_indices: Sequence[int],
    destination: Path,
) -> int:
    if len(source_frames) != len(frame_indices):
        raise ValueError("Source-frame and frame-index counts do not match")
    destination.mkdir(parents=True, exist_ok=False)
    written = 0
    for source_path, frame_index in zip(source_frames, frame_indices):
        with Image.open(source_path) as source:
            image = source.convert("RGB")
            image.load()
            if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise ValueError(
                    f"{sequence} selected frame changed resolution: {source_path}"
                )
            for _label, (left, top, right, bottom) in _crop_boxes():
                output = destination / (
                    f"f{frame_index:06d}_x{left:04d}_y{top:04d}.png"
                )
                crop = image.crop((left, top, right, bottom))
                crop.save(
                    output,
                    format="PNG",
                    compress_level=6,
                    optimize=False,
                )
                written += 1
    return written


def _prepare_from_yuv(
    sequence: str,
    raw_yuv: Path,
    destination: Path,
    ffmpeg: str,
) -> Dict[str, object]:
    frame_indices = _frame_indices(sequence)
    frame_count = _validate_raw_yuv(raw_yuv, sequence)
    with tempfile.TemporaryDirectory(
        prefix=f".{sequence}-frames-",
        dir=str(destination.parent),
    ) as temp_name:
        selected = _extract_selected_frames(
            ffmpeg,
            raw_yuv,
            frame_indices,
            Path(temp_name),
        )
        crop_count = _write_crops(
            sequence,
            selected,
            frame_indices,
            destination,
        )
    return {
        "source_kind": "raw_yuv420p",
        "source_path": str(raw_yuv.resolve()),
        "source_sha256": _sha256_file(raw_yuv),
        "source_bytes": raw_yuv.stat().st_size,
        "source_frame_count": frame_count,
        "selected_frame_indices": list(frame_indices),
        "crop_count": crop_count,
        "ffmpeg_command": _ffmpeg_command(
            ffmpeg,
            raw_yuv,
            frame_indices,
            Path("<temporary-frame-directory>/selected_%03d.png"),
        ),
    }


def _prepare_from_pngs(
    sequence: str,
    source_dir: Path,
    destination: Path,
) -> Dict[str, object]:
    frames, source_hash = _validate_png_source(sequence, source_dir)
    indices = _frame_indices(sequence)
    selected = [frames[index] for index in indices]
    crop_count = _write_crops(sequence, selected, indices, destination)
    return {
        "source_kind": "full_frame_png_directory",
        "source_path": str(source_dir.resolve()),
        "source_sha256": source_hash,
        "source_frame_count": len(frames),
        "selected_source_files": [
            path.relative_to(source_dir).as_posix() for path in selected
        ],
        "selected_frame_indices": list(indices),
        "crop_count": crop_count,
    }


def _prepare_local_sequence(
    sequence: str,
    source_dir: Path,
    destination: Path,
    ffmpeg: str,
) -> Dict[str, object]:
    if not source_dir.is_dir():
        raise NotADirectoryError(
            f"Missing local {sequence} source directory: {source_dir}"
        )
    yuv_files = _find_files(source_dir, ".yuv")
    if yuv_files:
        raw_yuv = _choose_yuv(source_dir, sequence)
        return _prepare_from_yuv(sequence, raw_yuv, destination, ffmpeg)
    return _prepare_from_pngs(sequence, source_dir, destination)


def _expected_crop_count(sequence: str) -> int:
    return len(_frame_indices(sequence)) * len(CROP_POSITIONS)


def _prepared_content_digest(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for sequence in ALL_SEQUENCES:
        for filename in _expected_crop_filenames(sequence):
            path = dataset_root / sequence / filename
            if not path.is_file():
                raise FileNotFoundError(f"Prepared crop is missing: {path}")
            relative = path.relative_to(dataset_root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(_sha256_file(path)))
            digest.update(b"\0")
    return digest.hexdigest()


def _validate_prepared_dataset(dataset_root: Path) -> Dict[str, object]:
    metadata_path = dataset_root / "_preparation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset metadata is missing: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (
        metadata.get("schema_version") != PREPARATION_SCHEMA_VERSION
        or metadata.get("protocol_name") != PROTOCOL_NAME
    ):
        raise ValueError("Prepared dataset uses an unsupported protocol")

    expected_splits = {
        "train": list(TRAIN_SEQUENCES),
        "val": list(VAL_SEQUENCES),
        "test": list(TEST_SEQUENCES),
    }
    if metadata.get("splits") != expected_splits:
        raise ValueError("Prepared dataset split assignment changed")
    if metadata.get("source_resolution") != [SOURCE_WIDTH, SOURCE_HEIGHT]:
        raise ValueError("Prepared dataset source resolution changed")
    if metadata.get("crop_resolution") != [CROP_SIZE, CROP_SIZE]:
        raise ValueError("Prepared dataset crop resolution changed")

    sampling = metadata.get("sampling")
    expected_positions = [
        {
            "label": label,
            "x": x,
            "y": y,
            "width": CROP_SIZE,
            "height": CROP_SIZE,
        }
        for label, x, y in CROP_POSITIONS
    ]
    if not isinstance(sampling, dict) or sampling.get(
        "zero_based_rule"
    ) != "12 + 25*k while index < source_frame_count":
        raise ValueError("Prepared dataset temporal sampling changed")
    if sampling.get("crop_positions") != expected_positions:
        raise ValueError("Prepared dataset crop positions changed")

    observed_directories = {
        path.name for path in dataset_root.iterdir() if path.is_dir()
    }
    if observed_directories != set(ALL_SEQUENCES):
        raise ValueError(
            "Prepared dataset top-level sequence directories changed"
        )

    sequence_metadata = metadata.get("sequences")
    if not isinstance(sequence_metadata, dict):
        raise ValueError("Prepared dataset sequence provenance is missing")
    for sequence in ALL_SEQUENCES:
        sequence_dir = dataset_root / sequence
        images = _find_files(sequence_dir, ".png")
        observed_names = tuple(
            sorted(path.relative_to(sequence_dir).as_posix() for path in images)
        )
        expected_names = _expected_crop_filenames(sequence)
        if observed_names != expected_names:
            raise ValueError(
                f"Prepared {sequence} crop filenames or count changed"
            )
        for path in images:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (CROP_SIZE, CROP_SIZE):
                    raise ValueError(
                        f"Prepared crop must be RGB 512x512: {path}"
                    )
        details = sequence_metadata.get(sequence)
        if not isinstance(details, dict):
            raise ValueError(f"Prepared {sequence} provenance is missing")
        if details.get("source_frame_count") != EXPECTED_SOURCE_FRAMES[sequence]:
            raise ValueError(f"Prepared {sequence} source-frame count changed")
        if details.get("selected_frame_indices") != list(
            _frame_indices(sequence)
        ):
            raise ValueError(f"Prepared {sequence} selected frames changed")
        if details.get("crop_count") != _expected_crop_count(sequence):
            raise ValueError(f"Prepared {sequence} crop count changed")

    observed_digest = _prepared_content_digest(dataset_root)
    if metadata.get("prepared_content_sha256") != observed_digest:
        raise ValueError("Prepared dataset content digest changed")
    return metadata


def _verify_protocol_bundle(bundle: FrozenSplitBundle) -> FrozenSplitBundle:
    if bundle.dataset_id != PROTOCOL_NAME:
        raise ValueError(
            f"Frozen bundle dataset ID is not {PROTOCOL_NAME}: "
            f"{bundle.dataset_id}"
        )
    if (bundle.image_width, bundle.image_height) != (CROP_SIZE, CROP_SIZE):
        raise ValueError("Frozen bundle resolution is not 512x512")

    expected_sequences = {
        "train": tuple(sorted(TRAIN_SEQUENCES)),
        "val": tuple(sorted(VAL_SEQUENCES)),
        "test": tuple(sorted(TEST_SEQUENCES)),
    }
    for split_name, sequences in expected_sequences.items():
        split = bundle.splits.get(split_name)
        if split is None or split.sequences != sequences:
            raise ValueError(
                f"Frozen bundle {split_name} sequence assignment changed"
            )
        expected_paths = _expected_relative_paths(sequences)
        if split.relative_paths != expected_paths:
            raise ValueError(
                f"Frozen bundle {split_name} paths or image count changed"
            )
    return bundle


def _bundle_summary(bundle: FrozenSplitBundle) -> Dict[str, object]:
    return {
        "bundle_id": bundle.bundle_id,
        "dataset_id": bundle.dataset_id,
        "dataset_root": bundle.dataset_root,
        "split_dir": bundle.split_dir,
        "resolution": [bundle.image_width, bundle.image_height],
        "splits": {
            name: {
                "sequences": list(split.sequences),
                "image_count": len(split.image_paths),
                "manifest_sha256": split.manifest_sha256,
                "file_sha256": split.file_sha256,
                "content_sha256": split.content_sha256,
            }
            for name, split in bundle.splits.items()
        },
    }


def _create_bundle_atomically(
    dataset_root: Path,
    split_dir: Path,
) -> FrozenSplitBundle:
    if split_dir.exists():
        return _verify_protocol_bundle(
            load_and_verify_frozen_split_bundle(
                split_dir=str(split_dir),
                dataset_root=str(dataset_root),
                expected_size=(CROP_SIZE, CROP_SIZE),
            )
        )

    split_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(
        tempfile.mkdtemp(
            prefix=".frozen-split-",
            dir=str(split_dir.parent),
        )
    )
    staged_bundle = temp_parent / "bundle"
    try:
        create_frozen_sequence_split_bundle(
            dataset_root=str(dataset_root),
            output_dir=str(staged_bundle),
            dataset_id=PROTOCOL_NAME,
            train_sequences=TRAIN_SEQUENCES,
            val_sequences=VAL_SEQUENCES,
            test_sequences=TEST_SEQUENCES,
        )
        os.replace(staged_bundle, split_dir)
    finally:
        if temp_parent.exists():
            shutil.rmtree(temp_parent)

    return _verify_protocol_bundle(
        load_and_verify_frozen_split_bundle(
            split_dir=str(split_dir),
            dataset_root=str(dataset_root),
            expected_size=(CROP_SIZE, CROP_SIZE),
        )
    )


def prepare_uvg_dsad_screen(
    source_root: Path,
    output_root: Optional[Path] = None,
) -> FrozenSplitBundle:
    """Prepare or verify the fixed screening dataset and frozen split."""

    source_root = source_root.expanduser().resolve()
    output_root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else source_root / "UVG_DSAD_screen_v1"
    )
    dataset_root = output_root / "dataset_512"
    split_dir = output_root / "frozen_split"

    if split_dir.exists():
        _validate_prepared_dataset(dataset_root)
        return _create_bundle_atomically(dataset_root, split_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    if dataset_root.exists():
        _validate_prepared_dataset(dataset_root)
        return _create_bundle_atomically(dataset_root, split_dir)

    available = shutil.disk_usage(output_root).free
    if available < MINIMUM_FREE_BYTES:
        raise OSError(
            "UVG preparation requires at least 6 GiB free in the output "
            f"filesystem; only {available / 1024**3:.2f} GiB is available"
        )

    wget = _require_tool("wget")
    seven_zip = _require_tool("7zz", "7z")
    ffmpeg = _require_tool("ffmpeg")

    downloads_dir = output_root / "downloads"
    raw_root = output_root / "official_raw"
    official_yuvs: Dict[str, Path] = {}
    archive_metadata: Dict[str, Dict[str, object]] = {}
    for sequence in ("ShakeNDry", "Beauty"):
        spec = OFFICIAL_ARCHIVES[sequence]
        archive = _download_archive(spec, downloads_dir, wget)
        raw_yuv = _extract_archive(spec, archive, raw_root, seven_zip)
        official_yuvs[sequence] = raw_yuv
        archive_metadata[sequence] = {
            **asdict(spec),
            "archive_path": str(archive.resolve()),
            "archive_sha256": _sha256_file(archive),
            "raw_path": str(raw_yuv.resolve()),
            "raw_sha256": _sha256_file(raw_yuv),
        }

    dataset_parent = dataset_root.parent
    stage = Path(
        tempfile.mkdtemp(
            prefix=".dataset-512-",
            dir=str(dataset_parent),
        )
    )
    try:
        sequence_metadata: Dict[str, Dict[str, object]] = {}
        for sequence in TRAIN_SEQUENCES:
            print(f"[prepare] {sequence}")
            source_dir = source_root / LOCAL_SOURCE_DIRS[sequence]
            sequence_metadata[sequence] = _prepare_local_sequence(
                sequence,
                source_dir,
                stage / sequence,
                ffmpeg,
            )
        for sequence in (*VAL_SEQUENCES, *TEST_SEQUENCES):
            print(f"[prepare] {sequence}")
            sequence_metadata[sequence] = _prepare_from_yuv(
                sequence,
                official_yuvs[sequence],
                stage / sequence,
                ffmpeg,
            )

        metadata = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "protocol_name": PROTOCOL_NAME,
            "purpose": "internal_causal_dsad_screening_not_publication_benchmark",
            "license": {
                "name": "Creative Commons BY-NC 3.0 Unported",
                "url": "https://creativecommons.org/licenses/by-nc/3.0/",
            },
            "source_resolution": [SOURCE_WIDTH, SOURCE_HEIGHT],
            "crop_resolution": [CROP_SIZE, CROP_SIZE],
            "pixel_format": "RGB8_PNG",
            "raw_conversion": {
                "input": "YUV420p 8-bit, BT.709 limited-range assumption",
                "output": "RGB8 full-range",
                "ffmpeg_version": _ffmpeg_version(ffmpeg),
                "threads": 1,
            },
            "sampling": {
                "zero_based_rule": "12 + 25*k while index < source_frame_count",
                "crop_positions": [
                    {
                        "label": label,
                        "x": x,
                        "y": y,
                        "width": CROP_SIZE,
                        "height": CROP_SIZE,
                    }
                    for label, x, y in CROP_POSITIONS
                ],
            },
            "splits": {
                "train": list(TRAIN_SEQUENCES),
                "val": list(VAL_SEQUENCES),
                "test": list(TEST_SEQUENCES),
            },
            "test_policy": (
                "Beauty is locked. Do not preview reconstructions, calculate "
                "metrics, or pass --evaluate-test until every DSAD decision is frozen."
            ),
            "official_archives": archive_metadata,
            "sequences": sequence_metadata,
        }
        metadata["prepared_content_sha256"] = _prepared_content_digest(stage)
        with (stage / "_preparation_metadata.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(stage, dataset_root)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    _validate_prepared_dataset(dataset_root)
    return _create_bundle_atomically(dataset_root, split_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the fixed sequence-disjoint UVG dataset for causal-DSAD "
            "screening. Beauty remains locked; this command computes no metrics."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help=(
            "Directory containing UVG_bosphorous, UVG_Honeybee and "
            "UVG_yachtride."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Output directory. Defaults to "
            "<source-root>/UVG_DSAD_screen_v1."
        ),
    )
    parser.add_argument(
        "--accept-uvg-by-nc",
        action="store_true",
        help=(
            "Confirm use under the UVG Creative Commons BY-NC "
            "non-commercial licence."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.accept_uvg_by_nc:
        raise SystemExit(
            "Refusing to download UVG data without --accept-uvg-by-nc. "
            "Dataset licence: https://creativecommons.org/licenses/by-nc/3.0/"
        )
    bundle = prepare_uvg_dsad_screen(
        source_root=args.source_root,
        output_root=args.output_root,
    )
    print("\nPreparation complete. Beauty remains locked.")
    print(json.dumps(_bundle_summary(bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
