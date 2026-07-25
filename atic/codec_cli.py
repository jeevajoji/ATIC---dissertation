"""Command-line interface for transferable ATIC files.

Examples::

    python -m atic.codec_cli compress \
        --input image.png --output image.atic \
        --checkpoint model.pth --config run_config.json \
        --height 512 --width 512

    python -m atic.codec_cli decompress \
        --input image.atic --output reconstruction.png \
        --checkpoint model.pth --config run_config.json \
        --height 512 --width 512
"""

from __future__ import annotations

import argparse
import hmac
import json
from dataclasses import fields
from pathlib import Path

from atic.bitstream import (
    ATICBitstreamError,
    normalise_model_hash,
    read_atic_file,
    sha256_file,
)
from atic.config import ArchitectureConfig


_MAX_CODED_DIMENSION = 16_384
_MAX_CODED_PIXELS = 8_192 * 8_192


def _configure_reference_runtime() -> None:
    """Select the CPU threading profile exercised by codec portability tests."""

    import torch

    torch.set_num_threads(1)


def _load_run_spec(
    path: str,
    *,
    height_override=None,
    width_override=None,
):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ATIC configuration must be a JSON object")
    architecture = payload.get("architecture", payload)
    if not isinstance(architecture, dict):
        raise ValueError("ATIC architecture configuration must be a JSON object")

    allowed = {field.name for field in fields(ArchitectureConfig)}
    unknown = sorted(set(architecture) - allowed)
    if unknown:
        raise ValueError(f"Unknown ATIC architecture fields: {', '.join(unknown)}")

    if (height_override is None) != (width_override is None):
        raise ValueError("--height and --width must be supplied together")
    configured_height = payload.get("height")
    configured_width = payload.get("width")
    if (configured_height is None) != (configured_width is None):
        raise ValueError("run_config.json must store both height and width")

    if height_override is not None:
        height = int(height_override)
        width = int(width_override)
        if configured_height is not None and (
            height != int(configured_height) or width != int(configured_width)
        ):
            raise ValueError(
                "CLI resolution does not match the trained resolution recorded "
                "in run_config.json"
            )
    elif configured_height is not None:
        height = int(configured_height)
        width = int(configured_width)
    else:
        raise ValueError(
            "This legacy run_config.json has no trained resolution; pass both "
            "--height and --width explicitly"
        )

    if (
        height <= 0
        or width <= 0
        or height > _MAX_CODED_DIMENSION
        or width > _MAX_CODED_DIMENSION
        or height * width > _MAX_CODED_PIXELS
    ):
        raise ValueError("ATIC coded resolution exceeds the CLI safety limit")
    return ArchitectureConfig(**architecture), height, width


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _load_model(config, checkpoint_path: str, height: int, width: int, device):
    from atic.model import ATICModel

    model = ATICModel(config, H=height, W=width)
    model.load_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def _load_image(path: str, device, *, expected_height: int, expected_width: int):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        if image.size != (expected_width, expected_height):
            raise ValueError(
                "Input image resolution does not match the checkpoint's trained "
                f"resolution: {image.height}x{image.width} versus "
                f"{expected_height}x{expected_width}"
            )
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _save_image(tensor, path: str) -> None:
    import numpy as np
    from PIL import Image

    array = (
        tensor[0]
        .detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    encoded = np.rint(array * 255.0).astype(np.uint8)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded, mode="RGB").save(target)


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_compress(args) -> None:
    _configure_reference_runtime()
    config, height, width = _load_run_spec(
        args.config,
        height_override=args.height,
        width_override=args.width,
    )
    device = _resolve_device(args.device)
    image = _load_image(
        args.input,
        device,
        expected_height=height,
        expected_width=width,
    )
    model = _load_model(
        config,
        args.checkpoint,
        height,
        width,
        device,
    )
    result = model.compress(
        image,
        output_path=args.output,
        quality_id=args.quality_id,
    )
    _print_json(
        {
            key: value
            for key, value in result.items()
            if key != "bitstream"
        }
    )


def command_decompress(args) -> None:
    _configure_reference_runtime()
    config, height, width = _load_run_spec(
        args.config,
        height_override=args.height,
        width_override=args.width,
    )
    container = read_atic_file(args.input)
    if (container.coded_height, container.coded_width) != (height, width):
        raise ATICBitstreamError(
            "ATIC coded resolution does not match the checkpoint's trained "
            f"resolution: {container.coded_height}x{container.coded_width} "
            f"versus {height}x{width}"
        )
    checkpoint_hash = normalise_model_hash(sha256_file(args.checkpoint))
    if not hmac.compare_digest(container.model_hash, checkpoint_hash):
        raise ATICBitstreamError(
            "ATIC model/checkpoint hash does not match the supplied checkpoint"
        )

    device = _resolve_device(args.device)
    model = _load_model(
        config,
        args.checkpoint,
        height,
        width,
        device,
    )
    result = model.decompress(args.input, return_info=True)
    _save_image(result["x_hat"], args.output)
    _print_json(
        {
            key: value
            for key, value in result.items()
            if key not in {"x_hat", "gain_map", "z_hat"}
        }
    )


def command_inspect(args) -> None:
    container = read_atic_file(args.input)
    payload_bytes = len(container.z_string) + len(container.y_string)
    _print_json(
        {
            "actual_bpp": (
                container.num_bytes
                * 8
                / (container.original_height * container.original_width)
            ),
            "bit_depth": container.bit_depth,
            "coded_height": container.coded_height,
            "coded_width": container.coded_width,
            "flags": container.flags,
            "format": "ATIC 1.0",
            "header_bytes": container.num_bytes - payload_bytes,
            "model_hash": container.model_hash_hex,
            "num_bytes": container.num_bytes,
            "original_height": container.original_height,
            "original_width": container.original_width,
            "payload_bytes": payload_bytes,
            "quality_id": container.quality_id,
            "y_bytes": len(container.y_string),
            "y_shape": [
                container.y_channels,
                container.y_height,
                container.y_width,
            ],
            "z_bytes": len(container.z_string),
            "z_shape": [
                container.z_height,
                container.z_width,
            ],
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress, decompress, or inspect transferable ATIC files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="encode an RGB image")
    compress_parser.add_argument("--input", required=True, help="source image path")
    compress_parser.add_argument("--output", required=True, help="output .atic path")
    compress_parser.add_argument("--checkpoint", required=True, help="model.pth path")
    compress_parser.add_argument("--config", required=True, help="run_config.json path")
    compress_parser.add_argument(
        "--height",
        type=int,
        help="trained height (required for legacy run configs)",
    )
    compress_parser.add_argument(
        "--width",
        type=int,
        help="trained width (required for legacy run configs)",
    )
    compress_parser.add_argument("--quality-id", type=int, default=0)
    compress_parser.add_argument("--device", default="cpu")
    compress_parser.set_defaults(handler=command_compress)

    decompress_parser = subparsers.add_parser(
        "decompress",
        help="decode a .atic file",
    )
    decompress_parser.add_argument("--input", required=True, help="source .atic path")
    decompress_parser.add_argument("--output", required=True, help="output image path")
    decompress_parser.add_argument("--checkpoint", required=True, help="model.pth path")
    decompress_parser.add_argument("--config", required=True, help="run_config.json path")
    decompress_parser.add_argument(
        "--height",
        type=int,
        help="trained height (required for legacy run configs)",
    )
    decompress_parser.add_argument(
        "--width",
        type=int,
        help="trained width (required for legacy run configs)",
    )
    decompress_parser.add_argument("--device", default="cpu")
    decompress_parser.set_defaults(handler=command_decompress)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="validate a .atic file and print its header",
    )
    inspect_parser.add_argument("--input", required=True, help="source .atic path")
    inspect_parser.set_defaults(handler=command_inspect)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
