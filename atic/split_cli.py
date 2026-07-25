"""Create and verify immutable sequence-disjoint dataset split bundles."""

import argparse
import json
from typing import List

from atic.dataset import (
    create_frozen_sequence_split_bundle,
    load_and_verify_frozen_split_bundle,
)


def _csv_values(raw: str) -> List[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one sequence is required")
    return values


def _bundle_summary(bundle) -> dict:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify frozen train/validation/test manifests whose "
            "source sequences cannot overlap."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--dataset-root", required=True)
    create.add_argument("--output-dir", required=True)
    create.add_argument("--dataset-id", required=True)
    create.add_argument("--train-sequences", required=True, type=_csv_values)
    create.add_argument("--val-sequences", required=True, type=_csv_values)
    create.add_argument("--test-sequences", required=True, type=_csv_values)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--dataset-root", required=True)
    verify.add_argument("--split-dir", required=True)
    verify.add_argument("--width", type=int)
    verify.add_argument("--height", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "create":
        bundle = create_frozen_sequence_split_bundle(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            train_sequences=args.train_sequences,
            val_sequences=args.val_sequences,
            test_sequences=args.test_sequences,
        )
    else:
        if (args.width is None) != (args.height is None):
            raise ValueError("--width and --height must be provided together")
        expected_size = (
            None
            if args.width is None
            else (args.width, args.height)
        )
        bundle = load_and_verify_frozen_split_bundle(
            split_dir=args.split_dir,
            dataset_root=args.dataset_root,
            expected_size=expected_size,
        )
    print(json.dumps(_bundle_summary(bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
