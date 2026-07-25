import json
import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    from atic.dataset import (
        create_frozen_sequence_split_bundle,
        get_frozen_split_dataloaders,
        load_and_verify_frozen_split_bundle,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight local hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"ATIC dataset dependencies unavailable: {IMPORT_ERROR}",
)
class FrozenSequenceSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "dataset"
        self.sequences = {
            "seq_train": (10, 20, 30),
            "seq_val": (40, 50, 60),
            "seq_test": (70, 80, 90),
        }
        for sequence, base_colour in self.sequences.items():
            sequence_dir = self.root / sequence
            sequence_dir.mkdir(parents=True)
            for index in range(2):
                colour = tuple(value + index for value in base_colour)
                Image.new("RGB", (16, 16), colour).save(
                    sequence_dir / f"{index:03d}.png"
                )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create(self, name="split"):
        return create_frozen_sequence_split_bundle(
            dataset_root=str(self.root),
            output_dir=str(Path(self.temp_dir.name) / name),
            dataset_id="tiny-sequence-fixture",
            train_sequences=["seq_train"],
            val_sequences=["seq_val"],
            test_sequences=["seq_test"],
        )

    def test_creation_is_deterministic_and_portable(self):
        first = self._create("split_a")
        second = self._create("split_b")
        self.assertEqual(first.bundle_id, second.bundle_id)
        for split_name in ("train", "val", "test"):
            self.assertEqual(
                first.splits[split_name].relative_paths,
                second.splits[split_name].relative_paths,
            )

        relocated_root = Path(self.temp_dir.name) / "relocated"
        shutil.copytree(self.root, relocated_root)
        relocated = load_and_verify_frozen_split_bundle(
            first.split_dir,
            str(relocated_root),
            expected_size=(16, 16),
        )
        self.assertEqual(relocated.bundle_id, first.bundle_id)
        self.assertTrue(
            all(
                path.startswith(str(relocated_root))
                for split in relocated.splits.values()
                for path in split.image_paths
            )
        )

    def test_refuses_overwrite_and_sequence_overlap(self):
        bundle = self._create()
        with self.assertRaises(FileExistsError):
            self._create()
        with self.assertRaises(ValueError):
            create_frozen_sequence_split_bundle(
                dataset_root=str(self.root),
                output_dir=str(Path(self.temp_dir.name) / "overlap"),
                dataset_id="bad-overlap",
                train_sequences=["seq_train"],
                val_sequences=["seq_train"],
                test_sequences=["seq_test"],
            )
        self.assertTrue(os.path.isdir(bundle.split_dir))

    def test_rejects_exact_duplicate_content_across_splits(self):
        source_path = self.root / "seq_train" / "000.png"
        duplicate_path = self.root / "seq_test" / "000.png"
        metadata = PngInfo()
        metadata.add_text("different-file-bytes", "same-decoded-RGB")
        with Image.open(source_path) as source:
            source.convert("RGB").save(duplicate_path, pnginfo=metadata)
        self.assertNotEqual(source_path.read_bytes(), duplicate_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate image content"):
            self._create()

    def test_detects_manifest_tampering(self):
        bundle = self._create()
        manifest = Path(bundle.splits["val"].manifest_path)
        manifest.write_text(
            manifest.read_text(encoding="utf-8") + "seq_val/extra.png\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            load_and_verify_frozen_split_bundle(
                bundle.split_dir,
                str(self.root),
            )

    def test_detects_file_change_even_when_decoded_pixels_are_identical(self):
        bundle = self._create()
        image_path = self.root / "seq_train" / "001.png"
        metadata = PngInfo()
        metadata.add_text("changed", "metadata-only")
        with Image.open(image_path) as source:
            pixels = source.convert("RGB")
            pixels.save(image_path, pnginfo=metadata)
        with self.assertRaisesRegex(ValueError, "image files"):
            load_and_verify_frozen_split_bundle(
                bundle.split_dir,
                str(self.root),
            )

    def test_detects_image_change_and_resolution_mismatch(self):
        bundle = self._create()
        Image.new("RGB", (16, 16), (1, 2, 3)).save(
            self.root / "seq_test" / "001.png"
        )
        with self.assertRaisesRegex(ValueError, "no longer match"):
            load_and_verify_frozen_split_bundle(
                bundle.split_dir,
                str(self.root),
            )

        with open(
            Path(bundle.split_dir) / "split_metadata.json",
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(handle)
        self.assertEqual(metadata["image_width"], 16)
        with self.assertRaisesRegex(ValueError, "experiment expects"):
            load_and_verify_frozen_split_bundle(
                bundle.split_dir,
                str(self.root),
                expected_size=(32, 32),
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchvision") is not None,
        "torch and torchvision are required for DataLoader verification",
    )
    def test_loader_counts_and_order_match_verified_manifests(self):
        bundle = self._create()
        loaders = get_frozen_split_dataloaders(
            bundle,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            seed=42,
        )
        self.assertEqual(len(loaders.train.dataset), 2)
        self.assertEqual(len(loaders.val.dataset), 2)
        self.assertEqual(len(loaders.test.dataset), 2)
        self.assertEqual(
            tuple(loaders.train.dataset.image_paths),
            bundle.splits["train"].image_paths,
        )


if __name__ == "__main__":
    unittest.main()
