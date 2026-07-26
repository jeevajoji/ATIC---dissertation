import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

try:
    import torch

    from atic.compressai_oracle import (
        EXPECTED_BUNDLE_ID,
        _build_monotonicity_report,
        _entropy_stream_byte_sizes,
        _parse_qualities,
        evaluate_quality,
        run_oracle,
    )
    from atic.dataset import FrozenSplit, FrozenSplitBundle

    ORACLE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    ORACLE_IMPORT_ERROR = exc


@unittest.skipIf(
    ORACLE_IMPORT_ERROR is not None,
    f"PyTorch oracle stack unavailable: {ORACLE_IMPORT_ERROR}",
)
class CompressAIOracleTests(unittest.TestCase):
    class FakeCodec(
        torch.nn.Module if ORACLE_IMPORT_ERROR is None else object
    ):
        def __init__(self, quality):
            super().__init__()
            self.quality = int(quality)
            self.last_input = None
            self.update_calls = []
            self.compress_calls = 0
            self.decompress_calls = 0

        def update(self, force=False):
            self.update_calls.append(bool(force))
            return False

        def compress(self, batch):
            self.compress_calls += 1
            self.last_input = batch.detach().clone()
            return {
                "strings": [
                    [b"y" * (16 * self.quality)],
                    [b"z" * (8 * self.quality)],
                ],
                "shape": (1, 1),
            }

        def decompress(self, *, strings, shape):
            self.decompress_calls += 1
            if self.last_input is None:
                raise RuntimeError("compress must run before decompress")
            self.assert_entropy_payload(strings, shape)
            error = 0.2 / self.quality
            return {"x_hat": (self.last_input + error).clamp(0.0, 1.0)}

        @staticmethod
        def assert_entropy_payload(strings, shape):
            if len(strings) != 2 or tuple(shape) != (1, 1):
                raise AssertionError("unexpected fake entropy payload")

    def test_quality_parser_requires_unique_official_points(self):
        self.assertEqual(_parse_qualities("4,1,2"), [1, 2, 4])
        for raw in ("", "1", "1,1", "0,2", "1,9", "one,2"):
            with self.subTest(raw=raw):
                with self.assertRaises(Exception):
                    _parse_qualities(raw)

    def test_entropy_payload_counts_each_top_level_stream(self):
        self.assertEqual(
            _entropy_stream_byte_sizes(
                [
                    [b"abc", b"de"],
                    (b"1234",),
                ]
            ),
            [5, 4],
        )
        for invalid in (None, [], [None], [[b""]]):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    _entropy_stream_byte_sizes(invalid)

    def test_evaluate_quality_uses_compress_and_decompress_payload(self):
        codec = self.FakeCodec(quality=2)
        batch = torch.zeros(1, 3, 8, 8)

        result = evaluate_quality(
            codec,
            [batch],
            quality=2,
            device="cpu",
        )

        self.assertEqual(codec.compress_calls, 1)
        self.assertEqual(codec.decompress_calls, 1)
        self.assertEqual(codec.update_calls, [False])
        self.assertEqual(result["payload_bytes"], 48)
        self.assertEqual(result["stream_bytes"], [32, 16])
        self.assertAlmostEqual(result["BPP"], 6.0)
        self.assertAlmostEqual(result["MSE"], 0.01, places=6)
        self.assertAlmostEqual(result["PSNR"], 20.0, places=5)

    def test_run_oracle_is_validation_only_and_writes_strict_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            val_dir = root / "ShakeNDry"
            val_dir.mkdir()
            val_paths = []
            for index in range(2):
                path = val_dir / f"frame_{index:03d}.png"
                Image.new("RGB", (512, 512), color=(0, 0, 0)).save(path)
                val_paths.append(str(path))

            # Deliberately nonexistent: the oracle must never construct or
            # iterate a dataset for the locked Beauty split.
            locked_test_path = root / "Beauty" / "must_not_be_opened.png"
            bundle = FrozenSplitBundle(
                dataset_id="uvg_dsad_screen_v1",
                dataset_root=str(root),
                split_dir=str(root / "split"),
                bundle_id=EXPECTED_BUNDLE_ID,
                image_width=512,
                image_height=512,
                splits={
                    "val": FrozenSplit(
                        name="val",
                        sequences=("ShakeNDry",),
                        relative_paths=(
                            "ShakeNDry/frame_000.png",
                            "ShakeNDry/frame_001.png",
                        ),
                        image_paths=tuple(val_paths),
                        manifest_path=str(root / "val_manifest.txt"),
                        manifest_sha256="val-manifest",
                        file_sha256="val-files",
                        content_sha256="val-content",
                    ),
                    "test": FrozenSplit(
                        name="test",
                        sequences=("Beauty",),
                        relative_paths=("Beauty/must_not_be_opened.png",),
                        image_paths=(str(locked_test_path),),
                        manifest_path=str(root / "test_manifest.txt"),
                        manifest_sha256="test-manifest",
                        file_sha256="test-files",
                        content_sha256="test-content",
                    ),
                },
            )
            loader_calls = []

            def fake_bundle_loader(**kwargs):
                loader_calls.append(kwargs)
                return bundle

            codecs = {}
            environment_calls = []

            def fake_factory(quality):
                codec = self.FakeCodec(quality)
                codecs[quality] = codec
                return codec

            def fake_environment_loader(**kwargs):
                environment_calls.append(kwargs)
                return {
                    "git": {"commit": "oracle-test", "is_dirty": False},
                    "deterministic_algorithms_enabled": True,
                }

            output = root / "oracle.json"
            report = run_oracle(
                dataset_root=str(root),
                frozen_split_dir=str(root / "split"),
                qualities=[1, 4],
                output_path=str(output),
                device="cpu",
                num_workers=0,
                pin_memory=False,
                model_factory=fake_factory,
                bundle_loader=fake_bundle_loader,
                environment_loader=fake_environment_loader,
                repo_dir=str(root),
                min_delta_bpp=0.001,
                expected_software=None,
            )

            self.assertEqual(
                loader_calls,
                [
                    {
                        "split_dir": str(root / "split"),
                        "dataset_root": str(root),
                        "expected_size": (512, 512),
                    }
                ],
            )
            self.assertFalse(locked_test_path.exists())
            self.assertEqual(
                environment_calls,
                [{"device": "cpu", "repo_dir": str(root.resolve())}],
            )
            self.assertEqual(report["protocol"]["evaluation_split"], "val")
            self.assertTrue(report["protocol"]["test_locked"])
            self.assertFalse(report["protocol"]["test_evaluated"])
            self.assertFalse(report["oracle"]["canonical_protocol"])
            self.assertTrue(report["monotonicity"]["passed"])
            self.assertEqual([row["quality"] for row in report["results"]], [1, 4])
            self.assertEqual(codecs[1].compress_calls, 2)
            self.assertEqual(codecs[4].compress_calls, 2)

            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written, report)
            with self.assertRaises(FileExistsError):
                run_oracle(
                    dataset_root=str(root),
                    frozen_split_dir=str(root / "split"),
                    qualities=[1, 4],
                    output_path=str(output),
                    device="cpu",
                    num_workers=0,
                    pin_memory=False,
                    model_factory=fake_factory,
                    bundle_loader=fake_bundle_loader,
                    environment_loader=fake_environment_loader,
                    repo_dir=str(root),
                    min_delta_bpp=0.001,
                    expected_software=None,
                )

    def test_monotonicity_requires_both_rate_and_quality_order(self):
        report = _build_monotonicity_report(
            [
                {"quality": 1, "BPP": 0.1, "PSNR": 25.0},
                {"quality": 2, "BPP": 0.2, "PSNR": 24.0},
            ]
        )
        self.assertTrue(report["BPP_non_decreasing"])
        self.assertFalse(report["PSNR_non_decreasing"])
        self.assertFalse(report["passed"])

    def test_monotonicity_rejects_equal_points_that_miss_margins(self):
        report = _build_monotonicity_report(
            [
                {"quality": 1, "BPP": 0.1, "PSNR": 25.0},
                {"quality": 8, "BPP": 0.1, "PSNR": 25.0},
            ]
        )
        self.assertTrue(report["BPP_non_decreasing"])
        self.assertTrue(report["PSNR_non_decreasing"])
        self.assertFalse(report["BPP_margin_passed"])
        self.assertFalse(report["PSNR_margin_passed"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
