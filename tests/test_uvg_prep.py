import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from atic.uvg_prep import (
    CROP_POSITIONS,
    OFFICIAL_ARCHIVES,
    PROTOCOL_NAME,
    _crop_boxes,
    _ffmpeg_command,
    _frame_indices,
    _prepare_from_pngs,
    _validate_prepared_dataset,
    build_parser,
)


class UVGPreparationTests(unittest.TestCase):
    def test_official_catalog_is_pinned(self):
        beauty = OFFICIAL_ARCHIVES["Beauty"]
        shake = OFFICIAL_ARCHIVES["ShakeNDry"]
        self.assertEqual(
            beauty.filename,
            "Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z",
        )
        self.assertEqual(beauty.expected_archive_bytes, 925_430_047)
        self.assertEqual(beauty.expected_raw_bytes, 1_866_240_000)
        self.assertEqual(shake.expected_archive_bytes, 460_046_003)
        self.assertEqual(shake.expected_raw_bytes, 933_120_000)
        self.assertTrue(beauty.url.startswith("https://ultravideo.fi/video/"))

    def test_sampling_and_crop_plan_are_fixed_and_in_bounds(self):
        self.assertEqual(
            _frame_indices("Beauty"),
            tuple(range(12, 600, 25)),
        )
        self.assertEqual(
            _frame_indices("ShakeNDry"),
            tuple(range(12, 300, 25)),
        )
        self.assertEqual(len(_frame_indices("Bosphorus")), 24)
        self.assertEqual(len(_frame_indices("ShakeNDry")), 12)
        boxes = _crop_boxes()
        self.assertEqual(len(boxes), 5)
        for _label, (left, top, right, bottom) in boxes:
            self.assertEqual(left % 2, 0)
            self.assertEqual(top % 2, 0)
            self.assertEqual(right - left, 512)
            self.assertEqual(bottom - top, 512)
            self.assertLessEqual(right, 1920)
            self.assertLessEqual(bottom, 1080)

        for index, (_label_a, box_a) in enumerate(boxes):
            for _label_b, box_b in boxes[index + 1 :]:
                overlap_width = max(
                    0,
                    min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]),
                )
                overlap_height = max(
                    0,
                    min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]),
                )
                self.assertEqual(overlap_width * overlap_height, 0)

    def test_ffmpeg_command_is_an_argument_list_and_handles_spaces(self):
        raw_path = Path("/tmp/raw data/Beauty.yuv")
        output_path = Path("/tmp/output frames/frame_%03d.png")
        command = _ffmpeg_command(
            "/usr/bin/ffmpeg",
            raw_path,
            (12, 37),
            output_path,
        )
        self.assertIsInstance(command, list)
        self.assertIn(str(raw_path), command)
        self.assertIn(str(output_path), command)
        self.assertIn("select=eq(n\\,12)+eq(n\\,37)", command[command.index("-vf") + 1])
        self.assertEqual(command[command.index("-frames:v") + 1], "2")

    def test_png_source_preparation_writes_expected_normalised_crops(self):
        from atic import uvg_prep

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source"
            destination = root / "prepared"
            source.mkdir()
            expected = 50
            for index in range(1, expected + 1):
                image = Image.new(
                    "RGB",
                    (16, 16),
                    (index % 251, (index * 3) % 251, (index * 7) % 251),
                )
                image.save(source / f"{index:04d}.png")
                alias_metadata = PngInfo()
                alias_metadata.add_text("alias", f"frame-{index}")
                image.save(
                    source / f"frame_{index:04d}.png",
                    pnginfo=alias_metadata,
                )

            with mock.patch.dict(
                uvg_prep.EXPECTED_SOURCE_FRAMES,
                {"ShakeNDry": expected},
            ), mock.patch.multiple(
                uvg_prep,
                SOURCE_WIDTH=16,
                SOURCE_HEIGHT=16,
                CROP_SIZE=8,
                CROP_POSITIONS=(("only", 0, 0),),
            ):
                metadata = _prepare_from_pngs(
                    "ShakeNDry",
                    source,
                    destination,
                )
            images = sorted(destination.glob("*.png"))
            self.assertEqual(len(images), 2)
            self.assertEqual(metadata["source_frame_count"], expected)
            self.assertEqual(
                metadata["selected_frame_indices"],
                [12, 37],
            )
            with Image.open(images[0]) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (8, 8))

    def test_prepared_dataset_validation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            (root / "_preparation_metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_name": PROTOCOL_NAME,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _validate_prepared_dataset(root)

    def test_parser_requires_source_root_and_license_is_explicit(self):
        parser = build_parser()
        args = parser.parse_args(["--source-root", "/tmp/datasets"])
        self.assertEqual(args.source_root, Path("/tmp/datasets"))
        self.assertFalse(args.accept_uvg_by_nc)
        accepted = parser.parse_args(
            ["--source-root", "/tmp/datasets", "--accept-uvg-by-nc"]
        )
        self.assertTrue(accepted.accept_uvg_by_nc)

    def test_external_runner_never_requests_a_shell(self):
        from atic import uvg_prep

        with mock.patch.object(uvg_prep.subprocess, "run") as run:
            uvg_prep._run(["tool", "arg with spaces"])
        run.assert_called_once_with(
            ["tool", "arg with spaces"],
            check=True,
            shell=False,
        )


if __name__ == "__main__":
    unittest.main()
