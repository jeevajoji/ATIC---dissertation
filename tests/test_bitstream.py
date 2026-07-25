import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from array import array
from pathlib import Path
from unittest import mock

from atic.bitstream import (
    HEADER_BYTES,
    MAGIC,
    ATICBitstreamError,
    bpp_from_num_bytes,
    normalise_model_hash,
    pack_atic,
    read_atic_file,
    unpack_atic,
    write_atic_bytes,
)


class ATICBitstreamTests(unittest.TestCase):
    def make_bitstream(self):
        return pack_atic(
            z_string=b"hyperlatent-rans",
            y_string=b"main-latent-rans-payload",
            original_width=512,
            original_height=384,
            coded_width=512,
            coded_height=384,
            y_width=8,
            y_height=6,
            z_width=2,
            z_height=2,
            y_channels=1024,
            model_hash="unit-test-checkpoint",
            flags=0b1111,
            quality_id=7,
        )

    def test_pack_is_deterministic_and_has_fixed_header(self):
        first = self.make_bitstream()
        second = self.make_bitstream()
        self.assertEqual(first, second)
        self.assertEqual(
            len(first),
            HEADER_BYTES
            + len(b"hyperlatent-rans")
            + len(b"main-latent-rans-payload"),
        )

    def test_header_has_independent_golden_offsets(self):
        encoded = self.make_bitstream()
        self.assertEqual(encoded[0:8], MAGIC)
        self.assertEqual(struct.unpack_from(">H", encoded, 8)[0], 1)
        self.assertEqual(struct.unpack_from(">H", encoded, 10)[0], 0)
        self.assertEqual(struct.unpack_from(">I", encoded, 12)[0], 128)
        self.assertEqual(struct.unpack_from(">I", encoded, 16)[0], 0b1111)
        self.assertEqual(
            struct.unpack_from(">8I", encoded, 20),
            (512, 384, 512, 384, 8, 6, 2, 2),
        )
        self.assertEqual(struct.unpack_from(">H", encoded, 52)[0], 1024)
        self.assertEqual(encoded[54:58], bytes((8, 1, 1, 1)))
        self.assertEqual(struct.unpack_from(">H", encoded, 58)[0], 7)
        self.assertEqual(
            struct.unpack_from(">QQ", encoded, 60),
            (len(b"hyperlatent-rans"), len(b"main-latent-rans-payload")),
        )
        self.assertEqual(
            encoded[76:108],
            normalise_model_hash("unit-test-checkpoint"),
        )
        self.assertEqual(
            struct.unpack_from(">I", encoded, 108)[0],
            zlib.crc32(b"hyperlatent-rans") & 0xFFFFFFFF,
        )
        self.assertEqual(
            struct.unpack_from(">I", encoded, 112)[0],
            zlib.crc32(b"main-latent-rans-payload") & 0xFFFFFFFF,
        )
        self.assertNotEqual(struct.unpack_from(">I", encoded, 116)[0], 0)
        self.assertEqual(encoded[120:128], b"\0" * 8)
        self.assertEqual(encoded[128:], b"hyperlatent-ransmain-latent-rans-payload")

    def test_unpack_preserves_fields_and_payload_order(self):
        encoded = self.make_bitstream()
        decoded = unpack_atic(encoded)
        self.assertEqual(decoded.original_width, 512)
        self.assertEqual(decoded.original_height, 384)
        self.assertEqual(decoded.coded_width, 512)
        self.assertEqual(decoded.coded_height, 384)
        self.assertEqual((decoded.y_height, decoded.y_width), (6, 8))
        self.assertEqual((decoded.z_height, decoded.z_width), (2, 2))
        self.assertEqual(decoded.y_channels, 1024)
        self.assertEqual(decoded.flags, 0b1111)
        self.assertEqual(decoded.quality_id, 7)
        self.assertEqual(decoded.z_string, b"hyperlatent-rans")
        self.assertEqual(decoded.y_string, b"main-latent-rans-payload")
        self.assertEqual(
            decoded.model_hash,
            normalise_model_hash("unit-test-checkpoint"),
        )

    def test_actual_bpp_includes_complete_file(self):
        encoded = self.make_bitstream()
        expected = len(encoded) * 8 / (384 * 512)
        self.assertEqual(
            bpp_from_num_bytes(len(encoded), 384, 512),
            expected,
        )

    def test_file_round_trip(self):
        encoded = self.make_bitstream()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "image.atic"
            written = write_atic_bytes(path, encoded)
            self.assertEqual(written, len(encoded))
            self.assertEqual(path.read_bytes(), encoded)
            self.assertEqual(read_atic_file(path).y_string, b"main-latent-rans-payload")

    def test_fresh_process_inspection_needs_no_torch(self):
        encoded = self.make_bitstream()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.atic"
            write_atic_bytes(path, encoded)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atic.codec_cli",
                    "inspect",
                    "--input",
                    str(path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["format"], "ATIC 1.0")
            self.assertEqual(report["header_bytes"], HEADER_BYTES)
            self.assertEqual(report["num_bytes"], len(encoded))

    def test_rejects_header_corruption(self):
        encoded = bytearray(self.make_bitstream())
        encoded[40] ^= 0x01
        with self.assertRaisesRegex(ATICBitstreamError, "header checksum"):
            unpack_atic(encoded)

    def test_rejects_z_corruption(self):
        encoded = bytearray(self.make_bitstream())
        encoded[HEADER_BYTES] ^= 0x01
        with self.assertRaisesRegex(ATICBitstreamError, "z payload checksum"):
            unpack_atic(encoded)

    def test_rejects_y_corruption(self):
        encoded = bytearray(self.make_bitstream())
        encoded[-1] ^= 0x01
        with self.assertRaisesRegex(ATICBitstreamError, "y payload checksum"):
            unpack_atic(encoded)

    def test_rejects_truncation_and_trailing_bytes(self):
        encoded = self.make_bitstream()
        for candidate in (encoded[:-1], encoded + b"x"):
            with self.subTest(length=len(candidate)):
                with self.assertRaisesRegex(ATICBitstreamError, "size mismatch"):
                    unpack_atic(candidate)

    def test_rejects_bad_magic_and_version(self):
        encoded = bytearray(self.make_bitstream())
        encoded[0] ^= 0x01
        with self.assertRaisesRegex(ATICBitstreamError, "magic"):
            unpack_atic(encoded)

        encoded = bytearray(self.make_bitstream())
        encoded[9] = 2
        with self.assertRaisesRegex(ATICBitstreamError, "version"):
            unpack_atic(encoded)

    def test_rejects_invalid_pack_arguments(self):
        common = dict(
            z_string=b"z",
            y_string=b"y",
            original_width=16,
            original_height=16,
            coded_width=16,
            coded_height=16,
            y_width=1,
            y_height=1,
            z_width=1,
            z_height=1,
            y_channels=1,
            model_hash="model",
        )
        for update in (
            {"z_string": b""},
            {"y_string": b""},
            {"original_width": 0},
            {"original_width": 1.5},
            {"y_channels": 0},
            {"bit_depth": 0},
            {"flags": True},
            {"flags": 0x10},
            {"original_width": 32},
        ):
            with self.subTest(update=update):
                arguments = {**common, **update}
                with self.assertRaises(ValueError):
                    pack_atic(**arguments)

    def test_binary_model_hash_must_be_sha256_length(self):
        with self.assertRaises(ValueError):
            normalise_model_hash(b"too short")

    def test_stream_inputs_must_be_bytes_like(self):
        with self.assertRaises(TypeError):
            unpack_atic(128)
        with self.assertRaises(TypeError):
            write_atic_bytes("unused.atic", 128)

    def test_typed_memoryview_limit_uses_byte_count_not_element_count(self):
        typed_view = memoryview(array("Q", (0, 0, 0)))
        self.assertEqual(len(typed_view), 3)
        self.assertEqual(typed_view.nbytes, 24)
        with mock.patch("atic.bitstream.MAX_ATIC_FILE_BYTES", 16):
            with self.assertRaisesRegex(ATICBitstreamError, "safety limit"):
                unpack_atic(typed_view)

    def test_bpp_rejects_non_finite_byte_counts(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    bpp_from_num_bytes(invalid, 16, 16)


if __name__ == "__main__":
    unittest.main()
