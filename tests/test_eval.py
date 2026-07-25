import os
import unittest
from unittest import mock

try:
    import torch

    from atic.eval import (
        _bits_from_likelihoods,
        _encoded_byte_breakdown,
        _format_optional_metric,
        eval_single,
    )

    TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight local hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    TORCH_IMPORT_ERROR = exc


class _FakeMetrics:
    def __init__(self, device):
        self.device = device

    def compute_all(self, x_hat, x, bpp=0.0):
        return {
            "BPP": float(bpp),
            "MSE": torch.mean((x_hat - x) ** 2).item(),
        }


class _FakeCodec:
    def to(self, device):
        return self

    def eval(self):
        return self

    def update(self, force=False):
        return True

    def __call__(self, batch):
        return {
            # This deliberately differs from decompress() so the test proves
            # actual-mode quality uses the decoded image.
            "x_hat": torch.zeros_like(batch),
            "likelihoods": {
                "y": torch.full(
                    (batch.size(0), 1, 1, 1),
                    0.5,
                    dtype=batch.dtype,
                    device=batch.device,
                )
            },
        }

    def compress(self, image, output_path=None):
        token = int(round(float(image.mean().item()) * 10))
        return {
            "bitstream": token,
            "num_bytes": token * 10,
            "payload_bytes": token,
            "header_bytes": token * 9,
            "y_bytes": token - 1,
            "z_bytes": 1,
        }

    def decompress(self, token):
        value = token / 10.0
        error = 1.0 if token == 3 else 0.0
        return torch.full((1, 3, 2, 2), value + error)


@unittest.skipIf(
    TORCH_IMPORT_ERROR is not None,
    f"PyTorch unavailable: {TORCH_IMPORT_ERROR}",
)
class EvaluationAggregationTests(unittest.TestCase):
    def test_unavailable_metric_is_not_presented_as_a_perfect_zero(self):
        self.assertEqual(
            _format_optional_metric({}, "LPIPS", 4),
            "unavailable",
        )
        self.assertEqual(
            _format_optional_metric({"LPIPS": 0.125}, "LPIPS", 4),
            "0.1250",
        )

    def test_estimated_bits_clamp_probability_at_one(self):
        bits = _bits_from_likelihoods(torch.tensor([2.0, 0.5]))
        self.assertAlmostEqual(bits, 1.0)
        self.assertGreaterEqual(bits, 0.0)

    def test_estimated_bits_reject_malformed_likelihoods(self):
        malformed = (
            torch.tensor([float("nan")]),
            torch.tensor([float("inf")]),
            torch.tensor([-0.1]),
        )
        for likelihood in malformed:
            with self.subTest(likelihood=likelihood):
                with self.assertRaises(ValueError):
                    _bits_from_likelihoods(likelihood)

    def test_byte_breakdown_rejects_inconsistent_codec_accounting(self):
        with self.assertRaisesRegex(ValueError, "y_bytes \\+ z_bytes"):
            _encoded_byte_breakdown(
                {
                    "num_bytes": 20,
                    "payload_bytes": 5,
                    "header_bytes": 15,
                    "y_bytes": 2,
                    "z_bytes": 2,
                }
            )

    def test_actual_rate_is_pixel_weighted_and_quality_is_per_image_decode(self):
        batches = [
            torch.stack(
                (
                    torch.full((3, 2, 2), 0.1),
                    torch.full((3, 2, 2), 0.2),
                )
            ),
            torch.full((1, 3, 2, 2), 0.3),
        ]

        with mock.patch("atic.eval.ATICMetrics", _FakeMetrics):
            result = eval_single(
                _FakeCodec(),
                batches,
                device="cpu",
                use_actual_bitstream=True,
            )

        self.assertEqual(result["num_images"], 3)
        self.assertEqual(result["num_pixels"], 12)
        self.assertEqual(result["bitstream_bytes"], 60)
        self.assertEqual(result["payload_bytes"], 6)
        self.assertEqual(result["y_bytes"], 3)
        self.assertEqual(result["z_bytes"], 3)
        self.assertEqual(result["header_bytes"], 54)
        self.assertAlmostEqual(result["BPP_actual"], 40.0)
        self.assertAlmostEqual(result["BPP_payload"], 4.0)
        self.assertAlmostEqual(result["y_bpp_actual"], 2.0)
        self.assertAlmostEqual(result["z_bpp_actual"], 2.0)
        self.assertAlmostEqual(result["header_bpp"], 36.0)
        self.assertAlmostEqual(result["z_fraction"], 0.05)
        self.assertAlmostEqual(result["BPP_estimated"], 0.25)
        self.assertAlmostEqual(
            result["BPP_payload_minus_estimated"],
            3.75,
        )
        self.assertAlmostEqual(
            result["BPP_actual_minus_estimated"],
            39.75,
        )
        self.assertAlmostEqual(
            result["BPP_actual"],
            result["y_bpp_actual"]
            + result["z_bpp_actual"]
            + result["header_bpp"],
        )
        # Errors are [0, 0, 1]. Per-image averaging is 1/3; the former
        # equal-per-batch averaging bug would have reported 1/2.
        self.assertAlmostEqual(result["MSE"], 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
