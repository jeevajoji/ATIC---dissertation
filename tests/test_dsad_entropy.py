import math
import os
import unittest
from unittest import mock

try:
    import torch

    from atic.blocks.entropy import CompressAIHyperpriorEntropy

    DSAD_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight hosts skip codec tests
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    DSAD_IMPORT_ERROR = exc


@unittest.skipIf(
    DSAD_IMPORT_ERROR is not None,
    f"PyTorch/CompressAI codec stack unavailable: {DSAD_IMPORT_ERROR}",
)
class DSADEntropyTests(unittest.TestCase):
    GAIN_MAX = 2.0

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_entropy(self):
        torch.manual_seed(73)
        entropy = CompressAIHyperpriorEntropy(
            latent_dim=4,
            hyper_dim=3,
            use_adaptive_quant=True,
            gain_max=self.GAIN_MAX,
        )
        entropy.eval()
        entropy.update(force=True)
        return entropy

    def assert_scale_normalised_gain(self, gain):
        self.assertTrue(torch.isfinite(gain).all())
        self.assertTrue(torch.all(gain > 0))
        gain_stats = gain.detach()
        self.assertGreaterEqual(
            float(gain_stats.min()),
            (1.0 / self.GAIN_MAX) - 1e-6,
        )
        self.assertLessEqual(
            float(gain_stats.max()),
            self.GAIN_MAX + 1e-6,
        )
        geometric_mean = torch.exp(
            torch.log(gain_stats).mean(dim=(-2, -1))
        )
        torch.testing.assert_close(
            geometric_mean,
            torch.ones_like(geometric_mean),
            rtol=0,
            atol=1e-6,
        )

    def test_student_gain_is_positive_bounded_and_scale_normalised(self):
        entropy = self.make_entropy()
        logits = torch.tensor(
            [
                [[[-100.0, -3.0, 0.0, 2.0], [100.0, 4.0, -1.0, 0.5]]],
                [[[5.0, -5.0, 1.0, -1.0], [0.25, -0.25, 20.0, -20.0]]],
            ],
            requires_grad=True,
        )

        gain = entropy._make_gain_map(logits)

        self.assertEqual(tuple(gain.shape), tuple(logits.shape))
        self.assert_scale_normalised_gain(gain)
        gain.square().mean().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

        disabled = CompressAIHyperpriorEntropy(
            latent_dim=4,
            hyper_dim=3,
            use_adaptive_quant=False,
            gain_max=self.GAIN_MAX,
        )
        self.assertTrue(
            torch.equal(
                disabled._make_gain_map(logits),
                torch.ones_like(logits),
            )
        )

    def test_teacher_projection_is_deterministic_detached_and_parameter_free(self):
        entropy = self.make_entropy()
        attention = torch.tensor(
            [[[[1.5, -0.5], [0.2, 0.8]]]],
            dtype=torch.float32,
            requires_grad=True,
        )

        teacher_first = entropy._make_teacher_map(attention, (5, 7))
        teacher_second = entropy._make_teacher_map(attention, (5, 7))

        self.assertIsNotNone(teacher_first)
        self.assertTrue(torch.equal(teacher_first, teacher_second))
        self.assertFalse(teacher_first.requires_grad)
        self.assertIsNone(teacher_first.grad_fn)
        self.assertFalse(hasattr(entropy, "attn_refine"))
        self.assertFalse(
            any("attn_refine" in name for name in entropy.state_dict())
        )
        self.assert_scale_normalised_gain(teacher_first)

    def test_operational_codec_never_invokes_teacher_and_decoder_skips_h_a(self):
        entropy = self.make_entropy()
        y = torch.randn(1, 4, 4, 4)

        with mock.patch.object(
            entropy,
            "_make_teacher_map",
            side_effect=AssertionError("teacher entered operational codec"),
        ) as teacher:
            encoded = entropy.compress(y)
            with mock.patch.object(
                entropy.h_a,
                "forward",
                side_effect=AssertionError("decoder invoked hyper-analysis"),
            ) as hyper_analysis:
                decoded, aux = entropy.decompress(
                    y_strings=encoded["y_strings"],
                    z_strings=encoded["z_strings"],
                    y_shape=encoded["y_shape"],
                    z_shape=encoded["z_shape"],
                )

        teacher.assert_not_called()
        hyper_analysis.assert_not_called()
        self.assertEqual(tuple(decoded.shape), tuple(y.shape))
        self.assert_scale_normalised_gain(aux["gain_map"])

    def test_gain_bound_is_persisted_in_checkpoint(self):
        source = CompressAIHyperpriorEntropy(
            latent_dim=4,
            hyper_dim=3,
            gain_max=3.0,
        )
        receiver = CompressAIHyperpriorEntropy(
            latent_dim=4,
            hyper_dim=3,
            gain_max=2.0,
        )

        receiver.load_state_dict(source.state_dict(), strict=True)

        self.assertAlmostEqual(receiver.gain_max, 3.0, places=6)
        self.assertTrue(
            torch.equal(
                receiver._log_gain_max,
                source._log_gain_max,
            )
        )

    def test_gain_max_must_define_a_nontrivial_finite_bound(self):
        for invalid in (1.0, 0.5, math.inf, math.nan):
            with self.subTest(gain_max=invalid):
                with self.assertRaisesRegex(ValueError, "gain_max"):
                    CompressAIHyperpriorEntropy(
                        latent_dim=4,
                        hyper_dim=3,
                        gain_max=invalid,
                    )


if __name__ == "__main__":
    unittest.main()
