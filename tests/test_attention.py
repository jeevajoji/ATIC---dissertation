import os
import unittest

try:
    import torch

    from atic.blocks.attention import SpatialAttentionGate

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"PyTorch attention stack unavailable: {IMPORT_ERROR}",
)
class SpatialAttentionNormalizationTests(unittest.TestCase):
    def test_batch_mode_preserves_archived_batchnorm_architecture(self):
        module = SpatialAttentionGate(16, normalization="batch")

        self.assertIsInstance(module.bn1, torch.nn.BatchNorm2d)
        self.assertIsInstance(module.bn2, torch.nn.BatchNorm2d)

    def test_group_mode_has_no_batchnorm_or_running_statistics(self):
        module = SpatialAttentionGate(16, normalization="group")

        self.assertIsInstance(module.bn1, torch.nn.GroupNorm)
        self.assertIsInstance(module.bn2, torch.nn.GroupNorm)
        self.assertFalse(
            any(
                isinstance(child, torch.nn.BatchNorm2d)
                for child in module.modules()
            )
        )

    def test_group_mode_is_identical_in_train_and_eval(self):
        torch.manual_seed(42)
        module = SpatialAttentionGate(16, normalization="group")
        sample = torch.randn(2, 16, 8, 8)

        module.train()
        train_gated, train_attention = module(sample)
        module.eval()
        eval_gated, eval_attention = module(sample)

        torch.testing.assert_close(
            train_gated,
            eval_gated,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            train_attention,
            eval_attention,
            rtol=0,
            atol=0,
        )

    def test_unknown_normalization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalization"):
            SpatialAttentionGate(16, normalization="instance")


if __name__ == "__main__":
    unittest.main()
