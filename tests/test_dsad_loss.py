import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import torch

    from atic.losses import ATICLoss

    LOSS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    LOSS_IMPORT_ERROR = exc

try:
    from atic.train import (
        LOSS_LOG_KEYS,
        TRAIN_GRADIENT_LOG_KEYS,
        _clip_optimizer_gradients,
        configure_optimizers,
        dsad_beta_for_epoch,
        train_loop,
    )

    SCHEDULE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    SCHEDULE_IMPORT_ERROR = exc


def _base_output(x_hat, *, gain_map=None, teacher_map=None):
    output = {
        "x_hat": x_hat,
        "likelihoods": {
            "y": torch.full((1, 2, 1, 1), 0.5, dtype=x_hat.dtype),
            "z": torch.full((1, 1, 1, 1), 0.5, dtype=x_hat.dtype),
        },
    }
    if gain_map is not None:
        output["gain_map"] = gain_map
    if teacher_map is not None:
        output["teacher_map"] = teacher_map
    return output


@unittest.skipIf(
    LOSS_IMPORT_ERROR is not None,
    f"PyTorch loss stack unavailable: {LOSS_IMPORT_ERROR}",
)
class DSADLossTests(unittest.TestCase):
    def test_standard_rd_objective_and_split_bpp(self):
        target = torch.zeros(1, 3, 2, 2)
        x_hat = torch.full_like(target, 0.5)
        criterion = ATICLoss(lambda_rd=0.01)

        result = criterion(_base_output(x_hat), target)

        self.assertAlmostEqual(result["y_bpp"].item(), 0.5)
        self.assertAlmostEqual(result["z_bpp"].item(), 0.25)
        self.assertAlmostEqual(result["total_bpp"].item(), 0.75)
        expected = 0.75 + 0.01 * (255.0**2) * 0.25
        self.assertAlmostEqual(result["loss"].item(), expected, places=5)
        self.assertTrue(torch.equal(result["loss"], result["rd_loss"]))

    def test_beta_zero_is_exact_base_loss_and_needs_no_teacher(self):
        target = torch.zeros(1, 3, 2, 2)
        x_hat = torch.full_like(target, 0.25, requires_grad=True)
        criterion = ATICLoss(lambda_rd=0.01, dsad_beta=0.0)

        result = criterion(_base_output(x_hat), target)

        self.assertTrue(torch.equal(result["loss"], result["rd_loss"]))
        self.assertEqual(result["weighted_dsad_loss"].item(), 0.0)
        self.assertEqual(result["dsad_loss"].item(), 0.0)
        result["loss"].backward()
        self.assertIsNotNone(x_hat.grad)

    def test_dsad_detaches_teacher_but_updates_student(self):
        target = torch.zeros(1, 3, 2, 2)
        x_hat = torch.zeros_like(target, requires_grad=True)
        student = torch.tensor(
            [[[[0.8, 1.1], [1.4, 0.9]]]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[[[1.4, 0.7], [0.9, 1.2]]]],
            requires_grad=True,
        )
        criterion = ATICLoss(
            lambda_rd=0.0,
            dsad_beta=0.5,
            dsad_loss_type="smooth_l1",
        )

        result = criterion(
            _base_output(
                x_hat,
                gain_map=student,
                teacher_map=teacher,
            ),
            target,
        )
        result["loss"].backward()

        self.assertIsNotNone(student.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0.0)
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(
            result["weighted_dsad_loss"].item(),
            0.5 * result["dsad_loss"].item(),
        )

    def test_centered_log_dsad_is_invariant_to_global_gain_scale(self):
        target = torch.zeros(1, 3, 2, 2)
        x_hat = torch.zeros_like(target)
        student = torch.tensor([[[[0.8, 1.1], [1.4, 0.9]]]])
        teacher = torch.tensor([[[[1.4, 0.7], [0.9, 1.2]]]])
        criterion = ATICLoss(lambda_rd=0.0, dsad_beta=1.0)

        original = criterion(
            _base_output(
                x_hat,
                gain_map=student,
                teacher_map=teacher,
            ),
            target,
        )
        rescaled = criterion(
            _base_output(
                x_hat,
                gain_map=student * 3.0,
                teacher_map=teacher * 0.25,
            ),
            target,
        )

        torch.testing.assert_close(
            original["dsad_loss"],
            rescaled["dsad_loss"],
            rtol=0,
            atol=1e-7,
        )

    def test_non_positive_gain_is_rejected(self):
        target = torch.zeros(1, 3, 2, 2)
        x_hat = torch.zeros_like(target)
        student = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
        teacher = torch.ones_like(student)

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            ATICLoss(dsad_beta=1.0)(
                _base_output(
                    x_hat,
                    gain_map=student,
                    teacher_map=teacher,
                ),
                target,
            )

    def test_non_finite_loss_weights_are_rejected(self):
        for kwargs in (
            {"lambda_rd": float("nan")},
            {"lambda_ssim": float("inf")},
            {"lambda_lpips": float("nan")},
            {"dsad_beta": float("inf")},
            {"gain_eps": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "finite"):
                    ATICLoss(**kwargs)


@unittest.skipIf(
    SCHEDULE_IMPORT_ERROR is not None,
    f"PyTorch training stack unavailable: {SCHEDULE_IMPORT_ERROR}",
)
class DSADScheduleTests(unittest.TestCase):
    def test_twenty_percent_warmup_then_ten_percent_ramp(self):
        schedule = [
            dsad_beta_for_epoch(epoch, 20, 0.1)
            for epoch in range(20)
        ]

        self.assertEqual(schedule[:4], [0.0] * 4)
        self.assertAlmostEqual(schedule[4], 0.05)
        self.assertAlmostEqual(schedule[5], 0.1)
        self.assertEqual(schedule[6:], [0.1] * 14)

    def test_one_epoch_pilot_exercises_dsad(self):
        self.assertEqual(dsad_beta_for_epoch(0, 1, 0.05), 0.05)

    def test_two_epoch_pilot_has_one_warmup_epoch(self):
        schedule = [
            dsad_beta_for_epoch(epoch, 2, 0.05)
            for epoch in range(2)
        ]

        self.assertEqual(schedule, [0.0, 0.05])

    def test_short_schedule_reaches_beta_max(self):
        schedule = [
            dsad_beta_for_epoch(
                epoch,
                3,
                0.3,
                warmup_fraction=0.5,
                ramp_fraction=0.5,
            )
            for epoch in range(3)
        ]

        self.assertEqual(schedule[0], 0.0)
        self.assertGreater(schedule[1], 0.0)
        self.assertLess(schedule[1], 0.3)
        self.assertEqual(schedule[2], 0.3)

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            dsad_beta_for_epoch(0, 10, 0.1, 0.8, 0.3)

    def test_main_clipping_excludes_large_auxiliary_quantile_gradient(self):
        class SplitOptimizerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = torch.nn.Parameter(torch.tensor(1.0))
                self.entropy = torch.nn.Module()
                self.entropy.register_parameter(
                    "quantiles",
                    torch.nn.Parameter(torch.tensor(1.0)),
                )

        model = SplitOptimizerModel()
        optimizer, aux_optimizer = configure_optimizers(model)
        self.assertIsNotNone(aux_optimizer)

        model.main.grad = torch.tensor(3.0)
        model.entropy.quantiles.grad = torch.tensor(4000.0)
        observed_norm = _clip_optimizer_gradients(
            optimizer,
            max_norm=10.0,
        )

        self.assertAlmostEqual(observed_norm, 3.0)
        self.assertAlmostEqual(float(model.main.grad), 3.0)
        self.assertAlmostEqual(
            float(model.entropy.quantiles.grad),
            4000.0,
        )

        model.main.grad = torch.tensor(30.0)
        observed_norm = _clip_optimizer_gradients(
            optimizer,
            max_norm=10.0,
        )
        self.assertAlmostEqual(observed_norm, 30.0)
        self.assertAlmostEqual(float(model.main.grad), 10.0, places=5)
        self.assertAlmostEqual(
            float(model.entropy.quantiles.grad),
            4000.0,
        )

    def test_train_loop_clears_auxiliary_gradients_and_logs_norms(self):
        class TinyAuxModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = torch.nn.Parameter(torch.tensor(0.1))
                self.entropy = torch.nn.Module()
                self.entropy.register_parameter(
                    "quantiles",
                    torch.nn.Parameter(torch.tensor(1.0)),
                )

            def forward(self, batch):
                return {
                    "x_hat": torch.sigmoid(self.main) * torch.ones_like(batch),
                    "likelihoods": {
                        "y": batch.new_full((1, 1, 1, 1), 0.5),
                        "z": batch.new_full((1, 1, 1, 1), 0.5),
                    },
                }

            def aux_loss(self):
                return 10000.0 * self.entropy.quantiles.square()

        batch = torch.zeros(1, 3, 2, 2)
        model = TinyAuxModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_loop(
                model,
                "tiny_aux",
                [batch, batch],
                epochs=1,
                device="cpu",
                lambda_rd=0.01,
                checkpoint_path=str(Path(temp_dir) / "model.pth"),
                save_reconstruction_each_epoch=False,
            )

        record = result["history"][0]
        for key in TRAIN_GRADIENT_LOG_KEYS:
            self.assertIn(key, record)
        self.assertTrue(
            all(
                bool(torch.isfinite(torch.tensor(record[key])))
                for key in TRAIN_GRADIENT_LOG_KEYS
            )
        )
        self.assertGreater(record["aux_grad_norm"], 1000.0)
        self.assertGreater(record["main_grad_clip_fraction"], 0.0)
        self.assertIsNone(model.entropy.quantiles.grad)

    def test_train_and_validation_log_every_loss_component(self):
        class TinyDSADModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, batch):
                x_hat = torch.sigmoid(self.logit) * torch.ones_like(batch)
                pattern = batch.new_tensor(
                    [[[[1.0, -1.0], [0.5, -0.5]]]]
                )
                gain_map = torch.exp(self.logit * pattern)
                teacher_map = torch.exp(0.2 * pattern)
                return {
                    "x_hat": x_hat,
                    "likelihoods": {
                        "y": batch.new_full((1, 1, 1, 1), 0.5),
                        "z": batch.new_full((1, 1, 1, 1), 0.5),
                    },
                    "gain_map": gain_map,
                    "teacher_map": teacher_map,
                }

        batch = torch.zeros(1, 3, 2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_path = temp_path / "train.jsonl"
            result = train_loop(
                TinyDSADModel(),
                "tiny_dsad",
                [batch],
                val_loader=[batch],
                epochs=1,
                device="cpu",
                lambda_rd=0.01,
                dsad_beta_max=0.05,
                checkpoint_path=str(temp_path / "model.pth"),
                train_log_path=str(log_path),
                save_reconstruction_each_epoch=False,
            )

            self.assertEqual(len(result["history"]), 1)
            record = result["history"][0]
            for key in LOSS_LOG_KEYS:
                self.assertIn(key, record)
                self.assertIn(f"val_{key}", record)
            for key in TRAIN_GRADIENT_LOG_KEYS:
                self.assertIn(key, record)
            self.assertAlmostEqual(record["beta"], 0.05)
            self.assertAlmostEqual(record["val_beta"], 0.05)

            logged = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(logged, record)

    def test_cosine_schedule_restores_best_validation_rd_checkpoint(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.2))

            def forward(self, batch):
                return {
                    "x_hat": self.value * torch.ones_like(batch),
                    "likelihoods": {
                        "y": batch.new_full((1, 1, 1, 1), 0.5),
                        "z": batch.new_full((1, 1, 1, 1), 0.5),
                    },
                }

        observed_values = []

        def validation_result(*, model, **_kwargs):
            observed_values.append(float(model.value.detach()))
            score = 1.0 if len(observed_values) == 1 else 2.0
            result = {
                f"val_{key}": 0.0
                for key in LOSS_LOG_KEYS
            }
            result.update(
                {
                    "val_loss": score,
                    "val_rd_loss": score,
                    "val_total_bpp": 0.1,
                    "val_mse_loss": 0.01,
                    "val_dsad_loss": 0.0,
                }
            )
            return result

        batch = torch.zeros(1, 3, 2, 2)
        model = TinyModel()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "atic.train.validate_loop",
            side_effect=validation_result,
        ):
            temp_path = Path(temp_dir)
            result = train_loop(
                model,
                "tiny_best",
                [batch],
                val_loader=[batch],
                epochs=2,
                device="cpu",
                lambda_rd=0.01,
                learning_rate=0.1,
                aux_learning_rate=0.01,
                lr_schedule="cosine",
                min_learning_rate=0.01,
                min_aux_learning_rate=0.001,
                checkpoint_selection="best_val_rd",
                checkpoint_selection_start_epoch=1,
                checkpoint_path=str(temp_path / "model.pth"),
                save_reconstruction_each_epoch=False,
            )

            self.assertEqual(result["selected_epoch"], 1)
            self.assertAlmostEqual(result["selected_val_rd_loss"], 1.0)
            self.assertAlmostEqual(
                float(model.value.detach()),
                observed_values[0],
            )
            self.assertAlmostEqual(
                result["history"][0]["learning_rate"],
                0.1,
            )
            self.assertAlmostEqual(
                result["history"][1]["learning_rate"],
                0.055,
            )
            selection = json.loads(
                (temp_path / "checkpoint_selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selection["strategy"], "best_val_rd")
            self.assertEqual(selection["selected_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
