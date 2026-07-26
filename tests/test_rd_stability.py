import json
import tempfile
import unittest
from pathlib import Path

from atic.rd_stability import DEFAULT_VARIANT, build_report, load_rows


class RDStabilityTests(unittest.TestCase):
    def _write_study(self, root: Path, points, *, unlocked=False) -> Path:
        study = root / "study"
        study.mkdir()
        (study / "study_config.json").write_text(
            json.dumps({"test_evaluation_enabled": unlocked}),
            encoding="utf-8",
        )
        rows = []
        for lambda_rd, bpp, psnr in points:
            run_dir = (
                study
                / "runs"
                / DEFAULT_VARIANT
                / f"lam_{lambda_rd:g}"
                / "seed_42"
            )
            run_dir.mkdir(parents=True)
            records = [
                {
                    "epoch": 1,
                    "val_rd_loss": 2.0,
                    "aux_loss": 10.0,
                },
                {
                    "epoch": 2,
                    "val_rd_loss": 1.0,
                    "aux_loss": 5.0,
                },
            ]
            (run_dir / "train_log.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            rows.append(
                {
                    "variant": DEFAULT_VARIANT,
                    "seed": 42,
                    "lambda_rd": lambda_rd,
                    "evaluation_split": "val",
                    "BPP_actual": bpp,
                    "PSNR": psnr,
                }
            )
        (study / "summary_metrics.json").write_text(
            json.dumps({"rows": rows}),
            encoding="utf-8",
        )
        return study

    def test_four_point_monotonic_validation_curve_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            study = self._write_study(
                Path(directory),
                [
                    (0.0018, 0.05, 11.0),
                    (0.0035, 0.10, 12.0),
                    (0.0067, 0.20, 13.0),
                    (0.0130, 0.30, 14.0),
                ],
            )
            rows = load_rows([study], variant=DEFAULT_VARIANT)
            report = build_report(
                rows,
                variant=DEFAULT_VARIANT,
                expected_lambdas=[0.0018, 0.0035, 0.0067, 0.013],
                expected_seeds=[42],
                tolerance=1e-8,
            )

        self.assertTrue(report["passed"])
        self.assertTrue(report["test_locked"])
        self.assertEqual(
            report["seeds"][0]["points"][0]["history"]["best_epoch_any"],
            2,
        )

    def test_psnr_reversal_fails_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            study = self._write_study(
                Path(directory),
                [
                    (0.0018, 0.05, 11.0),
                    (0.0035, 0.10, 13.0),
                    (0.0067, 0.20, 12.0),
                    (0.0130, 0.30, 14.0),
                ],
            )
            report = build_report(
                load_rows([study], variant=DEFAULT_VARIANT),
                variant=DEFAULT_VARIANT,
                expected_lambdas=None,
                expected_seeds=None,
                tolerance=1e-8,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["seeds"][0]["psnr_monotonic"])

    def test_unlocked_test_study_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            study = self._write_study(
                Path(directory),
                [(0.0018, 0.05, 11.0)],
                unlocked=True,
            )
            with self.assertRaisesRegex(RuntimeError, "unlocked test"):
                load_rows([study], variant=DEFAULT_VARIANT)


if __name__ == "__main__":
    unittest.main()
