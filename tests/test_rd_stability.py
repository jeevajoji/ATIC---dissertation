import json
import tempfile
import unittest
from pathlib import Path

from atic.rd_stability import (
    DEFAULT_VARIANT,
    _history_audit,
    _load_run_provenance,
    build_report,
    load_rows,
)


class RDStabilityTests(unittest.TestCase):
    def _write_study(
        self,
        root: Path,
        points,
        *,
        unlocked=False,
        selected_epoch=None,
        selected_loss=None,
    ) -> Path:
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
            row = {
                "variant": DEFAULT_VARIANT,
                "seed": 42,
                "lambda_rd": lambda_rd,
                "evaluation_split": "val",
                "BPP_actual": bpp,
                "PSNR": psnr,
            }
            if selected_epoch is not None:
                row["selected_epoch"] = selected_epoch
            if selected_loss is not None:
                row["selected_val_rd_loss"] = selected_loss
            rows.append(row)
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

    def test_history_distinguishes_any_best_from_selected_checkpoint(self):
        audit = _history_audit(
            [
                {
                    "epoch": 1,
                    "val_rd_loss": 0.5,
                    "checkpoint_selection_eligible": False,
                },
                {
                    "epoch": 2,
                    "val_rd_loss": 1.0,
                    "val_total_bpp": 0.2,
                    "val_mse_loss": 0.01,
                    "checkpoint_selection_eligible": True,
                },
                {
                    "epoch": 3,
                    "val_rd_loss": 1.5,
                    "checkpoint_selection_eligible": True,
                },
            ],
            selected_epoch=2,
        )

        self.assertEqual(audit["best_epoch_any"], 1)
        self.assertEqual(audit["best_epoch_eligible"], 2)
        self.assertEqual(audit["selected_epoch"], 2)
        self.assertAlmostEqual(audit["selected_val_total_bpp"], 0.2)
        self.assertAlmostEqual(audit["selected_val_psnr_from_mse"], 20.0)
        self.assertAlmostEqual(audit["final_over_selected_percent"], 50.0)

    def test_history_rejects_ineligible_or_nonbest_selection(self):
        records = [
            {
                "epoch": 1,
                "val_rd_loss": 0.5,
                "checkpoint_selection_eligible": False,
            },
            {
                "epoch": 2,
                "val_rd_loss": 1.0,
                "checkpoint_selection_eligible": True,
            },
            {
                "epoch": 3,
                "val_rd_loss": 1.5,
                "checkpoint_selection_eligible": True,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "ineligible"):
            _history_audit(records, selected_epoch=1)
        with self.assertRaisesRegex(RuntimeError, "minimum-RD eligible"):
            _history_audit(records, selected_epoch=3)

    def test_summary_selected_loss_must_match_history(self):
        with tempfile.TemporaryDirectory() as directory:
            study = self._write_study(
                Path(directory),
                [(0.0018, 0.05, 11.0)],
                selected_epoch=2,
                selected_loss=999.0,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "does not match training history",
            ):
                load_rows([study], variant=DEFAULT_VARIANT)

    def test_rate_provenance_requires_clean_matching_artifacts(self):
        architecture = {"use_sag": False}
        study_config = {
            "environment": {
                "git": {"commit": "abc123", "is_dirty": False},
                "python_version": "3.10",
                "torch_version": "2.6.0",
                "dependency_versions": {
                    "compressai": "1.2.8",
                    "numpy": "1.26.4",
                    "pillow": "12.2.0",
                    "timm": "0.6.13",
                    "torchvision": "0.21.0",
                },
            },
            "frozen_split_verified": True,
            "evaluation_split": "val",
            "data_protocol": "frozen_declared_sequence_groups_v1",
            "epochs": 60,
            "batch_size": 4,
            "training_controls": {
                "grad_clip_norm": 1.0,
                "checkpoint_selection": "best_val_rd",
                "checkpoint_selection_start_epoch": 1,
            },
            "variants": {
                DEFAULT_VARIANT: {
                    "architecture": architecture,
                    "dsad": {"beta_max": 0.0},
                },
            },
            "manifests": {
                "bundle_id": "bundle",
                "dataset_id": "dataset",
                "hashes": {"train": {"content_sha256": "train"}},
            },
            "split_counts": {"train": 360, "val": 60, "test": 120},
        }
        source = {
            "seed": 42,
            "lambda_rd": 0.0018,
            "initial_state_sha256": "initial",
            "frozen_bundle_id": "bundle",
            "data_protocol": "frozen_declared_sequence_groups_v1",
            "checkpoint_selection": "best_val_rd",
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "variant": DEFAULT_VARIANT,
                        "seed": 42,
                        "lambda_rd": 0.0018,
                        "evaluation_split": "val",
                        "data_protocol": "frozen_declared_sequence_groups_v1",
                        "frozen_bundle_id": "bundle",
                        "architecture": architecture,
                        "epochs": 60,
                        "batch_size": 4,
                        "height": 512,
                        "width": 512,
                        "training_controls": {
                            "grad_clip_norm": 1.0,
                            "checkpoint_selection": "best_val_rd",
                            "checkpoint_selection_start_epoch": 1,
                        },
                        "dsad": {"beta_max": 0.0},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "initial_state.json").write_text(
                json.dumps(
                    {
                        "sha256": "initial",
                        "seed": 42,
                        "variant": DEFAULT_VARIANT,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "environment.json").write_text(
                json.dumps(study_config["environment"]),
                encoding="utf-8",
            )

            provenance, digest = _load_run_provenance(
                study_config=study_config,
                run_dir=run_dir,
                source=source,
                variant=DEFAULT_VARIANT,
            )
            self.assertEqual(provenance["git"]["commit"], "abc123")
            self.assertEqual(len(digest), 64)

            study_config["environment"]["git"]["is_dirty"] = True
            (run_dir / "environment.json").write_text(
                json.dumps(study_config["environment"]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "clean Git commit"):
                _load_run_provenance(
                    study_config=study_config,
                    run_dir=run_dir,
                    source=source,
                    variant=DEFAULT_VARIANT,
                )

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
