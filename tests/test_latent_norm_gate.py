import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atic.latent_norm_gate import (
    CONTROL_VARIANT,
    HIGH_LAMBDA,
    INTERVENTION_VARIANT,
    LOW_LAMBDA,
    _resolve_studies,
    _validate_study_contributions,
    build_report,
    main,
)


def _digest(provenance):
    return hashlib.sha256(
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _architecture(use_encoder_latent_norm):
    return {
        "token_dim": 128,
        "use_overlapping_patches": True,
        "use_sag": False,
        "use_cbam": False,
        "use_adaptive_quant": False,
        "use_hyperprior": True,
        "use_encoder_latent_norm": use_encoder_latent_norm,
    }


def _provenance(
    variant,
    *,
    use_encoder_latent_norm,
    initial_state,
    common_initial_state="common-initial",
):
    return {
        "git": {"commit": "abc123", "is_dirty": False},
        "software": {
            "python_version": "3.10",
            "torch_version": "2.6.0",
            "dependency_versions": {
                "compressai": "1.2.8",
                "numpy": "1.26.4",
                "pillow": "12.2.0",
                "timm": "1.0.27",
                "torchvision": "0.21.0",
            },
            "cuda_version": "12.4",
            "cudnn_version": 90100,
            "deterministic_algorithms_enabled": True,
            "gpu_name": "NVIDIA RTX A6000",
        },
        "data": {
            "data_protocol": "frozen_declared_sequence_groups_v1",
            "frozen_split_verified": True,
            "bundle_id": "bundle",
            "dataset_id": "dataset",
            "hashes": {"train": {"content_sha256": "train"}},
            "split_counts": {"train": 360, "val": 60, "test": 120},
        },
        "run": {
            "variant": variant,
            "architecture": _architecture(use_encoder_latent_norm),
            "initial_state_sha256": initial_state,
            "paired_common_initial_state_sha256": common_initial_state,
            "paired_common_excluded_names": [
                "encoder.norm.bias",
                "encoder.norm.weight",
            ],
            "paired_common_excluded_present_names": (
                ["encoder.norm.bias", "encoder.norm.weight"]
                if use_encoder_latent_norm
                else []
            ),
            "seed": 42,
            "epochs": 60,
            "batch_size": 4,
            "height": 512,
            "width": 512,
            "training_controls": {
                "learning_rate": 1e-4,
                "grad_clip_norm": 1.0,
                "checkpoint_selection": "best_val_rd",
                "checkpoint_selection_start_epoch": 1,
            },
            "dsad": {"beta_max": 0.0},
        },
    }


def _mse(psnr):
    return 10.0 ** (-psnr / 10.0)


def _rows(
    variant,
    *,
    use_encoder_latent_norm,
    initial_state,
    low_bpp,
    high_bpp,
    low_estimated,
    high_estimated,
    low_psnr,
    high_psnr,
):
    provenance = _provenance(
        variant,
        use_encoder_latent_norm=use_encoder_latent_norm,
        initial_state=initial_state,
    )
    digest = _digest(provenance)
    return [
        {
            "variant": variant,
            "seed": 42,
            "lambda_rd": lambda_rd,
            "BPP_actual": bpp,
            "BPP_estimated": estimated,
            "PSNR": psnr,
            "MSE": _mse(psnr),
            "provenance": copy.deepcopy(provenance),
            "provenance_sha256": digest,
        }
        for lambda_rd, bpp, estimated, psnr in (
            (LOW_LAMBDA, low_bpp, low_estimated, low_psnr),
            (HIGH_LAMBDA, high_bpp, high_estimated, high_psnr),
        )
    ]


def _valid_rows():
    control = _rows(
        CONTROL_VARIANT,
        use_encoder_latent_norm=True,
        initial_state="control-initial",
        low_bpp=0.05,
        high_bpp=0.07,
        low_estimated=0.048,
        high_estimated=0.068,
        low_psnr=12.0,
        high_psnr=12.5,
    )
    intervention = _rows(
        INTERVENTION_VARIANT,
        use_encoder_latent_norm=False,
        initial_state="intervention-initial",
        low_bpp=0.04,
        high_bpp=0.06,
        low_estimated=0.038,
        high_estimated=0.058,
        low_psnr=13.0,
        high_psnr=13.5,
    )
    return control, intervention


def _replace_intervention_provenance(rows, mutate):
    for row in rows:
        provenance = copy.deepcopy(row["provenance"])
        mutate(provenance)
        row["provenance"] = provenance
        row["provenance_sha256"] = _digest(provenance)


class LatentNormGateTests(unittest.TestCase):
    def test_pass_requires_rate_response_and_lower_rd_at_both_rates(self):
        control, intervention = _valid_rows()
        report = build_report(control, intervention)

        self.assertTrue(report["passed"])
        self.assertTrue(report["test_locked"])
        self.assertFalse(report["beauty_evaluated"])
        self.assertTrue(
            report["checks"]["intervention_rate_response"]
        )
        self.assertLess(
            report["actual_rd"]["intervention"]["low"]["actual_rd"],
            report["actual_rd"]["control"]["low"]["actual_rd"],
        )
        self.assertLess(
            report["actual_rd"]["intervention"]["high"]["actual_rd"],
            report["actual_rd"]["control"]["high"]["actual_rd"],
        )

    def test_intervention_rate_response_failure_fails_overall(self):
        control, intervention = _valid_rows()
        intervention[1]["PSNR"] = intervention[0]["PSNR"] + 0.1
        intervention[1]["MSE"] = _mse(intervention[1]["PSNR"])

        report = build_report(control, intervention)

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["checks"]["intervention_rate_response"]
        )

    def test_control_rate_response_is_reported_but_not_decisive(self):
        control, intervention = _valid_rows()
        control[1]["PSNR"] = control[0]["PSNR"] + 0.1
        control[1]["MSE"] = _mse(control[1]["PSNR"])

        report = build_report(control, intervention)

        self.assertFalse(report["rate_response"]["control"]["passed"])
        self.assertTrue(report["passed"])

    def test_rd_must_be_strictly_lower_at_every_lambda(self):
        control, intervention = _valid_rows()
        intervention[1]["MSE"] = control[1]["MSE"]
        intervention[1]["BPP_actual"] = control[1]["BPP_actual"]

        report = build_report(control, intervention)

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["checks"]["intervention_actual_rd_lower_high"]
        )

    def test_cross_variant_controls_must_match(self):
        control, intervention = _valid_rows()
        _replace_intervention_provenance(
            intervention,
            lambda provenance: provenance["git"].update(
                {"commit": "different"}
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "differs outside variant",
        ):
            build_report(control, intervention)

    def test_architecture_must_change_only_latent_norm(self):
        control, intervention = _valid_rows()
        _replace_intervention_provenance(
            intervention,
            lambda provenance: provenance["run"]["architecture"].update(
                {"token_dim": 64}
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "differ only"):
            build_report(control, intervention)

    def test_initial_state_hashes_must_be_distinct(self):
        control, intervention = _valid_rows()
        _replace_intervention_provenance(
            intervention,
            lambda provenance: provenance["run"].update(
                {"initial_state_sha256": "control-initial"}
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "initial-state hash"):
            build_report(control, intervention)

    def test_common_initial_state_hashes_must_match(self):
        control, intervention = _valid_rows()
        _replace_intervention_provenance(
            intervention,
            lambda provenance: provenance["run"].update(
                {
                    "paired_common_initial_state_sha256": (
                        "different-common-initial"
                    )
                }
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "common initial tensors"):
            build_report(control, intervention)

    def test_common_hash_exclusion_metadata_is_strict(self):
        control, intervention = _valid_rows()
        _replace_intervention_provenance(
            intervention,
            lambda provenance: provenance["run"].update(
                {"paired_common_excluded_names": ["encoder.norm.weight"]}
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "did not exclude exactly"):
            build_report(control, intervention)

        control, intervention = _valid_rows()
        _replace_intervention_provenance(
            intervention,
            lambda provenance: provenance["run"].update(
                {
                    "paired_common_excluded_present_names": [
                        "encoder.norm.bias",
                        "encoder.norm.weight",
                    ]
                }
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected encoder"):
            build_report(control, intervention)

    def test_duplicate_and_nonfinite_rows_are_rejected(self):
        control, intervention = _valid_rows()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            build_report(control + [copy.deepcopy(control[0])], intervention)

        control, intervention = _valid_rows()
        intervention[0]["MSE"] = math.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite MSE"):
            build_report(control, intervention)

    def test_cli_requires_two_distinct_variant_studies_and_saves_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = tuple(root / f"study_{index}" for index in range(2))
            for study in studies:
                study.mkdir()

            with self.assertRaisesRegex(RuntimeError, "exactly two"):
                _resolve_studies(studies[:1])
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                _resolve_studies((studies[0], studies[0]))

            control, intervention = _valid_rows()
            for row in control:
                row["study_dir"] = str(studies[0].resolve())
            for row in intervention:
                row["study_dir"] = str(studies[1].resolve())
            output = root / "report.json"
            with mock.patch(
                "atic.latent_norm_gate.load_rows",
                side_effect=(control, intervention),
            ) as loader:
                exit_code = main(
                    [
                        *(str(study) for study in studies),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(loader.call_count, 2)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertFalse(payload["beauty_evaluated"])

    def test_each_variant_pair_must_come_from_one_distinct_study(self):
        control, intervention = _valid_rows()
        report = build_report(control, intervention)
        first = Path("control").resolve()
        second = Path("intervention").resolve()

        report["rate_response"]["control"]["low"]["study_dir"] = str(first)
        report["rate_response"]["control"]["high"]["study_dir"] = str(second)
        for label in ("low", "high"):
            report["rate_response"]["intervention"][label][
                "study_dir"
            ] = str(second)
        with self.assertRaisesRegex(RuntimeError, "must share one study"):
            _validate_study_contributions(report, (first, second))

        report["rate_response"]["control"]["high"]["study_dir"] = str(first)
        for label in ("low", "high"):
            report["rate_response"]["intervention"][label][
                "study_dir"
            ] = str(first)
        with self.assertRaisesRegex(RuntimeError, "distinct"):
            _validate_study_contributions(report, (first, second))


if __name__ == "__main__":
    unittest.main()
