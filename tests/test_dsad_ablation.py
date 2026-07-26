import os
import unittest
from dataclasses import asdict
from unittest import mock

try:
    from ablation import (
        ABLATION_VARIANTS,
        CAUSAL_DSAD_VARIANTS,
        _dsad_settings_for_variant,
        _evaluation_plan,
        _validate_dataset_protocol,
        _validate_dsad_hyperparameters,
        _validate_study_hyperparameters,
        build_arg_parser,
    )

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - lightweight local hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"ATIC training dependencies unavailable: {IMPORT_ERROR}",
)
class DSADAblationConfigurationTests(unittest.TestCase):
    def test_causal_arms_have_identical_architecture(self):
        no_dsad, dsad = CAUSAL_DSAD_VARIANTS
        self.assertEqual(no_dsad, "Full_ATIC_NoDSAD")
        self.assertEqual(dsad, "Full_ATIC_DSAD")
        self.assertEqual(
            asdict(ABLATION_VARIANTS[no_dsad]),
            asdict(ABLATION_VARIANTS[dsad]),
        )

    def test_no_adaptive_quant_changes_only_the_gain_mechanism(self):
        full = asdict(ABLATION_VARIANTS["Full_ATIC_NoDSAD"])
        no_gain = asdict(ABLATION_VARIANTS["No_AdaptiveQuant"])
        differences = {
            key
            for key in full
            if full[key] != no_gain[key]
        }

        self.assertEqual(differences, {"use_adaptive_quant"})
        self.assertTrue(full["use_adaptive_quant"])
        self.assertFalse(no_gain["use_adaptive_quant"])

    def test_plain_swin_sanity_variant_retains_full_latent_geometry(self):
        full = asdict(ABLATION_VARIANTS["Full_ATIC_NoDSAD"])
        plain = asdict(ABLATION_VARIANTS["Plain_Swin_Hyperprior"])
        differences = {
            key
            for key in full
            if full[key] != plain[key]
        }

        self.assertEqual(
            differences,
            {"use_sag", "use_cbam", "use_adaptive_quant"},
        )
        self.assertTrue(plain["use_overlapping_patches"])
        self.assertTrue(plain["use_hyperprior"])
        self.assertFalse(plain["use_sag"])
        self.assertFalse(plain["use_cbam"])
        self.assertFalse(plain["use_adaptive_quant"])

    def test_no_latent_norm_variant_changes_only_terminal_encoder_norm(self):
        plain = asdict(ABLATION_VARIANTS["Plain_Swin_Hyperprior"])
        no_norm = asdict(
            ABLATION_VARIANTS["Plain_Swin_Hyperprior_NoLatentNorm"]
        )
        differences = {
            key
            for key in plain
            if plain[key] != no_norm[key]
        }

        self.assertEqual(differences, {"use_encoder_latent_norm"})
        self.assertTrue(plain["use_encoder_latent_norm"])
        self.assertFalse(no_norm["use_encoder_latent_norm"])

    def test_groupnorm_sag_variant_changes_only_sag_normalization(self):
        batch = asdict(ABLATION_VARIANTS["No_AdaptiveQuant"])
        group = asdict(
            ABLATION_VARIANTS["No_AdaptiveQuant_GroupNormSAG"]
        )
        differences = {
            key
            for key in batch
            if batch[key] != group[key]
        }

        self.assertEqual(differences, {"sag_normalization"})
        self.assertEqual(batch["sag_normalization"], "batch")
        self.assertEqual(group["sag_normalization"], "group")

    def test_only_dsad_arm_receives_nonzero_beta(self):
        no_dsad = _dsad_settings_for_variant(
            "Full_ATIC_NoDSAD",
            beta_max=0.05,
            warmup_fraction=0.2,
            ramp_fraction=0.1,
        )
        dsad = _dsad_settings_for_variant(
            "Full_ATIC_DSAD",
            beta_max=0.05,
            warmup_fraction=0.2,
            ramp_fraction=0.1,
        )

        self.assertEqual(no_dsad["beta_max"], 0.0)
        self.assertEqual(dsad["beta_max"], 0.05)
        self.assertEqual(
            no_dsad["warmup_fraction"],
            dsad["warmup_fraction"],
        )
        self.assertEqual(no_dsad["ramp_fraction"], dsad["ramp_fraction"])
        historical = _dsad_settings_for_variant(
            "Full_ATIC",
            beta_max=0.05,
            warmup_fraction=0.2,
            ramp_fraction=0.1,
        )
        self.assertEqual(historical["beta_max"], 0.0)
        self.assertEqual(
            historical["comparison_role"],
            "not_a_causal_dsad_arm",
        )

    def test_parser_defaults_to_controlled_pair_and_pilot_schedule(self):
        empty_env = {
            "ATIC_VARIANTS": "",
            "ATIC_DSAD_BETA_MAX": "",
            "ATIC_DSAD_WARMUP_FRACTION": "",
            "ATIC_DSAD_RAMP_FRACTION": "",
        }
        with mock.patch.dict(os.environ, empty_env, clear=True):
            args = build_arg_parser().parse_args([])

        self.assertEqual(
            args.variants,
            "Full_ATIC_NoDSAD,Full_ATIC_DSAD",
        )
        self.assertEqual(args.dsad_beta_max, 0.05)
        self.assertEqual(args.dsad_warmup_fraction, 0.20)
        self.assertEqual(args.dsad_ramp_fraction, 0.10)
        self.assertEqual(args.lr_schedule, "cosine")
        self.assertEqual(args.grad_clip_norm, 1.0)
        self.assertEqual(args.checkpoint_selection, "best_val_rd")
        self.assertIsNone(args.dataset_root)
        self.assertIsNone(args.frozen_split_dir)
        self.assertFalse(args.evaluate_test)

    def test_frozen_protocol_requires_a_complete_pair_and_test_is_opt_in(self):
        self.assertEqual(_evaluation_plan(False), ("val",))
        self.assertEqual(_evaluation_plan(True), ("val", "test"))
        self.assertFalse(_validate_dataset_protocol(None, None, False))
        self.assertTrue(
            _validate_dataset_protocol("/data", "/splits/v1", False)
        )
        self.assertTrue(
            _validate_dataset_protocol("/data", "/splits/v1", True)
        )
        for values in (
            ("/data", None, False),
            (None, "/splits/v1", False),
            (None, None, True),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _validate_dataset_protocol(*values)

        args = build_arg_parser().parse_args(
            [
                "--dataset-root",
                "/data",
                "--frozen-split-dir",
                "/splits/v1",
            ]
        )
        self.assertFalse(args.evaluate_test)
        with mock.patch.dict(
            os.environ,
            {"ATIC_EVALUATE_TEST": "true"},
            clear=False,
        ):
            self.assertFalse(
                build_arg_parser().parse_args([]).evaluate_test
            )
        self.assertTrue(
            build_arg_parser().parse_args(
                [
                    "--dataset-root",
                    "/data",
                    "--frozen-split-dir",
                    "/splits/v1",
                    "--evaluate-test",
                ]
            ).evaluate_test
        )

    def test_schedule_validation_rejects_invalid_values(self):
        invalid_values = (
            (-0.01, 0.2, 0.1),
            (0.05, -0.1, 0.1),
            (0.05, 0.2, 1.1),
            (0.05, 0.8, 0.3),
            (float("nan"), 0.2, 0.1),
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _validate_dsad_hyperparameters(*values)

    def test_study_validation_rejects_silent_noop_or_overwrite_inputs(self):
        valid = {
            "epochs": 2,
            "batch_size": 1,
            "height": 64,
            "width": 64,
            "val_every": 2,
            "num_workers": 0,
            "seeds": [42],
            "lambda_rates": [0.0067],
            "run_variants": [
                "Full_ATIC_NoDSAD",
                "Full_ATIC_DSAD",
            ],
        }
        _validate_study_hyperparameters(**valid)

        invalid_overrides = (
            {"epochs": 0},
            {"batch_size": 0},
            {"val_every": 1},
            {"num_workers": -1},
            {"seeds": []},
            {"seeds": [42, 42]},
            {"lambda_rates": []},
            {"lambda_rates": [float("nan")]},
            {"lambda_rates": [-0.1]},
            {"lambda_rates": [0.0067, 0.0067]},
            {"run_variants": []},
            {
                "run_variants": [
                    "Full_ATIC_DSAD",
                    "Full_ATIC_DSAD",
                ]
            },
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                candidate = dict(valid)
                candidate.update(override)
                with self.assertRaises(ValueError):
                    _validate_study_hyperparameters(**candidate)

    def test_cuda_request_never_silently_falls_back_to_cpu(self):
        from ablation import run_ablation_study

        with mock.patch("ablation.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "silent CPU fallback"):
                run_ablation_study(
                    device="cuda:0",
                    epochs=1,
                    batch_size=1,
                    height=64,
                    width=64,
                    val_every=2,
                    num_workers=0,
                    run_variants=["Full_ATIC_NoDSAD"],
                    lambda_rates=[0.0067],
                    seeds=[42],
                )


if __name__ == "__main__":
    unittest.main()
