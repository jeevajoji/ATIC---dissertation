import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atic.dsad_screen import (
    CONTROL_VARIANT,
    DSAD_VARIANT,
    ScreenJob,
    _csv_floats,
    _csv_gpus,
    _validate_args,
    assign_jobs,
    build_ablation_command,
    build_report,
    calibration_jobs,
    child_environment,
)


def _args(**overrides):
    values = {
        "dataset_root": "/data/images",
        "frozen_split_dir": "/data/splits",
        "physical_gpus": ["0", "1"],
        "betas": [0.5, 2.0, 8.0],
        "epochs": 12,
        "batch_size": 4,
        "num_workers": 4,
        "seed": 42,
        "lambda_rd": 0.0067,
        "height": 512,
        "width": 512,
        "output_root": "out",
        "required_branch": "final",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DSADScreenTests(unittest.TestCase):
    def test_default_jobs_are_three_paired_beta_comparisons(self):
        jobs = calibration_jobs([0.5, 2.0, 8.0])
        self.assertEqual(
            [job.beta for job in jobs],
            [0.5, 2.0, 8.0],
        )

    def test_assignments_keep_each_complete_pair_on_one_gpu(self):
        assignments = assign_jobs(
            calibration_jobs([0.5, 2.0, 8.0]),
            ["0", "1"],
        )
        self.assertEqual(
            [job.beta for job in assignments["0"]],
            [0.5, 8.0],
        )
        self.assertEqual(
            [job.beta for job in assignments["1"]],
            [2.0],
        )

    def test_command_is_validation_only_and_child_sees_one_gpu(self):
        job = ScreenJob(2.0)
        with tempfile.TemporaryDirectory() as directory:
            command = build_ablation_command(
                python_executable="python",
                repo_root=Path(directory),
                args=_args(),
                job=job,
                job_output_root=Path(directory) / "job",
            )
        self.assertNotIn("--evaluate-test", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(
            command[command.index("--variants") + 1],
            f"{CONTROL_VARIANT},{DSAD_VARIANT}",
        )
        self.assertEqual(
            command[command.index("--dsad-beta-max") + 1],
            "2",
        )
        environment = child_environment(
            {"KEEP": "yes"},
            physical_gpu="1",
            seed=42,
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(environment["PYTHONHASHSEED"], "42")
        self.assertEqual(environment["OMP_NUM_THREADS"], "1")
        self.assertEqual(environment["MKL_NUM_THREADS"], "1")
        self.assertEqual(environment["KEEP"], "yes")

    def test_beta_parser_rejects_zero_negative_and_duplicates(self):
        for value in (
            "",
            "0",
            "-1",
            "0.5,0.5",
            "nan",
            "inf",
            "hello",
        ):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _csv_floats(value)

    def test_gpu_parser_normalises_aliases_before_distinctness_check(self):
        self.assertEqual(_csv_gpus("00,1"), ["0", "1"])
        with self.assertRaises(argparse.ArgumentTypeError):
            _csv_gpus("0,00")

    def test_parent_gpu_mapping_must_match_requested_mapping(self):
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "2,3"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _validate_args(_args())

    def test_nonfinite_beta_and_lambda_are_rejected(self):
        for overrides in (
            {"betas": [float("nan")]},
            {"betas": [float("inf")]},
            {"lambda_rd": float("nan")},
            {"lambda_rd": float("inf")},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    _validate_args(_args(**overrides))

    def test_report_uses_actual_bpp_and_keeps_test_locked(self):
        base = {
            "variant": CONTROL_VARIANT,
            "beta": 0.0,
            "pair_beta": 2.0,
            "seed": 42,
            "lambda_rd": 0.0067,
            "BPP_actual": 0.5,
            "MSE": 0.03,
            "PSNR": 15.0,
            "SSIM": 0.5,
            "MS-SSIM": 0.5,
            "LPIPS": 0.7,
            "DISTS": 0.7,
            "rd_objective_actual": 13.57,
        }
        dsad = dict(base)
        dsad.update(
            {
                "variant": DSAD_VARIANT,
                "beta": 2.0,
                "pair_beta": 2.0,
                "BPP_actual": 0.49,
                "PSNR": 15.1,
                "rd_objective_actual": 13.50,
            }
        )
        report = build_report(
            [base, dsad],
            run_id="run",
            repository={"commit": "abc"},
            bundle={"bundle_id": "bundle"},
        )
        self.assertFalse(report["test_evaluated"])
        self.assertTrue(report["test_locked"])
        self.assertEqual(report["lowest_validation_rd_delta_beta"], 2.0)
        row = report["rows"][1]
        self.assertAlmostEqual(row["delta_BPP_actual"], -0.01)
        self.assertEqual(row["BPP_actual_direction"], "better")
        self.assertEqual(row["PSNR_direction"], "better")


if __name__ == "__main__":
    unittest.main()
