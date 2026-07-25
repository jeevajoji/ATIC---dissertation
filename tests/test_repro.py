import os
import unittest
from unittest import mock

try:
    import torch

    from atic.repro import (
        get_environment_snapshot,
        hash_model_state,
        set_global_determinism,
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
class ReproducibilityTests(unittest.TestCase):
    def test_deterministic_mode_is_strict(self):
        with mock.patch.object(
            torch,
            "use_deterministic_algorithms",
        ) as deterministic:
            set_global_determinism(seed=42, deterministic=True)

        deterministic.assert_called_once_with(True)

    def test_environment_records_gpu_mapping_inputs(self):
        expected = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "42",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        with mock.patch.dict(os.environ, expected, clear=False):
            snapshot = get_environment_snapshot(
                device="cuda:0",
                repo_dir=os.getcwd(),
            )

        self.assertEqual(snapshot["process_environment"], expected)
        self.assertIn("deterministic_algorithms_enabled", snapshot)
        self.assertIn("visible_gpu_names", snapshot)

    def test_model_state_hash_proves_equal_initial_weights(self):
        torch.manual_seed(42)
        first = torch.nn.Linear(4, 3)
        torch.manual_seed(42)
        second = torch.nn.Linear(4, 3)

        self.assertEqual(hash_model_state(first), hash_model_state(second))
        with torch.no_grad():
            second.weight[0, 0] += 1
        self.assertNotEqual(hash_model_state(first), hash_model_state(second))


if __name__ == "__main__":
    unittest.main()
