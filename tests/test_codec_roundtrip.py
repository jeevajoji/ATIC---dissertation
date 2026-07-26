import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

try:
    import torch
    from PIL import Image

    from atic.bitstream import (
        ATICBitstreamError,
        pack_atic,
        sha256_file,
        unpack_atic,
    )
    from atic.config import ArchitectureConfig
    from atic.model import ATICModel

    CODEC_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on lightweight hosts
    if os.environ.get("ATIC_REQUIRE_CODEC_TESTS") == "1":
        raise
    CODEC_IMPORT_ERROR = exc


def small_config():
    return ArchitectureConfig(
        patch_size=8,
        token_dim=8,
        use_overlapping_patches=True,
        swin_stages=4,
        depths=[2, 2, 2, 2],
        num_heads_enc=[1, 2, 4, 8],
        window_size=2,
        use_sag=True,
        use_cbam=True,
        use_adaptive_quant=True,
        use_hyperprior=True,
    )


@unittest.skipIf(
    CODEC_IMPORT_ERROR is not None,
    f"PyTorch/CompressAI codec stack unavailable: {CODEC_IMPORT_ERROR}",
)
class ATICCodecRoundTripTests(unittest.TestCase):
    PIXEL_ATOL = 1e-6

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_model(self, height=64, width=64, model_id="unit-model"):
        torch.manual_seed(1234)
        model = ATICModel(
            small_config(),
            H=height,
            W=width,
            model_id=model_id,
        )
        model.eval()
        model.update(force=True)
        return model

    def test_groupnorm_sag_configuration_reaches_full_model(self):
        config = small_config()
        config.sag_normalization = "group"
        model = ATICModel(config, H=64, W=64)

        self.assertTrue(
            any(
                isinstance(module, torch.nn.GroupNorm)
                for module in model.modules()
            )
        )
        self.assertFalse(
            any(
                isinstance(module, torch.nn.BatchNorm2d)
                for module in model.modules()
            )
        )

    def test_encoder_terminal_latent_norm_toggle_reaches_model(self):
        torch.manual_seed(4321)
        control = ATICModel(small_config(), H=64, W=64)
        self.assertIsInstance(control.encoder.norm, torch.nn.LayerNorm)
        self.assertIsInstance(control.decoder.norm, torch.nn.LayerNorm)

        config = small_config()
        config.use_encoder_latent_norm = False
        torch.manual_seed(4321)
        no_norm = ATICModel(config, H=64, W=64)

        self.assertIsInstance(no_norm.encoder.norm, torch.nn.Identity)
        # The intervention is encoder-only; decoder normalisation is retained.
        self.assertIsInstance(no_norm.decoder.norm, torch.nn.LayerNorm)
        self.assertNotEqual(control.architecture_id, no_norm.architecture_id)

        control_state = control.state_dict()
        no_norm_state = no_norm.state_dict()
        self.assertEqual(
            set(control_state) - set(no_norm_state),
            {"encoder.norm.weight", "encoder.norm.bias"},
        )
        self.assertFalse(set(no_norm_state) - set(control_state))
        for name in no_norm_state:
            self.assertTrue(
                torch.equal(control_state[name], no_norm_state[name]),
                msg=f"Common initial tensor differs: {name}",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "no_norm.pth"
            torch.save(no_norm_state, checkpoint)
            with self.assertRaises(RuntimeError):
                control.load_checkpoint(checkpoint, strict=True)

    def assert_entropy_diagnostics_equal(self, sender, receiver):
        for name in ("z_symbols", "y_symbols", "indexes"):
            sender_tensor = sender[name].detach().cpu()
            receiver_tensor = receiver[name].detach().cpu()
            self.assertEqual(sender_tensor.dtype, torch.int32)
            self.assertEqual(receiver_tensor.dtype, sender_tensor.dtype)
            self.assertEqual(
                tuple(receiver_tensor.shape),
                tuple(sender_tensor.shape),
            )
            self.assertTrue(
                torch.equal(sender_tensor, receiver_tensor),
                msg=f"Entropy diagnostic mismatch: {name}",
            )

    def assert_reconstruction_close(self, expected, actual):
        self.assertEqual(tuple(expected.shape), tuple(actual.shape))
        self.assertTrue(torch.isfinite(expected).all())
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertLessEqual(float(actual.max()), 1.0)
        torch.testing.assert_close(
            expected,
            actual,
            rtol=0,
            atol=self.PIXEL_ATOL,
        )

    @staticmethod
    def entropy_buffer_snapshot(model):
        suffixes = (
            "_quantized_cdf",
            "_offset",
            "_cdf_length",
            "scale_table",
            "scale_bound",
            "_log_gain_max",
        )
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
            if name.startswith("entropy.")
            and name.endswith(suffixes)
        }

    def test_real_entropy_round_trip_matches_eval_forward(self):
        model = self.make_model()
        source = torch.rand(1, 3, 64, 64)

        with torch.no_grad():
            forward_output = model(source)
            encoded = model.compress(source, return_diagnostics=True)
            repeated = model.compress(source, return_diagnostics=True)
            decoded = model.decompress(
                encoded["bitstream"],
                return_info=True,
                return_diagnostics=True,
            )

        self.assert_entropy_diagnostics_equal(
            encoded["entropy_diagnostics"],
            decoded["entropy_diagnostics"],
        )
        self.assertTrue(
            torch.equal(
                forward_output["gain_map"],
                decoded["gain_map"],
            ),
            msg="Forward and decoder must derive the exact same gain from z_hat",
        )
        self.assertEqual(
            encoded["bitstream"],
            repeated["bitstream"],
            msg="Repeated encoding of the same image must be deterministic",
        )
        self.assert_reconstruction_close(
            forward_output["x_hat"],
            decoded["x_hat"],
        )
        self.assertEqual(encoded["num_bytes"], len(encoded["bitstream"]))
        self.assertEqual(
            encoded["bpp"],
            len(encoded["bitstream"]) * 8 / (64 * 64),
        )
        container = unpack_atic(encoded["bitstream"])
        self.assertEqual(
            encoded["payload_bytes"],
            len(container.z_string) + len(container.y_string),
        )
        self.assertEqual(
            model.entropy.entropy_bottleneck.entropy_coder.name,
            "ans",
        )
        self.assertEqual(
            model.entropy.gaussian_conditional.entropy_coder.name,
            "ans",
        )

    def test_unbound_random_weights_cannot_create_or_accept_streams(self):
        source_model = ATICModel(small_config(), H=64, W=64)
        decoder_model = ATICModel(small_config(), H=64, W=64)
        source_model.eval()
        decoder_model.eval()
        source = torch.rand(1, 3, 64, 64)

        with self.assertRaisesRegex(RuntimeError, "exact checkpoint identity"):
            source_model.compress(source)
        with self.assertRaisesRegex(RuntimeError, "exact checkpoint identity"):
            decoder_model.decompress(b"not-a-container")

    def test_operational_stream_does_not_invoke_teacher_projection(self):
        model = self.make_model()
        source = torch.rand(1, 3, 64, 64)

        with mock.patch.object(
            model.entropy,
            "_make_teacher_map",
            side_effect=AssertionError("teacher entered operational codec"),
        ) as teacher:
            first = model.compress(source)["bitstream"]
            decoded = model.decompress(first)

        teacher.assert_not_called()
        self.assertEqual(tuple(decoded.shape), tuple(source.shape))

    def test_wrong_decoder_model_id_is_rejected_before_entropy_decode(self):
        source_model = self.make_model(model_id="correct")
        source = torch.rand(1, 3, 64, 64)
        encoded = source_model.compress(source)["bitstream"]

        wrong_decoder = self.make_model(model_id="wrong")
        wrong_decoder.load_state_dict(source_model.state_dict())
        with self.assertRaisesRegex(ATICBitstreamError, "checkpoint hash"):
            wrong_decoder.decompress(encoded)

    def test_forged_latent_shape_is_rejected_before_entropy_decode(self):
        model = self.make_model()
        source = torch.rand(1, 3, 64, 64)
        container = unpack_atic(model.compress(source)["bitstream"])
        forged = pack_atic(
            z_string=container.z_string,
            y_string=container.y_string,
            original_width=container.original_width,
            original_height=container.original_height,
            coded_width=container.coded_width,
            coded_height=container.coded_height,
            y_width=container.y_width,
            y_height=container.y_height + 1,
            z_width=container.z_width,
            z_height=container.z_height,
            y_channels=container.y_channels,
            model_hash=container.model_hash,
            flags=container.flags,
            quality_id=container.quality_id,
            bit_depth=container.bit_depth,
            colorspace=container.colorspace,
            entropy_coder=container.entropy_coder,
        )

        with mock.patch.object(
            model.entropy,
            "decompress",
            wraps=model.entropy.decompress,
        ) as entropy_decode:
            with self.assertRaisesRegex(ATICBitstreamError, "y shape"):
                model.decompress(forged)
            entropy_decode.assert_not_called()

    def test_odd_latent_dimensions_round_trip(self):
        model = self.make_model(height=72, width=88)
        source = torch.rand(1, 3, 72, 88)
        with torch.no_grad():
            expected = model(source)["x_hat"]
            encoded = model.compress(source, return_diagnostics=True)
            decoded = model.decompress(
                encoded["bitstream"],
                return_info=True,
                return_diagnostics=True,
            )
        reconstructed = decoded["x_hat"]
        container = unpack_atic(encoded["bitstream"])

        self.assertEqual(tuple(reconstructed.shape), (1, 3, 72, 88))
        self.assert_entropy_diagnostics_equal(
            encoded["entropy_diagnostics"],
            decoded["entropy_diagnostics"],
        )
        self.assert_reconstruction_close(expected, reconstructed)
        self.assertEqual(
            (container.original_height, container.original_width),
            (72, 88),
        )
        self.assertEqual((container.y_height, container.y_width), (3, 3))
        self.assertEqual((container.z_height, container.z_width), (1, 1))

    def test_legacy_empty_gaussian_buffers_are_rebuilt_after_loading(self):
        source_model = self.make_model()
        state_dict = source_model.state_dict()
        for suffix in (
            "_quantized_cdf",
            "_offset",
            "_cdf_length",
            "scale_table",
        ):
            key = f"entropy.gaussian_conditional.{suffix}"
            state_dict[key] = torch.empty(0, dtype=state_dict[key].dtype)

        restored = ATICModel(
            small_config(),
            H=64,
            W=64,
            model_id="legacy-checkpoint",
        )
        restored.load_state_dict(state_dict, strict=True)
        restored.update(force=True)

        self.assertGreater(
            restored.entropy.gaussian_conditional.scale_table.numel(),
            0,
        )
        self.assertGreater(
            restored.entropy.gaussian_conditional._quantized_cdf.numel(),
            0,
        )

    def test_load_checkpoint_rejects_persisted_gain_bound_mismatch(self):
        source_config = small_config()
        source_config.gain_max = 3.0
        source_model = ATICModel(source_config, H=64, W=64)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "gain_max_3.pth"
            torch.save(source_model.state_dict(), checkpoint)

            receiver = ATICModel(small_config(), H=64, W=64)
            with self.assertRaisesRegex(
                ValueError,
                "Checkpoint gain_max does not match",
            ):
                receiver.load_checkpoint(checkpoint, map_location="cpu")

        self.assertIsNone(receiver._model_hash)

    def test_fresh_process_decodes_using_only_checkpoint_and_bitstream(self):
        model = self.make_model()
        source = torch.rand(1, 3, 64, 64)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pth"
            bitstream = root / "image.atic"
            subprocess_output = root / "decoded.pt"

            # Save populated entropy buffers to exercise registered-buffer
            # resizing when a fresh ATICModel loads the checkpoint.
            torch.save(model.state_dict(), checkpoint)
            model.set_model_id(sha256_file(checkpoint))
            with torch.no_grad():
                encoded = model.compress(
                    source,
                    output_path=bitstream,
                    return_diagnostics=True,
                )
                expected = model.decompress(
                    bitstream,
                    return_info=True,
                    return_diagnostics=True,
                )

            command = [
                sys.executable,
                "-m",
                "tests.fresh_decode_helper",
                "--checkpoint",
                str(checkpoint),
                "--bitstream",
                str(bitstream),
                "--output",
                str(subprocess_output),
                "--height",
                "64",
                "--width",
                "64",
            ]
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            try:
                actual = torch.load(
                    subprocess_output,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                actual = torch.load(subprocess_output, map_location="cpu")
            self.assert_entropy_diagnostics_equal(
                encoded["entropy_diagnostics"],
                expected["entropy_diagnostics"],
            )
            self.assert_entropy_diagnostics_equal(
                encoded["entropy_diagnostics"],
                actual["entropy_diagnostics"],
            )
            self.assert_reconstruction_close(
                expected["x_hat"].cpu(),
                actual["x_hat"],
            )

            expected_buffers = self.entropy_buffer_snapshot(model)
            for phase in ("before_decode", "after_decode"):
                actual_buffers = actual[f"entropy_buffers_{phase}"]
                self.assertEqual(
                    set(expected_buffers),
                    set(actual_buffers),
                )
                for name, tensor in expected_buffers.items():
                    actual_tensor = actual_buffers[name]
                    self.assertEqual(
                        tensor.dtype,
                        actual_tensor.dtype,
                        msg=f"Loaded entropy buffer dtype mismatch: {name}",
                    )
                    self.assertEqual(
                        tuple(tensor.shape),
                        tuple(actual_tensor.shape),
                        msg=f"Loaded entropy buffer shape mismatch: {name}",
                    )
                    self.assertTrue(
                        torch.equal(tensor, actual_tensor),
                        msg=(
                            "Loaded entropy buffer mismatch "
                            f"{phase}: {name}"
                        ),
                    )

    def test_cli_png_to_atic_to_png_in_separate_processes(self):
        model = self.make_model()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pth"
            run_config = root / "run_config.json"
            source_image = root / "source.png"
            bitstream = root / "image.atic"
            decoded_image = root / "decoded.png"

            torch.save(model.state_dict(), checkpoint)
            run_config.write_text(
                json.dumps(
                    {
                        "height": 64,
                        "width": 64,
                        "architecture": asdict(small_config()),
                    }
                ),
                encoding="utf-8",
            )
            Image.new("RGB", (64, 64), color=(31, 127, 223)).save(source_image)

            common = [
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(run_config),
            ]
            compress_command = [
                sys.executable,
                "-m",
                "atic.codec_cli",
                "compress",
                "--input",
                str(source_image),
                "--output",
                str(bitstream),
                *common,
            ]
            compress_result = subprocess.run(
                compress_command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                compress_result.returncode,
                0,
                msg=(
                    f"stdout:\n{compress_result.stdout}\n"
                    f"stderr:\n{compress_result.stderr}"
                ),
            )
            self.assertTrue(bitstream.is_file())

            decompress_command = [
                sys.executable,
                "-m",
                "atic.codec_cli",
                "decompress",
                "--input",
                str(bitstream),
                "--output",
                str(decoded_image),
                *common,
            ]
            decompress_result = subprocess.run(
                decompress_command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                decompress_result.returncode,
                0,
                msg=(
                    f"stdout:\n{decompress_result.stdout}\n"
                    f"stderr:\n{decompress_result.stderr}"
                ),
            )
            with Image.open(decoded_image) as decoded:
                self.assertEqual(decoded.mode, "RGB")
                self.assertEqual(decoded.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
