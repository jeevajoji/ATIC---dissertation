"""
model.py — ATIC full model assembly
===================================

Updated version:
    - Uses CompressAI-based hyperprior entropy model.
    - Produces real rANS y/z streams in a versioned .atic container.
    - Reports complete-file BPP while retaining likelihood BPP for training.
    - Keeps ATIC architecture: tokenizer, Swin encoder/decoder, SAG, CBAM,
      decoder-synchronised adaptive scaling, and patch reconstructor.
"""

import hashlib
import hmac
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from compressai.models import CompressionModel

from atic.bitstream import (
    COLORSPACE_RGB,
    ENTROPY_CODER_RANS,
    ATICBitstreamError,
    bpp_from_num_bytes,
    normalise_model_hash,
    pack_atic,
    read_atic_file,
    sha256_file,
    unpack_atic,
    write_atic_bytes,
)
from atic.config import ArchitectureConfig
from atic.blocks.tokenizer import OverlappingPatchTokenizer
from atic.blocks.encoder import SwinEncoder
from atic.blocks.entropy import CompressAIHyperpriorEntropy
from atic.blocks.decoder import SwinDecoder
from atic.blocks.reconstructor import OverlappingPatchReconstructor


class ATICModel(CompressionModel):
    """ATIC analysis/synthesis model with a transferable entropy bitstream."""

    def __init__(
        self,
        config: ArchitectureConfig,
        H: int = 512,
        W: int = 512,
        model_id: Optional[Union[str, bytes]] = None,
    ):
        super().__init__()

        self.config = config
        self.H = int(H)
        self.W = int(W)
        if self.H <= 0 or self.W <= 0:
            raise ValueError("ATIC image dimensions must be positive")
        if config.swin_stages != 4:
            raise ValueError("The current ATIC encoder/decoder requires 4 Swin stages")
        if len(config.depths) != 4 or len(config.num_heads_enc) != 4:
            raise ValueError("ATIC depths and encoder-head lists must each have 4 items")
        if not isinstance(config.patch_size, int) or config.patch_size <= 0:
            raise ValueError("ATIC patch_size must be a positive integer")
        if config.use_overlapping_patches and config.patch_size < 2:
            raise ValueError("Overlapping patches require patch_size >= 2")
        if not isinstance(config.token_dim, int) or config.token_dim <= 0:
            raise ValueError("ATIC token_dim must be a positive integer")
        if not isinstance(config.window_size, int) or config.window_size <= 0:
            raise ValueError("ATIC window_size must be a positive integer")
        if any(
            not isinstance(depth, int) or depth <= 0 or depth % 2 != 0
            for depth in config.depths
        ):
            raise ValueError("Every ATIC Swin depth must be a positive even integer")
        if any(
            not isinstance(heads, int)
            or heads <= 0
            or (config.token_dim * (2**stage)) % heads != 0
            for stage, heads in enumerate(config.num_heads_enc)
        ):
            raise ValueError(
                "Every ATIC head count must be positive and divide its stage width"
            )
        if not config.use_hyperprior:
            raise ValueError("The publication ATIC codec requires the hyperprior")

        if config.use_overlapping_patches:
            stride = config.patch_size // 2
            padding = config.patch_size // 4
        else:
            stride = config.patch_size
            padding = 0

        self.stride = stride
        self.padding = padding

        self.token_H = (
            self.H + 2 * padding - config.patch_size
        ) // stride + 1
        self.token_W = (
            self.W + 2 * padding - config.patch_size
        ) // stride + 1
        if self.token_H <= 0 or self.token_W <= 0:
            raise ValueError("Patch configuration produces an empty token grid")

        self.latent_H, self.latent_W = self.token_H, self.token_W
        for _ in range(3):
            self.latent_H = (self.latent_H + 1) // 2
            self.latent_W = (self.latent_W + 1) // 2
        self.hyper_H, self.hyper_W = self.latent_H, self.latent_W
        for _ in range(2):
            self.hyper_H = (self.hyper_H + 1) // 2
            self.hyper_W = (self.hyper_W + 1) // 2

        self.latent_dim = config.token_dim * (2 ** (config.swin_stages - 1))

        self.tokenizer = OverlappingPatchTokenizer(
            in_channels=3,
            embed_dim=config.token_dim,
            patch_size=config.patch_size,
            stride=stride,
            padding=padding,
        )

        self.encoder = SwinEncoder(
            embed_dim=config.token_dim,
            token_H=self.token_H,
            token_W=self.token_W,
            depths=config.depths,
            num_heads=config.num_heads_enc,
            window_size=config.window_size,
            use_sag=config.use_sag,
            use_cbam=config.use_cbam,
        )

        # CompressAI entropy model.
        # For scientific BPP, keep hyperprior enabled for all variants.
        self.entropy = CompressAIHyperpriorEntropy(
            latent_dim=self.latent_dim,
            hyper_dim=192,
            use_adaptive_quant=config.use_adaptive_quant,
        )

        self.decoder = SwinDecoder(
            embed_dim=config.token_dim,
            token_H=self.token_H,
            token_W=self.token_W,
            depths=config.depths,
            num_heads=list(reversed(config.num_heads_enc)),
            window_size=config.window_size,
            use_sag=config.use_sag,
            use_cbam=config.use_cbam,
        )

        self.reconstructor = OverlappingPatchReconstructor(
            embed_dim=config.token_dim,
            patch_size=config.patch_size,
            stride=stride,
            padding=padding,
            output_H=self.H,
            output_W=self.W,
        )

        architecture_payload = {
            "architecture": asdict(config),
            "height": self.H,
            "width": self.W,
        }
        architecture_json = json.dumps(
            architecture_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.architecture_id = hashlib.sha256(architecture_json).hexdigest()
        self._model_hash: Optional[bytes] = None
        if model_id is not None:
            self.set_model_id(model_id)

    def set_model_id(self, model_id: Union[str, bytes]) -> None:
        """Set the decoder identifier embedded in future bitstreams."""

        self._model_hash = normalise_model_hash(model_id)

    @property
    def model_hash_hex(self) -> str:
        return self._require_model_hash().hex()

    def _require_model_hash(self) -> bytes:
        if self._model_hash is None:
            raise RuntimeError(
                "ATIC entropy coding requires an exact checkpoint identity. "
                "Call load_checkpoint() or set_model_id() first."
            )
        return self._model_hash

    def _validate_input(
        self,
        x: torch.Tensor,
        *,
        require_single: bool,
        check_finite: bool = False,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("ATIC input must be a torch.Tensor")
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                f"Expected input shape (N,3,{self.H},{self.W}), "
                f"received {tuple(x.shape)}"
            )
        if x.size(2) != self.H or x.size(3) != self.W:
            raise ValueError(
                f"This ATIC instance is fixed to {self.H}x{self.W}; "
                f"received {x.size(2)}x{x.size(3)}"
            )
        if require_single and x.size(0) != 1:
            raise ValueError("ATIC version-1 files contain exactly one image")
        if not torch.is_floating_point(x):
            raise TypeError("ATIC input must be a floating-point tensor")
        if check_finite and not torch.isfinite(x).all():
            raise ValueError("ATIC input contains NaN or infinite values")

    def forward(self, x: torch.Tensor) -> Dict:
        self._validate_input(x, require_single=False)
        tokens = self.tokenizer(x)

        latent_y, attn_map = self.encoder(tokens)

        y_hat, likelihoods, entropy_aux = self.entropy(
            latent_y,
            attn_map=attn_map,
            return_aux=True,
        )

        decoded_tokens = self.decoder(y_hat)

        x_hat = self.reconstructor(decoded_tokens)

        # Keep output in valid image range.
        x_hat = torch.sigmoid(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": likelihoods,
            "attn_map": attn_map,
            "gain_map": entropy_aux.get("gain_map"),
            "teacher_map": entropy_aux.get("teacher_map"),
            "z_hat": entropy_aux.get("z_hat"),
            "scales_hat": entropy_aux.get("scales_hat"),
            "means_hat": entropy_aux.get("means_hat"),
            "y_hat": y_hat,
        }

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, os.PathLike[str]],
        *,
        map_location=None,
        strict: bool = True,
    ):
        """Load an ATIC state dict and bind its SHA-256 as the codec model ID."""

        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:  # PyTorch versions before weights_only
            state_dict = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("ATIC checkpoint must contain a model state_dict")

        result = self.load_state_dict(state_dict, strict=strict)
        self.set_model_id(sha256_file(checkpoint_path))
        return result

    def _codec_flags(self) -> int:
        flags = 0
        flags |= int(bool(self.config.use_overlapping_patches)) << 0
        flags |= int(bool(self.config.use_adaptive_quant)) << 1
        flags |= int(bool(self.config.use_sag)) << 2
        flags |= int(bool(self.config.use_cbam)) << 3
        return flags

    @torch.no_grad()
    def compress(
        self,
        x: torch.Tensor,
        output_path: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        quality_id: int = 0,
        return_diagnostics: bool = False,
    ) -> Dict[str, object]:
        """Encode one image to a real, transferable ``.atic`` byte container."""

        model_hash = self._require_model_hash()
        self._validate_input(x, require_single=True, check_finite=True)
        if self.training:
            raise RuntimeError("Call model.eval() before model.compress()")

        # Build both EntropyBottleneck and GaussianConditional CDF tables.
        self.update(force=False)
        tokens = self.tokenizer(x)
        latent_y, _ = self.encoder(tokens)
        entropy_output = self.entropy.compress(
            latent_y,
            return_diagnostics=return_diagnostics,
        )

        y_strings = entropy_output["y_strings"]
        z_strings = entropy_output["z_strings"]
        if len(y_strings) != 1 or len(z_strings) != 1:
            raise RuntimeError("ATIC version-1 expected one y and one z stream")

        y_channels, y_height, y_width = entropy_output["y_shape"]
        _, z_height, z_width = entropy_output["z_shape"]
        bitstream = pack_atic(
            z_string=z_strings[0],
            y_string=y_strings[0],
            original_width=self.W,
            original_height=self.H,
            coded_width=self.W,
            coded_height=self.H,
            y_width=y_width,
            y_height=y_height,
            z_width=z_width,
            z_height=z_height,
            y_channels=y_channels,
            model_hash=model_hash,
            flags=self._codec_flags(),
            quality_id=quality_id,
        )
        if output_path is not None:
            write_atic_bytes(output_path, bitstream)

        num_bytes = len(bitstream)
        payload_bytes = len(z_strings[0]) + len(y_strings[0])
        result = {
            "bitstream": bitstream,
            "path": str(Path(output_path)) if output_path is not None else None,
            "num_bytes": num_bytes,
            "payload_bytes": payload_bytes,
            "header_bytes": num_bytes - payload_bytes,
            "bpp": bpp_from_num_bytes(num_bytes, self.H, self.W),
            "payload_bpp": bpp_from_num_bytes(
                payload_bytes,
                self.H,
                self.W,
            ),
            "y_bytes": len(y_strings[0]),
            "z_bytes": len(z_strings[0]),
            "model_hash": self.model_hash_hex,
        }
        if return_diagnostics:
            result["entropy_diagnostics"] = entropy_output["diagnostics"]
        return result

    @staticmethod
    def _read_container(source):
        if isinstance(source, (str, os.PathLike)):
            return read_atic_file(source)
        if isinstance(source, (bytes, bytearray, memoryview)):
            return unpack_atic(source)
        raise TypeError("ATIC source must be file path or bytes")

    def _validate_container(self, container) -> None:
        model_hash = self._require_model_hash()
        if not hmac.compare_digest(container.model_hash, model_hash):
            raise ATICBitstreamError(
                "ATIC model/checkpoint hash does not match this decoder"
            )
        if container.colorspace != COLORSPACE_RGB or container.bit_depth != 8:
            raise ATICBitstreamError("This decoder supports only 8-bit RGB ATIC files")
        if container.entropy_coder != ENTROPY_CODER_RANS:
            raise ATICBitstreamError("Unsupported ATIC entropy-coder identifier")
        if (container.coded_height, container.coded_width) != (self.H, self.W):
            raise ATICBitstreamError(
                "ATIC coded resolution does not match this decoder instance: "
                f"{container.coded_height}x{container.coded_width} versus "
                f"{self.H}x{self.W}"
            )
        if container.original_height > self.H or container.original_width > self.W:
            raise ATICBitstreamError("ATIC original dimensions exceed coded dimensions")
        if container.y_channels != self.latent_dim:
            raise ATICBitstreamError(
                f"ATIC latent channels {container.y_channels} do not match "
                f"decoder channels {self.latent_dim}"
            )
        if (container.y_height, container.y_width) != (
            self.latent_H,
            self.latent_W,
        ):
            raise ATICBitstreamError(
                "ATIC y shape does not match the deterministic encoder shape: "
                f"{container.y_height}x{container.y_width} versus "
                f"{self.latent_H}x{self.latent_W}"
            )
        if (container.z_height, container.z_width) != (
            self.hyper_H,
            self.hyper_W,
        ):
            raise ATICBitstreamError(
                "ATIC z shape does not match the deterministic hyperprior shape: "
                f"{container.z_height}x{container.z_width} versus "
                f"{self.hyper_H}x{self.hyper_W}"
            )
        if container.flags != self._codec_flags():
            raise ATICBitstreamError(
                "ATIC architecture flags do not match this decoder configuration"
            )

    @torch.no_grad()
    def decompress(
        self,
        source,
        *,
        return_info: bool = False,
        return_diagnostics: bool = False,
    ):
        """Decode a ``.atic`` file using no information from the source image.

        Requesting entropy diagnostics implies the structured ``return_info``
        result because the diagnostics cannot be represented by the default
        reconstruction-only tensor return.
        """

        if return_diagnostics:
            return_info = True
        self._require_model_hash()
        if self.training:
            raise RuntimeError("Call model.eval() before model.decompress()")
        container = self._read_container(source)
        self._validate_container(container)
        self.update(force=False)

        y_hat, entropy_aux = self.entropy.decompress(
            y_strings=[container.y_string],
            z_strings=[container.z_string],
            y_shape=(
                container.y_channels,
                container.y_height,
                container.y_width,
            ),
            z_shape=(
                self.entropy.hyper_dim,
                container.z_height,
                container.z_width,
            ),
            return_diagnostics=return_diagnostics,
        )
        decoded_tokens = self.decoder(y_hat)
        x_hat = torch.sigmoid(self.reconstructor(decoded_tokens))
        x_hat = x_hat[
            :,
            :,
            : container.original_height,
            : container.original_width,
        ]

        if not return_info:
            return x_hat

        payload_bytes = len(container.z_string) + len(container.y_string)
        result = {
            "x_hat": x_hat,
            "num_bytes": container.num_bytes,
            "payload_bytes": payload_bytes,
            "header_bytes": container.num_bytes - payload_bytes,
            "bpp": bpp_from_num_bytes(
                container.num_bytes,
                container.original_height,
                container.original_width,
            ),
            "payload_bpp": bpp_from_num_bytes(
                payload_bytes,
                container.original_height,
                container.original_width,
            ),
            "y_bytes": len(container.y_string),
            "z_bytes": len(container.z_string),
            "model_hash": container.model_hash_hex,
            "gain_map": entropy_aux["gain_map"],
            "z_hat": entropy_aux["z_hat"],
        }
        if return_diagnostics:
            result["entropy_diagnostics"] = entropy_aux["diagnostics"]
        return result
