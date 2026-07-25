"""
entropy.py — ATIC CompressAI Hyperprior Entropy Model
=====================================================

This module replaces the previous simplified hyperprior with a CompressAI-based
entropy path.

Pipeline:
    y                  : encoder latent
    adaptive gain      : decoder-synchronised hyperprior latent scaling
    z = h_a(|y|)       : hyper-analysis transform
    z_hat              : EntropyBottleneck quantisation + likelihoods
    params = h_s(z_hat): predicts Gaussian scales and means
    y_hat_scaled       : GaussianConditional quantisation + likelihoods
    y_hat              : inverse adaptive scaling for decoder

Returned likelihoods are used to compute BPP:
    BPP = sum(-log2(likelihoods)) / num_pixels
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.base import get_scale_table


class CompressAIHyperpriorEntropy(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        hyper_dim: int = 192,
        use_adaptive_quant: bool = True,
        gain_max: float = 2.0,
    ):
        super().__init__()

        if not math.isfinite(gain_max) or gain_max <= 1.0:
            raise ValueError("gain_max must be a finite value greater than 1")

        self.latent_dim = latent_dim
        self.hyper_dim = hyper_dim
        self.use_adaptive_quant = use_adaptive_quant
        # Persist the bound in checkpoints because it changes decoding. A
        # plain Python attribute could silently diverge between sender and
        # receiver even when both use the same checkpoint file.
        self.register_buffer(
            "_log_gain_max",
            torch.tensor(math.log(float(gain_max)), dtype=torch.float32),
        )

        # Hyper-analysis: y -> z
        self.h_a = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_dim, hyper_dim, kernel_size=3, stride=2, padding=1),
        )

        # Entropy bottleneck for hyperlatent z
        self.entropy_bottleneck = EntropyBottleneck(
            hyper_dim,
            entropy_coder="ans",
        )

        # Hyper-synthesis: z_hat -> Gaussian params for y
        # Output: latent_dim scales + latent_dim means + optional 1 gain logit
        out_channels = (2 * latent_dim) + 1

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(
                hyper_dim,
                hyper_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hyper_dim,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

        # Gaussian conditional entropy model for main latent y
        self.gaussian_conditional = GaussianConditional(
            None,
            entropy_coder="ans",
        )

    @property
    def gain_max(self) -> float:
        """Return the checkpoint-bound maximum adaptive gain."""

        return math.exp(float(self._log_gain_max.detach().cpu()))

    @staticmethod
    def _crop_to_spatial(
        src: torch.Tensor,
        spatial_shape: Sequence[int],
    ) -> torch.Tensor:
        height, width = (int(value) for value in spatial_shape)
        if height <= 0 or width <= 0:
            raise ValueError("Spatial dimensions must be positive")
        if src.size(2) < height or src.size(3) < width:
            raise ValueError(
                "Hyper-synthesis output is smaller than the declared y shape: "
                f"{tuple(src.shape[-2:])} versus {(height, width)}"
            )
        return src[:, :, :height, :width]

    def _split_hyper_params(
        self,
        z_hat: torch.Tensor,
        y_spatial_shape: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hyper_params = self._crop_to_spatial(
            self.h_s(z_hat),
            y_spatial_shape,
        )
        scales_hat = hyper_params[:, : self.latent_dim, :, :]
        means_hat = hyper_params[
            :, self.latent_dim : 2 * self.latent_dim, :, :
        ]
        gain_logits = hyper_params[
            :, 2 * self.latent_dim : 2 * self.latent_dim + 1, :, :
        ]
        return scales_hat.abs().clamp(min=1e-6), means_hat, gain_logits

    def _score_to_gain_map(
        self,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Map a bounded score to a scale-normalised positive gain.

        Spatial centring makes the log-gain mean zero independently for every
        image and channel. Consequently the spatial geometric mean of the gain
        is one. A score in ``[-1, 1]`` has a centred range of ``[-2, 2]``, so
        the half-log scaling bounds the gain by
        ``[1 / gain_max, gain_max]``. This removes global scale as a shortcut;
        it does not by itself guarantee equal actual bitrate.
        """
        centred_score = score - score.mean(dim=(-2, -1), keepdim=True)
        log_gain_bound = self._log_gain_max.to(
            device=score.device,
            dtype=score.dtype,
        )
        log_gain = 0.5 * log_gain_bound * centred_score
        return torch.exp(log_gain)

    def _make_gain_map(
        self,
        gain_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Create the operational decoder-synchronised student gain.

        ``gain_logits`` comes only from hyper-synthesis of decoded ``z_hat``.
        The source image and encoder attention are therefore unnecessary at
        the decoder and no separate attention tensor enters the bitstream.
        """
        if not self.use_adaptive_quant:
            return torch.ones_like(gain_logits)
        return self._score_to_gain_map(torch.tanh(gain_logits))

    def _make_teacher_map(
        self,
        attn_map: Optional[torch.Tensor],
        spatial_shape: Sequence[int],
    ) -> Optional[torch.Tensor]:
        """Create the deterministic, parameter-free DSAD target transform.

        The deepest encoder SAG map supervises the decoder-synchronised
        student during training only. Detaching after interpolation prevents
        the distillation loss from changing SAG or the encoder through this
        target. SAG can still evolve through the ordinary RD objective, making
        this an online stop-gradient teacher. The target transform is never
        called by ``compress`` or ``decompress``.
        """

        if attn_map is None:
            return None
        target_shape = tuple(int(value) for value in spatial_shape)
        if attn_map.shape[-2:] != target_shape:
            attn_map = F.interpolate(
                attn_map,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            )
        teacher_attention = attn_map.detach().clamp(0.0, 1.0)
        teacher_score = (2.0 * teacher_attention) - 1.0
        return self._score_to_gain_map(teacher_score)

    def forward(
        self,
        y: torch.Tensor,
        attn_map: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        Args:
            y: encoder latent, shape (B, C, H, W)
            attn_map: optional SAG spatial attention map

        Returns:
            y_hat: quantised/modelled latent for decoder
            likelihoods: dict with y and z likelihoods
            aux: optional dict with gain_map, z_hat
        """

        # Important: use scaled latent for entropy modelling
        # Preliminary gain logits require z_hat, so first estimate z from |y|.
        z = self.h_a(torch.abs(y))

        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        scales_hat, means_hat, gain_logits = self._split_hyper_params(
            z_hat,
            y.shape[-2:],
        )
        gain_map = self._make_gain_map(gain_logits)
        teacher_map = self._make_teacher_map(attn_map, y.shape[-2:])

        y_scaled = y * gain_map

        y_hat_scaled, y_likelihoods = self.gaussian_conditional(
            y_scaled,
            scales_hat,
            means=means_hat,
        )

        y_hat = y_hat_scaled / gain_map.clamp(min=1e-6)

        likelihoods: Dict[str, torch.Tensor] = {
            "y": y_likelihoods,
            "z": z_likelihoods,
        }

        if return_aux:
            aux = {
                "z_hat": z_hat,
                "gain_map": gain_map,
                "scales_hat": scales_hat,
                "means_hat": means_hat,
                "teacher_map": teacher_map,
            }
            return y_hat, likelihoods, aux

        return y_hat, likelihoods

    def update(self, force: bool = False) -> bool:
        """Build the CDF tables required by the real entropy coder."""

        updated = bool(self.entropy_bottleneck.update(force=force))
        updated = bool(
            self.gaussian_conditional.update_scale_table(
                get_scale_table(),
                force=force,
            )
        ) or updated
        return updated

    def _require_rans(self) -> None:
        coders = (
            self.entropy_bottleneck.entropy_coder.name,
            self.gaussian_conditional.entropy_coder.name,
        )
        if coders != ("ans", "ans"):
            raise RuntimeError(
                "ATIC version 1 requires CompressAI's rANS entropy backend"
            )

    def _z_symbols(self, z_value: torch.Tensor) -> torch.Tensor:
        """Return the integer hyperlatent symbols used by EntropyBottleneck."""

        medians = self.entropy_bottleneck._get_medians().detach().unsqueeze(0)
        medians = medians.expand_as(z_value)
        return self.entropy_bottleneck.quantize(
            z_value,
            "symbols",
            medians,
        )

    @torch.no_grad()
    def compress(
        self,
        y: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Dict[str, object]:
        """Entropy-code ``y`` and its hyperlatent using decoder-safe context."""

        if y.ndim != 4 or y.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected y with shape (N,{self.latent_dim},H,W), "
                f"received {tuple(y.shape)}"
            )
        if self.training:
            raise RuntimeError("Entropy compression requires eval() mode")

        self._require_rans()
        self.update(force=False)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(
            z_strings,
            z.size()[-2:],
        )

        scales_hat, means_hat, gain_logits = self._split_hyper_params(
            z_hat,
            y.shape[-2:],
        )
        gain_map = self._make_gain_map(gain_logits)
        y_scaled = y * gain_map
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(
            y_scaled,
            indexes,
            means=means_hat,
        )

        result = {
            "y_strings": y_strings,
            "z_strings": z_strings,
            "y_shape": tuple(int(value) for value in y.shape[1:]),
            "z_shape": tuple(int(value) for value in z.shape[1:]),
            "gain_map": gain_map,
            "z_hat": z_hat,
        }
        if return_diagnostics:
            result["diagnostics"] = {
                "z_symbols": self._z_symbols(z),
                "y_symbols": self.gaussian_conditional.quantize(
                    y_scaled,
                    "symbols",
                    means_hat,
                ),
                "indexes": indexes,
            }
        return result

    @torch.no_grad()
    def decompress(
        self,
        *,
        y_strings: Sequence[bytes],
        z_strings: Sequence[bytes],
        y_shape: Sequence[int],
        z_shape: Sequence[int],
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Decode entropy strings without access to the source image."""

        if self.training:
            raise RuntimeError("Entropy decompression requires eval() mode")
        if len(y_shape) != 3 or int(y_shape[0]) != self.latent_dim:
            raise ValueError(
                f"Invalid y shape {tuple(y_shape)} for latent_dim={self.latent_dim}"
            )
        if len(z_shape) != 3 or int(z_shape[0]) != self.hyper_dim:
            raise ValueError(
                f"Invalid z shape {tuple(z_shape)} for hyper_dim={self.hyper_dim}"
            )
        if len(y_strings) != len(z_strings) or not y_strings:
            raise ValueError("y and z entropy stream counts must match and be non-empty")

        self._require_rans()
        self.update(force=False)
        z_hat = self.entropy_bottleneck.decompress(
            list(z_strings),
            tuple(int(value) for value in z_shape[-2:]),
        )
        scales_hat, means_hat, gain_logits = self._split_hyper_params(
            z_hat,
            tuple(int(value) for value in y_shape[-2:]),
        )
        gain_map = self._make_gain_map(gain_logits)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat_scaled = self.gaussian_conditional.decompress(
            list(y_strings),
            indexes,
            means=means_hat,
        )
        y_hat = y_hat_scaled / gain_map.clamp(min=1e-6)

        aux = {
            "gain_map": gain_map,
            "means_hat": means_hat,
            "scales_hat": scales_hat,
            "z_hat": z_hat,
        }
        if return_diagnostics:
            aux["diagnostics"] = {
                "z_symbols": self._z_symbols(z_hat),
                "y_symbols": self.gaussian_conditional.quantize(
                    y_hat_scaled,
                    "symbols",
                    means_hat,
                ),
                "indexes": indexes,
            }
        return y_hat, aux
