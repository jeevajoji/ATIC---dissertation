"""
entropy.py — ATIC CompressAI Hyperprior Entropy Model
=====================================================

This module replaces the previous simplified hyperprior with a CompressAI-based
entropy path.

Pipeline:
    y                  : encoder latent
    adaptive gain      : optional attention-guided latent scaling
    z = h_a(|y_scaled|): hyper-analysis transform
    z_hat              : EntropyBottleneck quantisation + likelihoods
    params = h_s(z_hat): predicts Gaussian scales and means
    y_hat_scaled       : GaussianConditional quantisation + likelihoods
    y_hat              : inverse adaptive scaling for decoder

Returned likelihoods are used to compute BPP:
    BPP = sum(-log2(likelihoods)) / num_pixels
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class CompressAIHyperpriorEntropy(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        hyper_dim: int = 192,
        use_adaptive_quant: bool = True,
        gain_min: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hyper_dim = hyper_dim
        self.use_adaptive_quant = use_adaptive_quant
        self.gain_min = gain_min

        # Hyper-analysis: y -> z
        self.h_a = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_dim, hyper_dim, kernel_size=3, stride=2, padding=1),
        )

        # Entropy bottleneck for hyperlatent z
        self.entropy_bottleneck = EntropyBottleneck(hyper_dim)

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
        self.gaussian_conditional = GaussianConditional(None)

        # Refines SAG attention map into a decoder-available gain prior
        self.attn_refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _crop_like(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return src[:, :, : target.size(2), : target.size(3)]

    def _make_gain_map(
        self,
        y: torch.Tensor,
        attn_map: Optional[torch.Tensor],
        gain_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Creates spatial gain map.

        If adaptive quantisation is disabled, returns ones.
        If enabled, combines hyperprior-predicted gain with optional SAG attention.
        """
        B, _, H, W = y.shape

        if not self.use_adaptive_quant:
            return torch.ones(B, 1, H, W, device=y.device, dtype=y.dtype)

        # Gain predicted from hyper-synthesis branch
        gain_from_hyper = torch.sigmoid(gain_logits) + self.gain_min

        if attn_map is None:
            return gain_from_hyper

        if attn_map.shape[-2:] != (H, W):
            attn_map = F.interpolate(
                attn_map,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        gain_from_attn = self.attn_refine(attn_map) + self.gain_min

        # Blend both signals.
        # Hyperprior signal is decoder-available.
        # Attention signal is derived from encoder, so for true codec use this needs care.
        gain_map = 0.5 * gain_from_hyper + 0.5 * gain_from_attn

        return gain_map

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

        hyper_params = self.h_s(z_hat)
        hyper_params = self._crop_like(hyper_params, y)

        scales_hat = hyper_params[:, : self.latent_dim, :, :]
        means_hat = hyper_params[:, self.latent_dim : 2 * self.latent_dim, :, :]
        gain_logits = hyper_params[:, 2 * self.latent_dim : 2 * self.latent_dim + 1, :, :]

        scales_hat = scales_hat.abs().clamp(min=1e-6)

        gain_map = self._make_gain_map(y, attn_map, gain_logits)

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
            }
            return y_hat, likelihoods, aux

        return y_hat, likelihoods