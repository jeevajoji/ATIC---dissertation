"""
entropy.py — ATIC Block 4+5: Simplified Hyperprior Entropy Model
=================================================================
compressai-free implementation. Suitable for Kaggle/Colab without
needing compressai installed.

Architecture:
    Hyper encoder: z → h  (compress latent to side info)
    Hyper decoder: h → μ, σ  (predict distribution params)
    Gaussian likelihood used for differentiable rate estimation.

Forward returns:
    z_hat       : quantised latent
    likelihoods : {"y": p(z_hat), "z": p(h_hat)}
                  used by ATICLoss to compute BPP
"""
import math
import torch
import torch.nn as nn
from typing import Dict, Tuple


class SimpleHyperpriorEntropy(nn.Module):
    def __init__(self, latent_dim: int = 1024):
        super().__init__()

        mid = latent_dim // 4   # 256

        # Hyper encoder: compress latent spatially and channel-wise
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim // 2, mid, kernel_size=3, stride=2, padding=1),
        )

        # Hyper decoder: upsample back to latent resolution, predict μ and σ
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(mid, latent_dim // 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(latent_dim // 2, latent_dim * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
        )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z: (B, latent_dim, H, W)

        Returns:
            z_hat       : (B, latent_dim, H, W)  quantised latent
            likelihoods : {"y": tensor, "z": tensor}  for rate loss
        """
        # Encode side info
        h = self.hyper_encoder(z)

        # Quantise side info
        if self.training:
            h_hat = h + torch.rand_like(h) - 0.5
        else:
            h_hat = torch.round(h)

        # Predict μ, σ from side info
        gaussian_params = self.hyper_decoder(h_hat)
        
        # Ensure dimensions strictly match z
        H, W = z.shape[2:]
        if gaussian_params.shape[2:] != (H, W):
            gaussian_params = gaussian_params[:, :, :H, :W]
            
        scales, means   = gaussian_params.chunk(2, dim=1)
        scales          = scales.abs().clamp(min=1e-5)

        # Quantise main latent
        if self.training:
            z_hat = z + torch.rand_like(z) - 0.5
        else:
            z_hat = torch.round(z)

        # Likelihoods
        likelihoods_y = self._gaussian_likelihood(z_hat, means, scales)
        # Side info: uniform approx (constant 0.1 is a placeholder;
        # a real implementation uses a non-parametric entropy model here)
        likelihoods_z = torch.full_like(h_hat, 0.1)

        return z_hat, {"y": likelihoods_y, "z": likelihoods_z}

    @staticmethod
    def _gaussian_likelihood(
        x: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian PDF evaluated at x."""
        p = torch.exp(-0.5 * ((x - mean) / scale) ** 2) / \
            (math.sqrt(2 * math.pi) * scale)
        return p.clamp(min=1e-5, max=0.999)


# Keep old name as alias so any existing imports don't break
HyperpriorEntropyModel = SimpleHyperpriorEntropy