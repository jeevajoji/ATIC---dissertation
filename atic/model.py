"""
model.py — ATIC full model assembly
===================================

Updated version:
    - Uses CompressAI-based hyperprior entropy model.
    - Uses likelihood-based BPP.
    - Keeps ATIC architecture: tokenizer, Swin encoder/decoder, SAG, CBAM,
      adaptive attention-guided entropy scaling, and patch reconstructor.
"""

import torch
import torch.nn as nn
from typing import Dict

from compressai.entropy_models import EntropyBottleneck

from atic.config import ArchitectureConfig
from atic.blocks.tokenizer import OverlappingPatchTokenizer
from atic.blocks.encoder import SwinEncoder
from atic.blocks.entropy import CompressAIHyperpriorEntropy
from atic.blocks.decoder import SwinDecoder
from atic.blocks.reconstructor import OverlappingPatchReconstructor


class ATICModel(nn.Module):
    def __init__(self, config: ArchitectureConfig, H: int = 512, W: int = 512):
        super().__init__()

        self.config = config
        self.H = H
        self.W = W

        if config.use_overlapping_patches:
            stride = config.patch_size // 2
            padding = config.patch_size // 4
        else:
            stride = config.patch_size
            padding = 0

        self.stride = stride
        self.padding = padding

        self.token_H = (H + 2 * padding - config.patch_size) // stride + 1
        self.token_W = (W + 2 * padding - config.patch_size) // stride + 1

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
            output_H=H,
            output_W=W,
        )

    def forward(self, x: torch.Tensor) -> Dict:
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
            "z_hat": entropy_aux.get("z_hat"),
            "scales_hat": entropy_aux.get("scales_hat"),
            "means_hat": entropy_aux.get("means_hat"),
            "y_hat": y_hat,
        }

    def aux_loss(self) -> torch.Tensor:
        """
        CompressAI entropy bottleneck auxiliary loss.
        Needed to train entropy quantiles.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for module in self.modules():
            if isinstance(module, EntropyBottleneck):
                loss = loss + module.loss()

        return loss

    def update(self, force: bool = False):
        updated = False
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "update"):
                try:
                    updated = module.update(force=force) or updated
                except TypeError:
                    updated = module.update() or updated
        return updated