"""
model.py — ATIC: Full model assembly
Wires all 7 blocks together. Entropy is fully connected.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict

from atic.config import ArchitectureConfig
from atic.blocks.tokenizer      import OverlappingPatchTokenizer
from atic.blocks.encoder        import SwinEncoder
from atic.blocks.quantizer      import AdaptiveQuantizer
from atic.blocks.entropy        import SimpleHyperpriorEntropy
from atic.blocks.decoder        import SwinDecoder
from atic.blocks.reconstructor  import OverlappingPatchReconstructor


class ATICModel(nn.Module):
    def __init__(self, config: ArchitectureConfig, H: int = 1088, W: int = 1920):
        super().__init__()
        self.config = config
        self.H = H
        self.W = W

        # Tokenizer stride/padding from overlap flag
        if config.use_overlapping_patches:
            stride  = config.patch_size // 2   # 8
            padding = config.patch_size // 4   # 4
        else:
            stride  = config.patch_size        # 16
            padding = 0

        # Token grid size
        self.token_H = (H + 2 * padding - config.patch_size) // stride + 1
        self.token_W = (W + 2 * padding - config.patch_size) // stride + 1

        # Latent channel dim: doubles 3 times through encoder stages
        self.latent_dim = config.token_dim * (2 ** (config.swin_stages - 1))  # 1024

        # Block 1
        self.tokenizer = OverlappingPatchTokenizer(
            in_channels=3,
            embed_dim=config.token_dim,
            patch_size=config.patch_size,
            stride=stride,
            padding=padding,
        )

        # Block 2
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

        # Block 3
        self.quantizer = AdaptiveQuantizer(
            latent_dim=self.latent_dim,
            use_adaptive=config.use_adaptive_quant,
        )

        # Block 4+5
        self.entropy = SimpleHyperpriorEntropy(
            latent_dim=self.latent_dim
        ) if config.use_hyperprior else None

        # Block 6
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

        # Block 7
        self.reconstructor = OverlappingPatchReconstructor(
            embed_dim=config.token_dim,
            patch_size=config.patch_size,
            stride=stride,
            padding=padding,
            output_H=H,
            output_W=W,
        )

    def forward(self, x: torch.Tensor) -> Dict:
        # 1. Tokenize
        tokens = self.tokenizer(x)

        # 2. Encode
        latent_z, attn_map = self.encoder(tokens)

        # 3. Quantize
        quantized_z, step_map = self.quantizer(latent_z, attn_map)

        # 4+5. Entropy
        likelihoods = None
        if self.entropy is not None:
            quantized_z, likelihoods = self.entropy(quantized_z)

        # 6. Decode
        decoded_tokens = self.decoder(quantized_z)

        # 7. Reconstruct
        x_hat = self.reconstructor(decoded_tokens)

        return {
            "x_hat"       : x_hat,
            "likelihoods" : likelihoods,   # dict {"y", "z"} or None
            "attn_map"    : attn_map,
            "step_map"    : step_map,
        }