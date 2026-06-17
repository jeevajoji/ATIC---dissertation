"""
config.py — ATIC Architecture Configuration

For the publication/dissertation version:
    All variants use CompressAI hyperprior entropy modelling.
    Ablations focus on:
        - overlapping patches
        - SAG
        - CBAM
        - adaptive quantisation/gain
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ArchitectureConfig:
    # Block 1: Tokenizer
    patch_size: int = 16
    token_dim: int = 128
    use_overlapping_patches: bool = True

    # Block 2 & 6: Swin Encoder/Decoder
    swin_stages: int = 4
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads_enc: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    window_size: int = 8

    # Attention stack inside each Swin block
    use_sag: bool = True       # Spatial Attention Gate
    use_cbam: bool = True      # CBAM channel attention

    # Block 3: Quantizer
    use_adaptive_quant: bool = True   # False = uniform quantisation

    # Block 4+5: Entropy model
    use_hyperprior: bool = True