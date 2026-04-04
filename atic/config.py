"""
config.py — ATIC Architecture Configuration
Controls which components are active for each ablation stage.

Ablation ladder:
    A1: no overlap, no SAG, no CBAM, no hyperprior  ← true baseline
    A2: + overlapping patches
    A3: + SAG
    A4: + CBAM
    A5: + adaptive quantizer  (SAG map fed to quantizer)
    A6: + hyperprior           ← full ATIC
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