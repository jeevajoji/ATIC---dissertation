"""
quantizer.py — ATIC Block 3: Attention-Guided Adaptive Quantizer
=================================================================
Quantises the encoder latent into integers ready for entropy coding.

Standard uniform quantizer:
    Every element rounded to the nearest integer.
    Same step size everywhere → wastes bits on smooth regions.

ATIC adaptive quantizer:
    Step size is PER-REGION, driven by the spatial attention map from SAG.
    High attention (complex region) → small step → fine quantisation → more bits
    Low  attention (smooth region)  → large step → coarse quantisation → fewer bits

Step size formula:
    step(x, y) = base_step * exp(−alpha * attn(x, y))

    When attn → 1.0:  step → base_step * exp(−alpha)  ← small, fine
    When attn → 0.0:  step → base_step * 1.0           ← large, coarse

    alpha controls how aggressively attention drives the step size.
    base_step is a learned per-channel scale.

Straight-through estimator (STE):
    Rounding has zero gradient (almost everywhere), which would stop
    training dead. STE fixes this:
        Forward:  y = round(x / step) * step   ← real quantisation
        Backward: dy/dx ≈ 1                    ← pretend it's identity

    In eval mode: true rounding.
    In train mode: additive uniform noise of width step (standard practice
                   from Ballé et al. 2017) approximates quantisation
                   without killing gradients.

Ablation:
    use_adaptive=False → uniform quantisation (stage A5 baseline)
    use_adaptive=True  → attention-guided adaptive (stage A5 full)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from typing import Tuple


class AdaptiveQuantizer(nn.Module):
    """
    Attention-guided adaptive quantizer.

    Args:
        latent_dim   : channel dimension of the encoder latent (1024)
        base_step    : initial uniform step size (tuned per lambda)
        alpha        : attention sensitivity — higher = more aggressive
                       bit reallocation between regions
        use_adaptive : if False, falls back to uniform quantisation
                       (useful for ablation stage A1–A4)
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        base_step: float = 1.0,
        alpha: float = 2.0,
        use_adaptive: bool = True,
    ):
        super().__init__()

        self.use_adaptive = use_adaptive
        self.alpha        = alpha

        # Learned per-channel base step size.
        # Initialised to base_step, trained jointly with the codec.
        # Using log-space parameterisation ensures step > 0 at all times.
        self.log_base_step = nn.Parameter(
            torch.full((latent_dim, 1, 1), fill_value=torch.log(torch.tensor(base_step)))
        )

        if use_adaptive:
            # Small conv network: maps the SAG attention map (1ch) to a
            # per-spatial-location gain map (1ch), in [0,1].
            # This lets the model refine the raw SAG signal before using
            # it to modulate step sizes.
            self.attn_refine = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )

    def _compute_step_map(
        self,
        latent: torch.Tensor,
        attn_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the per-spatial-location step size tensor.

        Args:
            latent   : (B, C, H_l, W_l)
            attn_map : (B, 1, H_a, W_a) or None

        Returns:
            step_map : (B, C, H_l, W_l)
        """
        B, C, H_l, W_l = latent.shape
        base_step = torch.exp(self.log_base_step)    # (C, 1, 1) all positive

        if not self.use_adaptive or attn_map is None:
            # Uniform: same step everywhere
            return base_step.expand(B, C, H_l, W_l)

        # Resize attention map to match latent spatial dims if needed
        if attn_map.shape[-2:] != (H_l, W_l):
            attn_map = F.interpolate(
                attn_map,
                size=(H_l, W_l),
                mode="bilinear",
                align_corners=False,
            )
        # attn_map: (B, 1, H_l, W_l)

        # Optional learned refinement of the raw SAG map
        gain = self.attn_refine(attn_map)   # (B, 1, H_l, W_l)

        # step(x,y) = base_step * exp(−alpha * gain(x,y))
        # High gain → small step (fine) | Low gain → large step (coarse)
        step_map = base_step * torch.exp(-self.alpha * gain)
        # Broadcast base_step (C,1,1) across batch and spatial dims
        step_map = step_map.expand(B, C, H_l, W_l)

        return step_map

    def forward(
        self,
        latent: torch.Tensor,
        attn_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent   : (B, C, H, W)   encoder output
            attn_map : (B, 1, H_a, W_a) spatial attention from SAG, or None

        Returns:
            z_hat    : (B, C, H, W)   quantised latent (float, scaled back)
            step_map : (B, C, H, W)   step size map (passed to entropy model)
        """
        step_map = self._compute_step_map(latent, attn_map)

        if self.training:
            # Training: additive uniform noise approximates quantisation.
            # Gradients flow through cleanly (no zero-gradient problem).
            noise = torch.empty_like(latent).uniform_(-0.5, 0.5) * step_map
            z_hat = latent + noise
        else:
            # Inference: true rounding to integer multiples of step_map.
            z_hat = torch.round(latent / step_map) * step_map

        return z_hat, step_map


# Add missing Tuple import
from typing import Tuple   # noqa — kept at bottom to not disrupt reading flow


# ---------------------------------------------------------------------------
# Smoke test — python quantizer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    quantizer = AdaptiveQuantizer(
        latent_dim=1024,
        base_step=1.0,
        alpha=2.0,
        use_adaptive=True,
    )

    latent   = torch.randn(1, 1024, 16, 30)
    attn_map = torch.rand(1, 1, 16, 30)     # SAG output, values in [0,1]

    print(f"Latent in:   {list(latent.shape)}")

    # Training mode (noise-based)
    quantizer.train()
    z_hat_train, step_map = quantizer(latent, attn_map)
    print(f"z_hat train: {list(z_hat_train.shape)}")
    print(f"Step map:    {list(step_map.shape)}")
    print(f"Step range:  [{step_map.min().item():.4f}, {step_map.max().item():.4f}]")

    # Eval mode (true rounding)
    quantizer.eval()
    with torch.no_grad():
        z_hat_eval, _ = quantizer(latent, attn_map)
    print(f"z_hat eval:  {list(z_hat_eval.shape)}")
    print(f"Parameters:  {sum(p.numel() for p in quantizer.parameters()):,}")