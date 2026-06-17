import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_bpp_from_likelihoods(
    likelihoods: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]],
    target: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Computes likelihood-based BPP.

    BPP = sum(-log2(likelihoods)) / number_of_pixels

    This is the standard differentiable rate estimate used in learned
    image compression training.

    Args:
        likelihoods:
            Either a dict such as {"y": y_likelihoods, "z": z_likelihoods}
            or a single likelihood tensor.
        target:
            Original image tensor, shape (N, C, H, W).
        eps:
            Small clamp value to avoid log(0).

    Returns:
        Scalar tensor containing BPP.
    """
    device = target.device

    if likelihoods is None:
        return torch.zeros((), device=device)

    N, _, H, W = target.size()
    num_pixels = N * H * W

    bpp = torch.zeros((), device=device)

    if isinstance(likelihoods, dict):
        for likelihood in likelihoods.values():
            if likelihood is None:
                continue
            likelihood = likelihood.clamp(min=eps)
            bpp = bpp + torch.log(likelihood).sum() / (-math.log(2.0) * num_pixels)

    elif isinstance(likelihoods, torch.Tensor):
        likelihoods = likelihoods.clamp(min=eps)
        bpp = torch.log(likelihoods).sum() / (-math.log(2.0) * num_pixels)

    else:
        raise TypeError(
            f"Unsupported likelihoods type: {type(likelihoods)}. "
            "Expected dict, Tensor, or None."
        )

    return bpp


class ATICLoss(nn.Module):
    """
    ATIC training loss.

    Total Loss =
        lambda_rate * BPP
        + lambda_mse * MSE
        + lambda_ssim * (1 - SSIM)
        + lambda_lpips * LPIPS

    Note:
        This keeps your original interpretation:
            lambda_rate controls the strength of the rate penalty.

        Therefore:
            lower lambda_rate  -> weaker BPP penalty -> higher quality / higher BPP
            higher lambda_rate -> stronger BPP penalty -> lower BPP / lower quality
    """

    def __init__(
        self,
        lambda_rate: float = 0.01,
        lambda_mse: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_lpips: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()

        self.lambda_rate = lambda_rate
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.device = device

        self.lpips_fn = None
        self.ssim_fn = None

        # Differentiable LPIPS loss
        if self.lambda_lpips > 0:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
                self.lpips_fn.eval()

                # LPIPS network is used as a fixed perceptual feature extractor.
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False

            except ImportError:
                print("Warning: lpips library missing. LPIPS loss will be 0.")
                self.lpips_fn = None

        # Differentiable SSIM loss
        if self.lambda_ssim > 0:
            try:
                from piq import ssim

                self.ssim_fn = ssim
            except ImportError:
                print("Warning: piq library missing. SSIM loss will be 0.")
                self.ssim_fn = None

    def forward(self, output_dict, target):
        """
        Args:
            output_dict:
                Model output dictionary. Must contain:
                    output_dict["x_hat"]
                    output_dict["likelihoods"]
            target:
                Ground-truth image tensor, shape (N, 3, H, W), range [0, 1].

        Returns:
            Dictionary of total and component losses.
        """
        x_hat = output_dict["x_hat"]
        likelihoods = output_dict.get("likelihoods", None)

        # Keep loss range consistent.
        # If model.py already applies sigmoid, this clamp is just a safety guard.
        x_hat_clamped = torch.clamp(x_hat, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # 1. Distortion: MSE
        mse_loss = F.mse_loss(x_hat_clamped, target)

        # 2. Rate: likelihood-based BPP
        bpp_loss = compute_bpp_from_likelihoods(likelihoods, target)

        # 3. Structural loss: 1 - SSIM
        ssim_loss = torch.zeros((), device=target.device)
        if self.lambda_ssim > 0 and self.ssim_fn is not None:
            try:
                ssim_val = self.ssim_fn(
                    x_hat_clamped,
                    target,
                    data_range=1.0,
                )

                # piq.ssim usually returns a scalar, but this keeps it safe.
                if isinstance(ssim_val, torch.Tensor):
                    ssim_val = ssim_val.mean()

                ssim_loss = 1.0 - ssim_val

            except Exception as e:
                print(f"Warning: SSIM loss skipped due to error: {e}")
                ssim_loss = torch.zeros((), device=target.device)

        # 4. Perceptual loss: LPIPS
        lpips_loss = torch.zeros((), device=target.device)
        if self.lambda_lpips > 0 and self.lpips_fn is not None:
            try:
                # LPIPS expects image range [-1, 1].
                x_hat_norm = x_hat_clamped * 2.0 - 1.0
                target_norm = target * 2.0 - 1.0

                lpips_loss = self.lpips_fn(
                    x_hat_norm,
                    target_norm,
                ).mean()

            except Exception as e:
                print(f"Warning: LPIPS loss skipped due to error: {e}")
                lpips_loss = torch.zeros((), device=target.device)

        # 5. Total loss
        total_loss = (
            self.lambda_rate * bpp_loss
            + self.lambda_mse * mse_loss
            + self.lambda_ssim * ssim_loss
            + self.lambda_lpips * lpips_loss
        )

        return {
            "loss": total_loss,
            "bpp_loss": bpp_loss,
            "mse_loss": mse_loss,
            "ssim_loss": ssim_loss,
            "lpips_loss": lpips_loss,
        }