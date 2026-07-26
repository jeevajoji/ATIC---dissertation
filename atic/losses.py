"""Rate-distortion and decoder-synchronised attention distillation losses."""

import math
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Likelihoods = Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]


def _bpp_for_tensor(
    likelihood: Optional[torch.Tensor],
    *,
    num_pixels: int,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if likelihood is None:
        return reference.new_zeros(())
    return -torch.log2(likelihood.clamp(min=eps, max=1.0)).sum() / num_pixels


def compute_bpp_components(
    likelihoods: Likelihoods,
    target: torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total_bpp, y_bpp, z_bpp)`` from entropy likelihoods."""

    if target.ndim != 4:
        raise ValueError(
            "BPP requires a target with shape (N,C,H,W); "
            f"received {tuple(target.shape)}"
        )

    num_pixels = int(target.size(0) * target.size(2) * target.size(3))
    zero = target.new_zeros(())

    if likelihoods is None:
        return zero, zero, zero

    if isinstance(likelihoods, torch.Tensor):
        y_bpp = _bpp_for_tensor(
            likelihoods,
            num_pixels=num_pixels,
            reference=target,
            eps=eps,
        )
        return y_bpp, y_bpp, zero

    if not isinstance(likelihoods, dict):
        raise TypeError(
            f"Unsupported likelihoods type: {type(likelihoods)}. "
            "Expected dict, Tensor, or None."
        )

    components = {
        name: _bpp_for_tensor(
            likelihood,
            num_pixels=num_pixels,
            reference=target,
            eps=eps,
        )
        for name, likelihood in likelihoods.items()
        if likelihood is not None
    }
    total_bpp = sum(components.values(), zero)
    return (
        total_bpp,
        components.get("y", zero),
        components.get("z", zero),
    )


def compute_bpp_from_likelihoods(
    likelihoods: Likelihoods,
    target: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compatibility wrapper returning the complete likelihood-based BPP."""

    total_bpp, _, _ = compute_bpp_components(
        likelihoods,
        target,
        eps=eps,
    )
    return total_bpp


def _centered_log_map(gain_map: torch.Tensor, eps: float) -> torch.Tensor:
    """Remove each map's global log-gain (geometric-mean) component."""

    log_map = torch.log(gain_map.clamp(min=eps))
    return log_map - log_map.mean(dim=(-2, -1), keepdim=True)


class ATICLoss(nn.Module):
    """Standard learned-compression RD loss with optional DSAD.

    The base objective is

    ``total_bpp + lambda_rd * 255^2 * MSE``.

    DSAD compares spatially centred log gain maps, so it teaches the
    decoder-synchronised student *where* to allocate bits without supervising
    a global log-scale offset. This scale normalisation does not guarantee
    equal actual bitrate; the coded-file evaluation measures that separately.
    The encoder-attention teacher is detached inside this loss even if a caller
    forgets to detach it.

    ``lambda_rate`` is accepted only as a deprecated compatibility spelling
    for ``lambda_rd``.  It no longer weights BPP.
    """

    VALID_DSAD_LOSSES = {"smooth_l1", "l1"}

    def __init__(
        self,
        lambda_rd: Optional[float] = None,
        lambda_ssim: float = 0.0,
        lambda_lpips: float = 0.0,
        dsad_beta: float = 0.0,
        dsad_loss_type: str = "smooth_l1",
        device: str = "cuda",
        *,
        lambda_rate: Optional[float] = None,
        gain_eps: float = 1e-6,
    ):
        super().__init__()

        if lambda_rd is not None and lambda_rate is not None:
            raise ValueError("Pass lambda_rd, not both lambda_rd and lambda_rate")
        if lambda_rd is None:
            if lambda_rate is not None:
                warnings.warn(
                    "lambda_rate is deprecated and is interpreted as lambda_rd; "
                    "the standard objective weights distortion, not BPP.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                lambda_rd = lambda_rate
            else:
                lambda_rd = 0.01

        if not math.isfinite(lambda_rd) or lambda_rd < 0:
            raise ValueError("lambda_rd must be finite and non-negative")
        if (
            not math.isfinite(lambda_ssim)
            or not math.isfinite(lambda_lpips)
            or lambda_ssim < 0
            or lambda_lpips < 0
        ):
            raise ValueError(
                "Optional loss weights must be finite and non-negative"
            )
        if not math.isfinite(dsad_beta) or dsad_beta < 0:
            raise ValueError("dsad_beta must be finite and non-negative")
        if dsad_loss_type not in self.VALID_DSAD_LOSSES:
            raise ValueError(
                f"dsad_loss_type must be one of {sorted(self.VALID_DSAD_LOSSES)}"
            )
        if not math.isfinite(gain_eps) or gain_eps <= 0:
            raise ValueError("gain_eps must be finite and positive")

        self.lambda_rd = float(lambda_rd)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_lpips = float(lambda_lpips)
        self.dsad_beta = float(dsad_beta)
        self.dsad_loss_type = dsad_loss_type
        self.gain_eps = float(gain_eps)

        self.lpips_fn = None
        self.ssim_fn = None

        if self.lambda_lpips > 0:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    "lambda_lpips > 0 requires the optional lpips package"
                ) from exc

            self.lpips_fn = lpips.LPIPS(net="vgg").to(device)
            self.lpips_fn.eval()
            for param in self.lpips_fn.parameters():
                param.requires_grad = False

        if self.lambda_ssim > 0:
            try:
                from piq import ssim
            except ImportError as exc:
                raise ImportError(
                    "lambda_ssim > 0 requires the optional piq package"
                ) from exc
            self.ssim_fn = ssim

    def _dsad_metrics(
        self,
        student_map: Optional[torch.Tensor],
        teacher_map: Optional[torch.Tensor],
        reference: torch.Tensor,
        *,
        require_maps: bool,
        track_student_grad: bool,
    ) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())

        if student_map is None or teacher_map is None:
            if require_maps:
                missing = []
                if student_map is None:
                    missing.append("gain_map")
                if teacher_map is None:
                    missing.append("teacher_map")
                raise ValueError(
                    "Positive DSAD beta requires model output "
                    + " and ".join(missing)
                )

            return {
                "dsad_loss": zero,
                "dsad_mae": zero,
                "dsad_cosine": zero,
                "teacher_spatial_std": zero,
            }

        if student_map.shape != teacher_map.shape:
            raise ValueError(
                "DSAD student and teacher maps must have identical shapes; "
                f"received {tuple(student_map.shape)} and "
                f"{tuple(teacher_map.shape)}"
            )
        if student_map.ndim < 3:
            raise ValueError("DSAD gain maps must include two spatial dimensions")
        if not torch.isfinite(student_map).all():
            raise ValueError("DSAD student gain map contains non-finite values")
        if not torch.isfinite(teacher_map).all():
            raise ValueError("DSAD teacher gain map contains non-finite values")
        if not (student_map > 0).all():
            raise ValueError("DSAD student gain map must be strictly positive")
        if not (teacher_map > 0).all():
            raise ValueError("DSAD teacher gain map must be strictly positive")

        # Defensive detach is the causal boundary of the distillation design:
        # the teacher supervises the student but cannot optimise itself to make
        # the distillation target easier.
        teacher_map = teacher_map.detach()
        if not track_student_grad:
            student_map = student_map.detach()

        student_centered = _centered_log_map(student_map, self.gain_eps)
        teacher_centered = _centered_log_map(teacher_map, self.gain_eps)

        if self.dsad_loss_type == "smooth_l1":
            dsad_loss = F.smooth_l1_loss(student_centered, teacher_centered)
        else:
            dsad_loss = F.l1_loss(student_centered, teacher_centered)

        dsad_mae = F.l1_loss(student_centered, teacher_centered)
        student_flat = student_centered.flatten(start_dim=1)
        teacher_flat = teacher_centered.flatten(start_dim=1)
        dsad_cosine = F.cosine_similarity(
            student_flat,
            teacher_flat,
            dim=1,
            eps=self.gain_eps,
        ).mean()
        teacher_spatial_std = teacher_map.std(
            dim=(-2, -1),
            unbiased=False,
        ).mean()

        return {
            "dsad_loss": dsad_loss,
            "dsad_mae": dsad_mae,
            "dsad_cosine": dsad_cosine,
            "teacher_spatial_std": teacher_spatial_std,
        }

    def forward(
        self,
        output_dict: Dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute RD, optional perceptual, and optional DSAD components."""

        beta_value = self.dsad_beta if beta is None else float(beta)
        if not math.isfinite(beta_value) or beta_value < 0:
            raise ValueError("DSAD beta must be a finite, non-negative scalar")

        x_hat = output_dict["x_hat"]
        x_hat_clamped = torch.clamp(x_hat, 0.0, 1.0)
        target_clamped = torch.clamp(target, 0.0, 1.0)

        mse_loss = F.mse_loss(x_hat_clamped, target_clamped)
        total_bpp, y_bpp, z_bpp = compute_bpp_components(
            output_dict.get("likelihoods"),
            target_clamped,
        )
        scaled_mse_loss = self.lambda_rd * (255.0**2) * mse_loss

        ssim_loss = target_clamped.new_zeros(())
        if self.lambda_ssim > 0:
            ssim_val = self.ssim_fn(
                x_hat_clamped,
                target_clamped,
                data_range=1.0,
            )
            ssim_loss = 1.0 - ssim_val.mean()

        lpips_loss = target_clamped.new_zeros(())
        if self.lambda_lpips > 0:
            lpips_loss = self.lpips_fn(
                x_hat_clamped * 2.0 - 1.0,
                target_clamped * 2.0 - 1.0,
            ).mean()

        weighted_ssim_loss = self.lambda_ssim * ssim_loss
        weighted_lpips_loss = self.lambda_lpips * lpips_loss
        rd_loss = (
            total_bpp
            + scaled_mse_loss
            + weighted_ssim_loss
            + weighted_lpips_loss
        )

        # At beta zero DSAD is observational only. Detaching the student keeps
        # the exact base-loss graph and also permits outputs without a teacher.
        dsad = self._dsad_metrics(
            output_dict.get("gain_map"),
            output_dict.get("teacher_map"),
            target_clamped,
            require_maps=beta_value > 0,
            track_student_grad=beta_value > 0,
        )

        if beta_value == 0.0:
            weighted_dsad_loss = target_clamped.new_zeros(())
            total_loss = rd_loss
        else:
            weighted_dsad_loss = beta_value * dsad["dsad_loss"]
            total_loss = rd_loss + weighted_dsad_loss

        student_map = output_dict.get("gain_map")
        if student_map is None:
            student_gain_min = target_clamped.new_zeros(())
            student_gain_max = target_clamped.new_zeros(())
            student_gain_mean = target_clamped.new_zeros(())
            student_gain_geomean = target_clamped.new_zeros(())
            student_gain_std = target_clamped.new_zeros(())
        else:
            student_stats = student_map.detach()
            if not torch.isfinite(student_stats).all():
                raise ValueError("Student gain statistics require finite values")
            if not (student_stats > 0).all():
                raise ValueError(
                    "Student gain statistics require strictly positive values"
                )
            student_gain_min = student_stats.amin()
            student_gain_max = student_stats.amax()
            student_gain_mean = student_stats.mean()
            student_gain_geomean = torch.exp(
                torch.log(student_stats.clamp(min=self.gain_eps)).mean()
            )
            student_gain_std = student_stats.std(unbiased=False)

        latent_y = output_dict.get("latent_y")
        means_hat = output_dict.get("means_hat")
        if latent_y is None:
            latent_rms = target_clamped.new_zeros(())
            latent_abs_mean = target_clamped.new_zeros(())
            latent_symbol_zero_fraction = target_clamped.new_zeros(())
        else:
            latent_stats = latent_y.detach()
            if not torch.isfinite(latent_stats).all():
                raise ValueError("Latent diagnostics require finite values")
            latent_rms = latent_stats.square().mean().sqrt()
            latent_abs_mean = latent_stats.abs().mean()

            if means_hat is None:
                latent_symbol_zero_fraction = target_clamped.new_zeros(())
            else:
                means_stats = means_hat.detach()
                if means_stats.shape != latent_stats.shape:
                    raise ValueError(
                        "Latent means must match the encoder latent shape"
                    )
                gain_stats = (
                    torch.ones_like(latent_stats[:, :1])
                    if student_map is None
                    else student_map.detach()
                )
                if gain_stats.shape[-2:] != latent_stats.shape[-2:]:
                    raise ValueError(
                        "Latent gain diagnostics require matching spatial shape"
                    )
                centred_scaled = latent_stats * gain_stats - means_stats
                latent_symbol_zero_fraction = (
                    torch.round(centred_scaled) == 0
                ).to(dtype=latent_stats.dtype).mean()

        scales_hat = output_dict.get("scales_hat")
        if scales_hat is None:
            scale_min = target_clamped.new_zeros(())
            scale_mean = target_clamped.new_zeros(())
            scale_max = target_clamped.new_zeros(())
            scale_below_table_min_fraction = target_clamped.new_zeros(())
        else:
            scale_stats = scales_hat.detach()
            if not torch.isfinite(scale_stats).all():
                raise ValueError("Scale diagnostics require finite values")
            if not (scale_stats > 0).all():
                raise ValueError("Scale diagnostics require positive values")
            scale_min = scale_stats.amin()
            scale_mean = scale_stats.mean()
            scale_max = scale_stats.amax()
            # CompressAI's default Gaussian scale table begins at 0.11.
            scale_below_table_min_fraction = (
                scale_stats < 0.11
            ).to(dtype=scale_stats.dtype).mean()

        return {
            "loss": total_loss,
            "rd_loss": rd_loss,
            "bpp_loss": total_bpp,
            "total_bpp": total_bpp,
            "y_bpp": y_bpp,
            "z_bpp": z_bpp,
            "mse_loss": mse_loss,
            "scaled_mse_loss": scaled_mse_loss,
            "ssim_loss": ssim_loss,
            "weighted_ssim_loss": weighted_ssim_loss,
            "lpips_loss": lpips_loss,
            "weighted_lpips_loss": weighted_lpips_loss,
            "dsad_loss": dsad["dsad_loss"],
            "weighted_dsad_loss": weighted_dsad_loss,
            "dsad_mae": dsad["dsad_mae"],
            "dsad_cosine": dsad["dsad_cosine"],
            "beta": target_clamped.new_tensor(beta_value),
            "student_gain_min": student_gain_min,
            "student_gain_max": student_gain_max,
            "student_gain_mean": student_gain_mean,
            "student_gain_geomean": student_gain_geomean,
            "student_gain_std": student_gain_std,
            "teacher_spatial_std": dsad["teacher_spatial_std"],
            "latent_rms": latent_rms,
            "latent_abs_mean": latent_abs_mean,
            "latent_symbol_zero_fraction": latent_symbol_zero_fraction,
            "scale_min": scale_min,
            "scale_mean": scale_mean,
            "scale_max": scale_max,
            "scale_below_table_min_fraction": (
                scale_below_table_min_fraction
            ),
        }
