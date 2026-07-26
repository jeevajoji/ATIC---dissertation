import json
import math
import os
import warnings
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from atic.bitstream import sha256_file
from atic.losses import ATICLoss


LOSS_LOG_KEYS = (
    "loss",
    "rd_loss",
    "bpp_loss",
    "total_bpp",
    "y_bpp",
    "z_bpp",
    "mse_loss",
    "scaled_mse_loss",
    "ssim_loss",
    "weighted_ssim_loss",
    "lpips_loss",
    "weighted_lpips_loss",
    "dsad_loss",
    "weighted_dsad_loss",
    "dsad_mae",
    "dsad_cosine",
    "beta",
    "student_gain_min",
    "student_gain_max",
    "student_gain_mean",
    "student_gain_geomean",
    "student_gain_std",
    "teacher_spatial_std",
)
TRAIN_GRADIENT_LOG_KEYS = (
    "main_grad_norm",
    "main_grad_norm_max",
    "main_grad_clip_fraction",
    "aux_grad_norm",
    "aux_grad_norm_max",
)
VALIDATION_MIN_KEYS = {"student_gain_min"}
VALIDATION_MAX_KEYS = {"student_gain_max"}


def dsad_beta_for_epoch(
    epoch: int,
    total_epochs: int,
    beta_max: float,
    warmup_fraction: float = 0.20,
    ramp_fraction: float = 0.10,
) -> float:
    """Return the DSAD weight for a zero-based epoch.

    The first ``warmup_fraction`` of epochs use zero DSAD weight, the next
    ``ramp_fraction`` linearly ramp to ``beta_max``, and all remaining epochs
    use ``beta_max``. A one-epoch pilot uses ``beta_max`` so it actually
    exercises DSAD rather than becoming an accidental no-DSAD experiment.
    """

    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if epoch < 0 or epoch >= total_epochs:
        raise ValueError(
            f"epoch must be in [0, {total_epochs}); received {epoch}"
        )
    if not math.isfinite(beta_max) or beta_max < 0:
        raise ValueError("beta_max must be a finite, non-negative scalar")
    if (
        not math.isfinite(warmup_fraction)
        or not math.isfinite(ramp_fraction)
        or warmup_fraction < 0
        or ramp_fraction < 0
        or warmup_fraction + ramp_fraction > 1
    ):
        raise ValueError(
            "DSAD warmup/ramp fractions must be finite, non-negative, "
            "and sum to at most 1"
        )
    if beta_max == 0:
        return float(beta_max)
    if total_epochs == 1:
        return float(beta_max)

    # Allocate whole epochs while reserving room for the positive-DSAD phase.
    # A non-zero warm-up always receives at least one epoch in a multi-epoch
    # run, which makes the documented two-epoch pilot [0, beta_max].  Capping
    # the ramp by the remaining epochs prevents rounded counts from overflowing
    # short runs and guarantees that the last epoch reaches beta_max.
    warmup_epochs = 0
    if warmup_fraction > 0:
        warmup_epochs = max(1, math.floor(total_epochs * warmup_fraction))
        warmup_epochs = min(warmup_epochs, total_epochs - 1)

    if epoch < warmup_epochs:
        return 0.0

    if ramp_fraction == 0:
        return float(beta_max)

    remaining_epochs = total_epochs - warmup_epochs
    ramp_epochs = max(1, math.ceil(total_epochs * ramp_fraction))
    ramp_epochs = min(ramp_epochs, remaining_epochs)
    ramp_step = epoch - warmup_epochs + 1
    return float(beta_max) * min(ramp_step / ramp_epochs, 1.0)


def _append_jsonl(file_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _extract_images(batch):
    """
    Supports both:
        Dataset returning image only:
            batch = tensor

        Dataset returning image and path:
            batch = (tensor, path)
    """
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


@torch.no_grad()
def save_reconstruction_comparison(
    model,
    dataloader,
    device: str,
    save_path: str,
    title: str = "Reconstruction",
):
    """
    Saves original vs reconstructed image comparison.

    Used after each epoch to visually monitor reconstruction progress.

    This function saves:
        - epoch-specific preview image
        - latest/final reconstruction image, depending on caller path
    """
    if dataloader is None:
        return

    import matplotlib.pyplot as plt

    model.eval()

    try:
        batch = next(iter(dataloader))
        batch = _extract_images(batch).to(device, non_blocking=True)

        outputs = model(batch)
        x_hat = outputs["x_hat"]

        x_orig = (
            batch[0]
            .detach()
            .cpu()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )

        x_recon = (
            x_hat[0]
            .detach()
            .cpu()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(x_orig)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(x_recon)
        axes[1].set_title(title)
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved reconstruction preview: {save_path}")

    except Exception as e:
        print(f"Reconstruction preview skipped: {e}")

    model.train()


def configure_optimizers(
    model,
    learning_rate: float = 1e-4,
    aux_learning_rate: float = 1e-3,
):
    """
    CompressAI-style optimizer setup.

    Main optimizer:
        Updates normal model parameters.

    Aux optimizer:
        Updates entropy bottleneck quantile parameters.
        These usually have names ending with ".quantiles".
    """
    parameters = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not name.endswith(".quantiles")
    }

    aux_parameters = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name.endswith(".quantiles")
    }

    params_dict = dict(model.named_parameters())

    optimizer = optim.Adam(
        (params_dict[name] for name in sorted(parameters)),
        lr=learning_rate,
    )

    aux_optimizer = None
    if len(aux_parameters) > 0:
        aux_optimizer = optim.Adam(
            (params_dict[name] for name in sorted(aux_parameters)),
            lr=aux_learning_rate,
        )

    return optimizer, aux_optimizer


def _optimizer_parameters(optimizer) -> tuple:
    """Return each parameter owned by ``optimizer`` exactly once."""

    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in seen:
                parameters.append(parameter)
                seen.add(identity)
    return tuple(parameters)


def _clip_optimizer_gradients(
    optimizer,
    *,
    max_norm: float,
) -> float:
    """Clip only gradients owned by one optimizer and return their old norm."""

    parameters = _optimizer_parameters(optimizer)
    if not parameters:
        return 0.0
    return float(
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
        )
    )


def _cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    """Clone a model state to CPU for validation-based checkpoint selection."""

    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _validate_training_controls(
    *,
    learning_rate: float,
    aux_learning_rate: float,
    lr_schedule: str,
    min_learning_rate: float,
    min_aux_learning_rate: float,
    grad_clip_norm: Optional[float],
    checkpoint_selection: str,
    checkpoint_selection_start_epoch: int,
    epochs: int,
) -> None:
    rates = {
        "learning_rate": learning_rate,
        "aux_learning_rate": aux_learning_rate,
        "min_learning_rate": min_learning_rate,
        "min_aux_learning_rate": min_aux_learning_rate,
    }
    for name, value in rates.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if learning_rate <= 0 or aux_learning_rate <= 0:
        raise ValueError("initial learning rates must be positive")
    if min_learning_rate > learning_rate:
        raise ValueError("min_learning_rate must not exceed learning_rate")
    if min_aux_learning_rate > aux_learning_rate:
        raise ValueError(
            "min_aux_learning_rate must not exceed aux_learning_rate"
        )
    if (
        grad_clip_norm is not None
        and (
            not math.isfinite(grad_clip_norm)
            or grad_clip_norm < 0
        )
    ):
        raise ValueError(
            "grad_clip_norm must be finite and non-negative, or None"
        )
    if lr_schedule not in {"none", "cosine"}:
        raise ValueError("lr_schedule must be 'none' or 'cosine'")
    if checkpoint_selection not in {"final", "best_val_rd"}:
        raise ValueError(
            "checkpoint_selection must be 'final' or 'best_val_rd'"
        )
    if (
        not isinstance(checkpoint_selection_start_epoch, int)
        or isinstance(checkpoint_selection_start_epoch, bool)
        or not 1 <= checkpoint_selection_start_epoch <= epochs
    ):
        raise ValueError(
            "checkpoint_selection_start_epoch must be an integer in "
            f"[1, {epochs}]"
        )


@torch.no_grad()
def validate_loop(
    model,
    dataloader,
    criterion,
    device: str = "cuda",
    beta: float = 0.0,
) -> Optional[Dict[str, float]]:
    """
    Optional validation loop.

    Returns averaged validation losses.
    If val_loader is None, returns None.
    """
    if dataloader is None:
        return None

    model.eval()

    totals = {
        f"val_{key}": (
            math.inf
            if key in VALIDATION_MIN_KEYS
            else -math.inf
            if key in VALIDATION_MAX_KEYS
            else 0.0
        )
        for key in LOSS_LOG_KEYS
    }
    steps = 0

    for batch in dataloader:
        batch = _extract_images(batch).to(device, non_blocking=True)

        outputs = model(batch)
        loss_dict = criterion(outputs, batch, beta=beta)

        for key in LOSS_LOG_KEYS:
            name = f"val_{key}"
            value = float(loss_dict[key].item())
            if key in VALIDATION_MIN_KEYS:
                totals[name] = min(totals[name], value)
            elif key in VALIDATION_MAX_KEYS:
                totals[name] = max(totals[name], value)
            else:
                totals[name] += value

        steps += 1

    model.train()

    if steps == 0:
        return None

    return {
        key: (
            value
            if key.removeprefix("val_") in (
                VALIDATION_MIN_KEYS | VALIDATION_MAX_KEYS
            )
            else value / steps
        )
        for key, value in totals.items()
    }


def train_loop(
    model,
    variant_name,
    dataloader,
    epochs: int = 5,
    device: str = "cuda",
    lambda_rd: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    train_log_path: Optional[str] = None,
    val_loader=None,
    learning_rate: float = 1e-4,
    aux_learning_rate: float = 1e-3,
    lr_schedule: str = "none",
    min_learning_rate: float = 1e-6,
    min_aux_learning_rate: float = 1e-5,
    grad_clip_norm: float = 1.0,
    checkpoint_selection: str = "final",
    checkpoint_selection_start_epoch: int = 1,
    reconstruction_path: Optional[str] = None,
    save_reconstruction_each_epoch: bool = True,
    dsad_beta_max: float = 0.0,
    dsad_warmup_fraction: float = 0.20,
    dsad_ramp_fraction: float = 0.10,
    dsad_loss_type: str = "smooth_l1",
    lambda_ssim: float = 0.0,
    lambda_lpips: float = 0.0,
    *,
    lambda_rate: Optional[float] = None,
):
    """
    Trains ATIC model.

    Compatible with CompressAI entropy modules:
        - uses main optimizer
        - uses auxiliary optimizer for entropy bottleneck quantiles
        - calls model.aux_loss() when available
        - calls model.update(force=True) before saving when available

    Reconstruction previews:
        If save_reconstruction_each_epoch=True, this saves:
            run_dir/epoch_previews/epoch_001.png
            run_dir/epoch_previews/epoch_002.png
            ...
            run_dir/reconstruction.png

        reconstruction.png is overwritten every epoch and therefore always
        contains the latest/final reconstruction preview.

    Args:
        model:
            ATICModel.
        variant_name:
            Name used for logs/checkpoints.
        dataloader:
            Training DataLoader.
        epochs:
            Number of epochs.
        device:
            cuda or cpu.
        lambda_rd:
            Weight on ``255^2 * MSE`` in the standard learned-compression
            objective. Defaults to 0.01.
        checkpoint_path:
            Where to save final model.state_dict().
        train_log_path:
            Optional JSONL log path.
        val_loader:
            Optional validation loader.
        learning_rate:
            Main optimizer LR.
        aux_learning_rate:
            Entropy bottleneck auxiliary LR.
        lr_schedule:
            ``none`` or a fixed ``cosine`` schedule. The cosine schedule is
            deterministic and therefore identical across paired ablation arms.
        min_learning_rate:
            Final main-optimizer LR used by the cosine schedule.
        min_aux_learning_rate:
            Final auxiliary-optimizer LR used by the cosine schedule.
        grad_clip_norm:
            Gradient clipping threshold.
        checkpoint_selection:
            ``final`` or ``best_val_rd``. The latter restores the checkpoint
            with the lowest validation rate-distortion loss before entropy
            table update and actual-bitstream evaluation.
        checkpoint_selection_start_epoch:
            First one-based epoch eligible for best-validation selection.
            Paired DSAD studies use the first epoch at full beta so an early
            beta-zero warm-up checkpoint cannot masquerade as a DSAD result.
        reconstruction_path:
            Path to latest/final reconstruction image.
        save_reconstruction_each_epoch:
            Whether to save reconstruction preview after every epoch.
        dsad_beta_max:
            Maximum decoder-synchronised attention distillation weight.
        dsad_warmup_fraction:
            Initial fraction of epochs with DSAD disabled.
        dsad_ramp_fraction:
            Following fraction of epochs used to linearly ramp DSAD.
        lambda_rate:
            Deprecated compatibility spelling for ``lambda_rd``. It does not
            weight BPP.

    Returns:
        {
            "history": [...],
            "checkpoint_path": checkpoint_path
        }
    """
    if dataloader is None:
        print(f"[{variant_name}] Dataloader not found. Skipping training.")
        return {"history": [], "checkpoint_path": None}

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    _validate_training_controls(
        learning_rate=learning_rate,
        aux_learning_rate=aux_learning_rate,
        lr_schedule=lr_schedule,
        min_learning_rate=min_learning_rate,
        min_aux_learning_rate=min_aux_learning_rate,
        grad_clip_norm=grad_clip_norm,
        checkpoint_selection=checkpoint_selection,
        checkpoint_selection_start_epoch=checkpoint_selection_start_epoch,
        epochs=epochs,
    )
    if checkpoint_selection == "best_val_rd" and val_loader is None:
        raise ValueError(
            "best_val_rd checkpoint selection requires a validation loader"
        )
    if lambda_rd is not None and lambda_rate is not None:
        raise ValueError("Pass lambda_rd, not both lambda_rd and lambda_rate")
    if lambda_rd is None:
        if lambda_rate is not None:
            warnings.warn(
                "train_loop(lambda_rate=...) is deprecated; the value is now "
                "interpreted as lambda_rd and weights 255^2*MSE, not BPP.",
                DeprecationWarning,
                stacklevel=2,
            )
            lambda_rd = lambda_rate
        else:
            lambda_rd = 0.01

    # Validate the complete schedule before allocating model/optimizer state.
    for schedule_epoch in range(epochs):
        dsad_beta_for_epoch(
            schedule_epoch,
            epochs,
            dsad_beta_max,
            dsad_warmup_fraction,
            dsad_ramp_fraction,
        )

    model.to(device)
    model.train()

    optimizer, aux_optimizer = configure_optimizers(
        model,
        learning_rate=learning_rate,
        aux_learning_rate=aux_learning_rate,
    )
    aux_parameters = (
        ()
        if aux_optimizer is None
        else _optimizer_parameters(aux_optimizer)
    )
    scheduler = None
    aux_scheduler = None
    if lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_learning_rate,
        )
        if aux_optimizer is not None:
            aux_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                aux_optimizer,
                T_max=epochs,
                eta_min=min_aux_learning_rate,
            )

    criterion = ATICLoss(
        lambda_rd=lambda_rd,
        lambda_ssim=lambda_ssim,
        lambda_lpips=lambda_lpips,
        dsad_beta=0.0,
        dsad_loss_type=dsad_loss_type,
        device=device,
    )

    print(
        f"[{variant_name}] Training started on {device} | "
        f"lambda_rd={lambda_rd} | dsad_beta_max={dsad_beta_max} | "
        f"lr={learning_rate} | aux_lr={aux_learning_rate} | "
        f"grad_clip_norm={grad_clip_norm} | "
        f"lr_schedule={lr_schedule} | "
        f"checkpoint_selection={checkpoint_selection}"
    )

    if aux_optimizer is None:
        print(f"[{variant_name}] No auxiliary entropy parameters found.")

    history = []
    best_state_dict = None
    best_val_rd_loss = math.inf
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_aux_learning_rate = (
            None
            if aux_optimizer is None
            else float(aux_optimizer.param_groups[0]["lr"])
        )
        epoch_beta = dsad_beta_for_epoch(
            epoch,
            epochs,
            dsad_beta_max,
            dsad_warmup_fraction,
            dsad_ramp_fraction,
        )

        epoch_totals = {key: 0.0 for key in LOSS_LOG_KEYS}
        epoch_totals["aux_loss"] = 0.0
        main_grad_norm_total = 0.0
        main_grad_norm_max = 0.0
        main_grad_clipped_steps = 0
        aux_grad_norm_total = 0.0
        aux_grad_norm_max = 0.0
        aux_grad_steps = 0
        epoch_steps = 0

        pbar = tqdm(
            dataloader,
            desc=f"{variant_name} | Epoch [{epoch + 1}/{epochs}]",
        )

        for batch_idx, batch in enumerate(pbar):
            batch = _extract_images(batch).to(device, non_blocking=True)

            # -------------------------
            # Main model update
            # -------------------------
            optimizer.zero_grad(set_to_none=True)
            if aux_optimizer is not None:
                # Quantile parameters belong only to the auxiliary optimizer.
                # Clear them before the main backward pass so no gradient from
                # an earlier auxiliary step can survive into this batch.
                aux_optimizer.zero_grad(set_to_none=True)

            outputs = model(batch)
            loss_dict = criterion(outputs, batch, beta=epoch_beta)
            loss = loss_dict["loss"]

            loss.backward()

            main_clip_limit = (
                float(grad_clip_norm)
                if grad_clip_norm is not None and grad_clip_norm > 0
                else math.inf
            )
            # Do not pass model.parameters() here. CompressAI quantiles are
            # updated by a separate optimizer and historically retained their
            # large auxiliary gradients until the next batch, corrupting the
            # global norm used to scale the main codec gradients.
            main_grad_norm = _clip_optimizer_gradients(
                optimizer,
                max_norm=main_clip_limit,
            )
            if not math.isfinite(main_grad_norm):
                raise RuntimeError("Main gradient norm became non-finite")
            main_grad_norm_total += main_grad_norm
            main_grad_norm_max = max(main_grad_norm_max, main_grad_norm)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                main_grad_clipped_steps += int(
                    main_grad_norm > float(grad_clip_norm)
                )

            optimizer.step()

            # -------------------------
            # CompressAI auxiliary update
            # -------------------------
            aux_loss_value = 0.0

            if aux_optimizer is not None and hasattr(model, "aux_loss"):
                aux_optimizer.zero_grad(set_to_none=True)

                aux_loss = model.aux_loss()
                aux_loss.backward()
                aux_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        aux_parameters,
                        max_norm=math.inf,
                    )
                )
                if not math.isfinite(aux_grad_norm):
                    raise RuntimeError(
                        "Auxiliary entropy gradient norm became non-finite"
                    )
                aux_grad_norm_total += aux_grad_norm
                aux_grad_norm_max = max(aux_grad_norm_max, aux_grad_norm)
                aux_grad_steps += 1
                aux_optimizer.step()
                # Keep the optimizer boundary explicit after the step as well
                # as at the start of the next batch.
                aux_optimizer.zero_grad(set_to_none=True)

                aux_loss_value = float(aux_loss.item())

            # -------------------------
            # Logging
            # -------------------------
            for key in LOSS_LOG_KEYS:
                epoch_totals[key] += float(loss_dict[key].item())
            epoch_totals["aux_loss"] += aux_loss_value
            epoch_steps += 1

            pbar.set_postfix(
                {
                    "Loss": f"{loss_dict['loss'].item():.4f}",
                    "BPP": f"{loss_dict['total_bpp'].item():.4f}",
                    "y": f"{loss_dict['y_bpp'].item():.3f}",
                    "z": f"{loss_dict['z_bpp'].item():.3f}",
                    "MSE": f"{loss_dict['mse_loss'].item():.6f}",
                    "DSAD": f"{loss_dict['dsad_loss'].item():.4f}",
                    "beta": f"{epoch_beta:.4g}",
                    "Aux": f"{aux_loss_value:.4f}",
                    "Grad": f"{main_grad_norm:.2f}",
                }
            )

        # -------------------------
        # Epoch averages
        # -------------------------
        if epoch_steps > 0:
            epoch_avg = {
                "epoch": epoch + 1,
                "steps": epoch_steps,
                "aux_loss": epoch_totals["aux_loss"] / epoch_steps,
                "lambda_rd": lambda_rd,
                "dsad_beta_max": dsad_beta_max,
                "variant": variant_name,
                "learning_rate": epoch_learning_rate,
                "aux_learning_rate": epoch_aux_learning_rate,
            }
            epoch_avg.update(
                {
                    key: epoch_totals[key] / epoch_steps
                    for key in LOSS_LOG_KEYS
                }
            )
            epoch_avg.update(
                {
                    "main_grad_norm": main_grad_norm_total / epoch_steps,
                    "main_grad_norm_max": main_grad_norm_max,
                    "main_grad_clip_fraction": (
                        main_grad_clipped_steps / epoch_steps
                    ),
                    "aux_grad_norm": (
                        aux_grad_norm_total / aux_grad_steps
                        if aux_grad_steps > 0
                        else 0.0
                    ),
                    "aux_grad_norm_max": aux_grad_norm_max,
                }
            )

            # Optional validation
            val_avg = validate_loop(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                beta=epoch_beta,
            )
            if val_avg is not None:
                epoch_avg.update(val_avg)

            selection_eligible = (
                checkpoint_selection == "best_val_rd"
                and val_avg is not None
                and epoch + 1 >= checkpoint_selection_start_epoch
            )
            selected_this_epoch = False
            if selection_eligible:
                val_rd_loss = float(epoch_avg["val_rd_loss"])
                if not math.isfinite(val_rd_loss):
                    raise RuntimeError(
                        "Validation RD loss became non-finite at "
                        f"epoch {epoch + 1}"
                    )
                if val_rd_loss < best_val_rd_loss:
                    best_state_dict = _cpu_state_dict(model)
                    best_val_rd_loss = val_rd_loss
                    best_epoch = epoch + 1
                    selected_this_epoch = True
            epoch_avg.update(
                {
                    "checkpoint_selection_eligible": selection_eligible,
                    "selected_as_best": selected_this_epoch,
                    "best_epoch_so_far": best_epoch,
                    "best_val_rd_loss_so_far": (
                        None if best_epoch is None else best_val_rd_loss
                    ),
                }
            )

            history.append(epoch_avg)

            if train_log_path is not None:
                _append_jsonl(train_log_path, epoch_avg)

            val_msg = ""
            if val_avg is not None:
                val_msg = (
                    f" | Val Loss: {epoch_avg['val_loss']:.4f}"
                    f" | Val BPP: {epoch_avg['val_total_bpp']:.4f}"
                    f" | Val MSE: {epoch_avg['val_mse_loss']:.6f}"
                    f" | Val DSAD: {epoch_avg['val_dsad_loss']:.4f}"
                )

            print(
                f"[{variant_name}] Epoch {epoch + 1}/{epochs}"
                f" | Loss: {epoch_avg['loss']:.4f}"
                f" | BPP: {epoch_avg['total_bpp']:.4f}"
                f" (y={epoch_avg['y_bpp']:.4f}, z={epoch_avg['z_bpp']:.4f})"
                f" | MSE: {epoch_avg['mse_loss']:.6f}"
                f" | DSAD: {epoch_avg['dsad_loss']:.4f}"
                f" (weighted={epoch_avg['weighted_dsad_loss']:.4f},"
                f" beta={epoch_avg['beta']:.4g})"
                f" | Aux: {epoch_avg['aux_loss']:.4f}"
                f" | Main Grad: {epoch_avg['main_grad_norm']:.3g}"
                f" (clipped={epoch_avg['main_grad_clip_fraction']:.1%})"
                f" | Aux Grad: {epoch_avg['aux_grad_norm']:.3g}"
                f" | LR: {epoch_learning_rate:.3g}"
                f"{val_msg}"
            )

            # -------------------------------------------------
            # Save reconstruction preview after every epoch
            # -------------------------------------------------
            if save_reconstruction_each_epoch:
                preview_loader = val_loader if val_loader is not None else dataloader

                if checkpoint_path is not None:
                    run_dir = os.path.dirname(checkpoint_path)
                else:
                    run_dir = "ablation_results"

                # 1. Save epoch-specific preview for inspection
                epoch_preview_path = os.path.join(
                    run_dir,
                    "epoch_previews",
                    f"epoch_{epoch + 1:03d}.png",
                )

                save_reconstruction_comparison(
                    model=model,
                    dataloader=preview_loader,
                    device=device,
                    save_path=epoch_preview_path,
                    title=f"{variant_name} | epoch {epoch + 1}",
                )

                # 2. Also overwrite latest/final reconstruction.png
                latest_preview_path = reconstruction_path or os.path.join(
                    run_dir,
                    "reconstruction.png",
                )

                save_reconstruction_comparison(
                    model=model,
                    dataloader=preview_loader,
                    device=device,
                    save_path=latest_preview_path,
                    title=f"{variant_name} | latest epoch {epoch + 1}",
                )

        if scheduler is not None:
            scheduler.step()
        if aux_scheduler is not None:
            aux_scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_epoch = epochs
    selected_val_rd_loss = None
    if checkpoint_selection == "best_val_rd":
        if best_state_dict is None or best_epoch is None:
            raise RuntimeError(
                "No validation checkpoint was eligible for selection"
            )
        model.load_state_dict(best_state_dict)
        selected_epoch = best_epoch
        selected_val_rd_loss = best_val_rd_loss
        print(
            f"[{variant_name}] Restored best validation-RD checkpoint "
            f"from epoch {selected_epoch} "
            f"(val_rd_loss={selected_val_rd_loss:.6f})."
        )

    # Update entropy CDF tables if model supports it.
    # This is useful before final evaluation/checkpointing.
    if hasattr(model, "update"):
        try:
            model.update(force=True)
            print(f"[{variant_name}] Entropy model updated.")
        except Exception as e:
            print(f"[{variant_name}] Warning: model.update(force=True) failed: {e}")

    if checkpoint_path is None:
        checkpoint_path = f"ablation_results/{variant_name}.pth"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Keep this as state_dict for compatibility with your old code.
    torch.save(model.state_dict(), checkpoint_path)
    if hasattr(model, "set_model_id"):
        # Bind any subsequently produced .atic files to this exact checkpoint.
        model.set_model_id(sha256_file(checkpoint_path))

    print(f"[{variant_name}] Saved checkpoint to {checkpoint_path}.")

    selection_path = os.path.join(
        os.path.dirname(checkpoint_path),
        "checkpoint_selection.json",
    )
    with open(selection_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "strategy": checkpoint_selection,
                "selection_metric": (
                    "val_rd_loss"
                    if checkpoint_selection == "best_val_rd"
                    else None
                ),
                "selection_start_epoch": checkpoint_selection_start_epoch,
                "selected_epoch": selected_epoch,
                "selected_val_rd_loss": selected_val_rd_loss,
                "total_epochs": epochs,
                "lr_schedule": lr_schedule,
                "initial_learning_rate": learning_rate,
                "minimum_learning_rate": min_learning_rate,
                "initial_aux_learning_rate": aux_learning_rate,
                "minimum_aux_learning_rate": min_aux_learning_rate,
                "grad_clip_norm": grad_clip_norm,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    return {
        "history": history,
        "checkpoint_path": checkpoint_path,
        "checkpoint_selection_path": selection_path,
        "selected_epoch": selected_epoch,
        "selected_val_rd_loss": selected_val_rd_loss,
    }
