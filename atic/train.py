import json
import os
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from atic.losses import ATICLoss


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


@torch.no_grad()
def validate_loop(
    model,
    dataloader,
    criterion,
    device: str = "cuda",
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
        "val_loss": 0.0,
        "val_bpp_loss": 0.0,
        "val_mse_loss": 0.0,
        "val_ssim_loss": 0.0,
        "val_lpips_loss": 0.0,
    }
    steps = 0

    for batch in dataloader:
        batch = _extract_images(batch).to(device, non_blocking=True)

        outputs = model(batch)
        loss_dict = criterion(outputs, batch)

        totals["val_loss"] += float(loss_dict["loss"].item())
        totals["val_bpp_loss"] += float(loss_dict["bpp_loss"].item())
        totals["val_mse_loss"] += float(loss_dict["mse_loss"].item())
        totals["val_ssim_loss"] += float(loss_dict["ssim_loss"].item())
        totals["val_lpips_loss"] += float(loss_dict["lpips_loss"].item())

        steps += 1

    model.train()

    if steps == 0:
        return None

    return {k: v / steps for k, v in totals.items()}


def train_loop(
    model,
    variant_name,
    dataloader,
    epochs: int = 5,
    device: str = "cuda",
    lambda_rate: float = 0.01,
    checkpoint_path: Optional[str] = None,
    train_log_path: Optional[str] = None,
    val_loader=None,
    learning_rate: float = 1e-4,
    aux_learning_rate: float = 1e-3,
    grad_clip_norm: float = 1.0,
    reconstruction_path: Optional[str] = None,
    save_reconstruction_each_epoch: bool = True,
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
        lambda_rate:
            Weight for BPP term in ATICLoss.
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
        grad_clip_norm:
            Gradient clipping threshold.
        reconstruction_path:
            Path to latest/final reconstruction image.
        save_reconstruction_each_epoch:
            Whether to save reconstruction preview after every epoch.

    Returns:
        {
            "history": [...],
            "checkpoint_path": checkpoint_path
        }
    """
    if dataloader is None:
        print(f"[{variant_name}] Dataloader not found. Skipping training.")
        return {"history": [], "checkpoint_path": None}

    model.to(device)
    model.train()

    optimizer, aux_optimizer = configure_optimizers(
        model,
        learning_rate=learning_rate,
        aux_learning_rate=aux_learning_rate,
    )

    criterion = ATICLoss(
        lambda_rate=lambda_rate,
        lambda_mse=1.0,
        lambda_ssim=0.5,
        lambda_lpips=0.1,
        device=device,
    )

    print(
        f"[{variant_name}] Training started on {device} | "
        f"lambda_rate={lambda_rate} | "
        f"lr={learning_rate} | aux_lr={aux_learning_rate}"
    )

    if aux_optimizer is None:
        print(f"[{variant_name}] No auxiliary entropy parameters found.")

    history = []

    for epoch in range(epochs):
        model.train()

        epoch_totals = {
            "loss": 0.0,
            "bpp_loss": 0.0,
            "mse_loss": 0.0,
            "ssim_loss": 0.0,
            "lpips_loss": 0.0,
            "aux_loss": 0.0,
        }
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

            outputs = model(batch)
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]

            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm,
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
                aux_optimizer.step()

                aux_loss_value = float(aux_loss.item())

            # -------------------------
            # Logging
            # -------------------------
            epoch_totals["loss"] += float(loss_dict["loss"].item())
            epoch_totals["bpp_loss"] += float(loss_dict["bpp_loss"].item())
            epoch_totals["mse_loss"] += float(loss_dict["mse_loss"].item())
            epoch_totals["ssim_loss"] += float(loss_dict["ssim_loss"].item())
            epoch_totals["lpips_loss"] += float(loss_dict["lpips_loss"].item())
            epoch_totals["aux_loss"] += aux_loss_value
            epoch_steps += 1

            pbar.set_postfix(
                {
                    "Loss": f"{loss_dict['loss'].item():.4f}",
                    "BPP": f"{loss_dict['bpp_loss'].item():.4f}",
                    "MSE": f"{loss_dict['mse_loss'].item():.6f}",
                    "SSIM_L": f"{loss_dict['ssim_loss'].item():.4f}",
                    "LPIPS": f"{loss_dict['lpips_loss'].item():.4f}",
                    "Aux": f"{aux_loss_value:.4f}",
                }
            )

        # -------------------------
        # Epoch averages
        # -------------------------
        if epoch_steps > 0:
            epoch_avg = {
                "epoch": epoch + 1,
                "steps": epoch_steps,
                "loss": epoch_totals["loss"] / epoch_steps,
                "bpp_loss": epoch_totals["bpp_loss"] / epoch_steps,
                "mse_loss": epoch_totals["mse_loss"] / epoch_steps,
                "ssim_loss": epoch_totals["ssim_loss"] / epoch_steps,
                "lpips_loss": epoch_totals["lpips_loss"] / epoch_steps,
                "aux_loss": epoch_totals["aux_loss"] / epoch_steps,
                "lambda_rate": lambda_rate,
                "variant": variant_name,
            }

            # Optional validation
            val_avg = validate_loop(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
            if val_avg is not None:
                epoch_avg.update(val_avg)

            history.append(epoch_avg)

            if train_log_path is not None:
                _append_jsonl(train_log_path, epoch_avg)

            val_msg = ""
            if val_avg is not None:
                val_msg = (
                    f" | Val Loss: {epoch_avg['val_loss']:.4f}"
                    f" | Val BPP: {epoch_avg['val_bpp_loss']:.4f}"
                    f" | Val MSE: {epoch_avg['val_mse_loss']:.6f}"
                )

            print(
                f"[{variant_name}] Epoch {epoch + 1}/{epochs}"
                f" | Loss: {epoch_avg['loss']:.4f}"
                f" | BPP: {epoch_avg['bpp_loss']:.4f}"
                f" | MSE: {epoch_avg['mse_loss']:.6f}"
                f" | SSIM_L: {epoch_avg['ssim_loss']:.4f}"
                f" | LPIPS: {epoch_avg['lpips_loss']:.4f}"
                f" | Aux: {epoch_avg['aux_loss']:.4f}"
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    print(f"[{variant_name}] Saved checkpoint to {checkpoint_path}.")

    return {
        "history": history,
        "checkpoint_path": checkpoint_path,
    }