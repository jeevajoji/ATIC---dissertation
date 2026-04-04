import os
import json
import torch
import torch.optim as optim
from atic.losses import ATICLoss
from tqdm import tqdm


def _append_jsonl(file_path, payload):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def train_loop(
    model,
    variant_name,
    dataloader,
    epochs=5,
    device="cuda",
    lambda_rate=0.01,
    checkpoint_path=None,
    train_log_path=None,
):
    if dataloader is None:
        print(f"[{variant_name}] Dataloader not found. Skipping training.")
        return {"history": [], "checkpoint_path": None}

    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Matching custom loss constraints
    criterion = ATICLoss(
        lambda_rate=lambda_rate, 
        lambda_mse=1.0, 
        lambda_ssim=0.5, 
        lambda_lpips=0.1, 
        device=device
    )
    
    print(f"[{variant_name}] Training started on {device} (Lambda Rate: {lambda_rate})...")

    history = []
    
    for epoch in range(epochs):
        epoch_totals = {
            "loss": 0.0,
            "bpp_loss": 0.0,
            "mse_loss": 0.0,
            "ssim_loss": 0.0,
            "lpips_loss": 0.0,
        }
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(batch)
            
            # Loss
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "BPP": f"{loss_dict['bpp_loss'].item():.4f}", 
                "MSE": f"{loss_dict['mse_loss'].item():.4f}"
            })

            epoch_totals["loss"] += loss.item()
            epoch_totals["bpp_loss"] += loss_dict["bpp_loss"].item()
            epoch_totals["mse_loss"] += loss_dict["mse_loss"].item()
            epoch_totals["ssim_loss"] += loss_dict["ssim_loss"].item()
            epoch_totals["lpips_loss"] += loss_dict["lpips_loss"].item()
            epoch_steps += 1

        if epoch_steps > 0:
            epoch_avg = {
                "epoch": epoch + 1,
                "steps": epoch_steps,
                "loss": epoch_totals["loss"] / epoch_steps,
                "bpp_loss": epoch_totals["bpp_loss"] / epoch_steps,
                "mse_loss": epoch_totals["mse_loss"] / epoch_steps,
                "ssim_loss": epoch_totals["ssim_loss"] / epoch_steps,
                "lpips_loss": epoch_totals["lpips_loss"] / epoch_steps,
                "lambda_rate": lambda_rate,
                "variant": variant_name,
            }
            history.append(epoch_avg)
            if train_log_path is not None:
                _append_jsonl(train_log_path, epoch_avg)
            
        # Free up GPU memory at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Save checkpoint after all epochs for this specific variant and lambda
    if checkpoint_path is None:
        checkpoint_path = f"ablation_results/{variant_name}.pth"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[{variant_name}] Saved checkpoint to {checkpoint_path}.")

    return {
        "history": history,
        "checkpoint_path": checkpoint_path,
    }
