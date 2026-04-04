import os
import torch
import torch.optim as optim
from atic.losses import ATICLoss
from tqdm import tqdm

def train_loop(model, variant_name, dataloader, epochs=5, device="cuda", lambda_rate=0.01):
    if dataloader is None:
        print(f"[{variant_name}] Dataloader not found. Skipping training.")
        return

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
    
    for epoch in range(epochs):
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
            
        # Free up GPU memory at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Save checkpoint specifically for this lambda rate target
    torch.save(model.state_dict(), f"ablation_results/{variant_name}_rate_{lambda_rate}.pth")
    print(f"[{variant_name}] Saved checkpoint.")
