import os
import torch
import torch.optim as optim
from atic.losses import ATICLoss

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
        for batch_idx, batch in enumerate(dataloader):
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
            
            if batch_idx % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Total Loss: {loss.item():.4f} | BPP: {loss_dict['bpp_loss'].item():.4f} | MSE: {loss_dict['mse_loss'].item():.4f}")
            
    # Save checkpoint specifically for this lambda rate target
    torch.save(model.state_dict(), f"ablation_results/{variant_name}_rate_{lambda_rate}.pth")
    print(f"[{variant_name}] Saved checkpoint.")
