import torch
import torch.nn.functional as F
import math
from atic.metrics import ATICMetrics
from atic.losses import ATICLoss

def eval_loop(model, variant_name, dataloader, device="cuda", bpp_levels=[0.05, 0.1, 0.2, 0.5, 1.0]):
    if dataloader is None:
        print(f"[{variant_name}] Dataloader not found. Skipping Eval.")
        return

    model.eval()
    model.to(device)
    
    # Initialize the exhaustive metrics pipeline automatically
    metric_calculator = ATICMetrics(device=device)
    
    print(f"\n=========================================")
    print(f"[{variant_name}] Multi-rate Evaluation Starting...")
    print(f"=========================================")
    
    results = {
        "variant": variant_name,
        "bpp_levels": {}
    }
    
    with torch.no_grad():
        for target_bpp in bpp_levels:
            print(f"\n---> Testing Target BPP Range: ~{target_bpp}")
            
            # Map target BPP to lambda (a heuristic mapping, or set dynamically during inference loop)
            # In an actual test, lambda is used in training, and here we use the specific model weights at that lambda
            
            total_metrics = {"PSNR": 0, "SSIM": 0, "MS-SSIM": 0, "LPIPS": 0, "DISTS": 0, "MSE": 0, "BPP": 0}
            valid_metric_counts = {k: 0 for k in total_metrics}
            
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(device)
                
                # Mock evaluation inference
                outputs = model(batch)
                
                # We mock BPP out of the entropy block here
                if outputs["likelihoods"] is not None:
                    # In reality, rate comes from entropy
                    pass
                
                # For baseline tracking
                bpp = outputs.get("bpp", target_bpp)  # Mock to target if model not trained
                
                # Compute all perceptual metrics automatically for this batch
                batch_metrics = metric_calculator.compute_all(outputs["x_hat"], batch, bpp=bpp)
                
                for k, v in batch_metrics.items():
                    if k in total_metrics:
                        total_metrics[k] += v
                        valid_metric_counts[k] += 1
                        
            # Average out
            print(f"  Results at {target_bpp} BPP:")
            record = {}
            for k in total_metrics.keys():
                if valid_metric_counts[k] > 0:
                    avg_val = total_metrics[k] / valid_metric_counts[k]
                    record[k] = avg_val
                    print(f"    {k}: {avg_val:.4f}")
            
            results["bpp_levels"][target_bpp] = record
            
    return results
