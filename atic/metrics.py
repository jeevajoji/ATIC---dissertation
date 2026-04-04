import torch
import torch.nn.functional as F
import math

class ATICMetrics:
    """
    Helper class to compute standard and perceptual image quality metrics.
    Note: Requires external libraries for some metrics:
    pip install piq lpips pytorch-msssim
    """
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load LPIPS model if available
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
            self.lpips_fn.eval()
        except ImportError:
            self.lpips_fn = None
            print("Warning: lpips not installed. Run pip install lpips.")

        try:
            from piq import DISTS, multi_scale_ssim, ssim
            self.dists_fn = DISTS().to(device)
            self.ms_ssim_fn = multi_scale_ssim
            self.ssim_fn = ssim
        except ImportError:
            self.dists_fn = None
            self.ms_ssim_fn = None
            self.ssim_fn = None
            print("Warning: piq not installed. Run pip install piq for MS-SSIM, SSIM, and DISTS.")

    def psnr(self, img1, img2, data_range=1.0):
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 10 * math.log10((data_range ** 2) / mse.item())

    def compute_all(self, x_hat, x, bpp=0.0):
        """
        x, x_hat: Tensors of shape (B, 3, H, W) in range [0, 1]
        """
        x_hat = torch.clamp(x_hat, 0.0, 1.0)
        
        metrics = {
            "BPP": bpp,
            "PSNR": self.psnr(x_hat, x),
            "MSE": F.mse_loss(x_hat, x).item()
        }

        # Compute SSIM / MS-SSIM
        if self.ssim_fn is not None:
            metrics["SSIM"] = self.ssim_fn(x_hat, x, data_range=1.0).item()
            metrics["MS-SSIM"] = self.ms_ssim_fn(x_hat, x, data_range=1.0).item()

        # Compute LPIPS
        if self.lpips_fn is not None:
            # LPIPS expects [-1, 1] image range
            x_hat_lpips = x_hat * 2.0 - 1.0
            x_lpips = x * 2.0 - 1.0
            metrics["LPIPS"] = self.lpips_fn(x_hat_lpips, x_lpips).mean().item()

        # Compute DISTS
        if self.dists_fn is not None:
            metrics["DISTS"] = self.dists_fn(x_hat, x).item()

        # Note: VMAF and FID are dataset/video level metrics. 
        # Usually evaluated externally using ffmpeg/libvmaf and torchmetrics.FID.
        
        return metrics

import os
import matplotlib.pyplot as plt

def plot_rate_distortion_curves(results_dict, metrics_to_plot=['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS', 'DISTS'], save_dir="ablation_results/plots"):
    """
    Plots Rate-Distortion curves charting BPP (x-axis) vs PSNR/SSIM/LPIPS (y-axis).
    Allows comparing ATIC variants directly with JPEG, WebP, or CompressAI baselines.
    
    Expected results_dict structure:
    {
        "A1_Baseline": {
            0.05: {"PSNR": 28.1, "SSIM": 0.82, "LPIPS": 0.15, ...},
            0.10: {"PSNR": 30.5, "SSIM": 0.88, "LPIPS": 0.10, ...},
            ...
        },
        "JPEG (Baseline)": {
            ...
        }
    }
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        plotted_any = False
        for model_name, bpp_data in results_dict.items():
            if not bpp_data:
                continue
                
            sorted_bpps = sorted(list(bpp_data.keys()))
            valid_bpps = []
            metric_vals = []
            
            for b in sorted_bpps:
                if metric in bpp_data[b]:
                    valid_bpps.append(b)
                    metric_vals.append(bpp_data[b][metric])
                    
            if valid_bpps and metric_vals:
                plotted_any = True
                # Add markers and line matching the requested plotting style
                plt.plot(valid_bpps, metric_vals, marker='o', linewidth=2, markersize=8, label=model_name)
        
        if not plotted_any:
            plt.close()
            continue
            
        plt.xlabel('Bits Per Pixel (BPP)', fontsize=12, fontweight='bold')
        
        # Display note for lower-is-better vs higher-is-better metrics
        if metric in ['LPIPS', 'DISTS', 'MSE']:
            plt.ylabel(f'{metric} (Lower is Better)', fontsize=12, fontweight='bold')
        else:
            plt.ylabel(f'{metric} (Higher is Better)', fontsize=12, fontweight='bold')
            
        plt.title(f'Rate-Distortion Evaluation: {metric} vs BPP', fontsize=14, fontweight='bold')
        plt.grid(True, which="both", linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        
        # Save output png
        save_path = os.path.join(save_dir, f'RD_Curve_{metric}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        print(f"Generated RD Graph: {save_path}")
        plt.show() # Display inline in Kaggle notebook
        plt.close()
