import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ATICLoss(nn.Module):
    """
    Total Loss = λ₁ * Rate + λ₂ * MSE + λ₃ * SSIM + λ₄ * LPIPS
    """
    def __init__(self, lambda_rate=0.01, lambda_mse=1.0, lambda_ssim=0.5, lambda_lpips=0.1, device="cuda"):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.device = device
        
        # Load differentiable LPIPS
        if self.lambda_lpips > 0:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
            except ImportError:
                print("Warning: lpips library missing. LPIPS loss will be 0.")
                self.lpips_fn = None
                
        # Load differentiable SSIM
        if self.lambda_ssim > 0:
            try:
                from piq import ssim
                self.ssim_fn = ssim
            except ImportError:
                print("Warning: piq library missing. SSIM loss will be 0.")
                self.ssim_fn = None

    def forward(self, output_dict, target):
        x_hat = output_dict['x_hat']
        likelihoods = output_dict['likelihoods']
        
        # 1. MSE Loss (Pixel Error)
        mse_loss = F.mse_loss(x_hat, target)
        
        # 2. Rate Loss (BPP)
        bpp_loss = torch.tensor(0.0, device=self.device)
        if likelihoods is not None:
            N, _, H, W = target.size()
            num_pixels = N * H * W
            if isinstance(likelihoods, dict):
                for likelihood in likelihoods.values():
                    bpp_loss += torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
            elif isinstance(likelihoods, torch.Tensor):
                bpp_loss = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                
        # 3. SSIM Loss (Structural)
        ssim_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_ssim > 0 and self.ssim_fn:
            # SSIM is higher=better (max 1.0), so loss is (1 - SSIM)
            ssim_val = self.ssim_fn(torch.clamp(x_hat, 0, 1), target, data_range=1.0)
            ssim_loss = 1.0 - ssim_val
            
        # 4. LPIPS Loss (Perceptual)
        lpips_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_lpips > 0 and self.lpips_fn:
            # LPIPS expects [-1, 1]
            x_hat_norm = torch.clamp(x_hat, 0, 1) * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            lpips_loss = self.lpips_fn(x_hat_norm, target_norm).mean()
            
        # Total Loss Calculation
        total_loss = (self.lambda_rate * bpp_loss) + \
                     (self.lambda_mse * mse_loss) + \
                     (self.lambda_ssim * ssim_loss) + \
                     (self.lambda_lpips * lpips_loss)
                     
        return {
            "loss": total_loss,
            "bpp_loss": bpp_loss,
            "mse_loss": mse_loss,
            "ssim_loss": ssim_loss,
            "lpips_loss": lpips_loss
        }
