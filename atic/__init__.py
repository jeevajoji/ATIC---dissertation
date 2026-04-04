from atic.config import ArchitectureConfig
from atic.model import ATICModel
import torch

cfg = ArchitectureConfig()
model = ATICModel(cfg, H=1088, W=1920)
x = torch.randn(1, 3, 1088, 1920)
out = model(x)
print(out["x_hat"].shape)   # should be [1, 3, 1088, 1920]