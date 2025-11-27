import torch
import os
from ddpm_core.models import SmallUNet

def load_model(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")
    model = SmallUNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)