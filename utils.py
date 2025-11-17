import torch
from models import SmallUNet

def load_model(path, max_timestep=1000, device="cpu"):
    model = SmallUNet(max_timestep=max_timestep)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model