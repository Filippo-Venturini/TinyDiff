import torch
from models import SmallUNet

def load_model(path, device="cpu"):
    model = SmallUNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model