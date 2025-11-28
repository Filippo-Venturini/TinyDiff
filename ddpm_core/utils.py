import torch
import os
from ddpm_core.models import SmallUNet
from ddpm_text.models_text import SmallUNetText
from ddpm_text.text_encoder import SimpleTextEncoder

def load_model(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")
    model = SmallUNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_model_text(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")
    model = SmallUNetText()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_text_encoder(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at {path}")
    text_encoder = SimpleTextEncoder(vocab_size=12)
    text_encoder.load_state_dict(torch.load(path, map_location=device))
    text_encoder.to(device)
    text_encoder.eval()
    return text_encoder

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)