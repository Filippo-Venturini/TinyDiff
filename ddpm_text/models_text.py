import torch
import torch.nn as nn
from ddpm_core.diffusion import sinusoidal_embedding
from ddpm_core.models import DownBlock, UpBlock

class SmallUNetText(nn.Module):
    def __init__(self, img_ch=1, base_ch=32, t_dim=64, text_dim=64):
        super().__init__()
        self.t_dim = t_dim
        self.text_dim = text_dim
        assert t_dim == text_dim

        self.down1 = DownBlock(img_ch, base_ch, t_dim)
        self.down2 = DownBlock(base_ch, base_ch*2, t_dim)

        self.mid = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = UpBlock(base_ch*2, base_ch*2, base_ch, t_dim)
        self.up1 = UpBlock(base_ch, base_ch, base_ch, t_dim)

        self.final = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    def forward(self, x, t, text_emb):
        """
        x: [B,C,H,W]
        t: [B] long tensor
        text_emb: [B, text_dim]
        """
        t_emb = sinusoidal_embedding(t, dim=self.t_dim)   # [B, t_dim]
        combined = t_emb + text_emb                       # [B, t_dim]

        d1, h1 = self.down1(x, combined)
        d2, h2 = self.down2(d1, combined)

        h = self.mid(d2)

        h = self.up2(h, h2, combined)
        h = self.up1(h, h1, combined)

        out = self.final(h)
        return out
