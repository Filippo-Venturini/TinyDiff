import torch
import torch.nn as nn
from ddpm_core.diffusion import sinusoidal_embedding

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.mlp = nn.Linear(t_dim, out_ch*2)

    def forward(self, x, t_emb):
        scale, shift = self.mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.act(self.conv1(x))
        h = h * scale + shift
        h_down = self.act(self.conv2(h))
        h_down = h_down * scale + shift
        return h_down, h

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU()
        self.mlp = nn.Linear(t_dim, out_ch*2)

    def forward(self, x, skip, t_emb):
        x = self.up(x)

        # ensure sizes match
        min_h = min(x.size(-2), skip.size(-2))
        min_w = min(x.size(-1), skip.size(-1))
        x = x[:, :, :min_h, :min_w]
        skip = skip[:, :, :min_h, :min_w]

        x = torch.cat([x, skip], dim=1)

        scale, shift = self.mlp(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        x = self.act(self.conv(x))
        x = x * scale + shift
        return x

class SmallUNet(nn.Module):
    def __init__(self, img_ch=1, base_ch=32, t_dim=64):
        super().__init__()

        # Downsampling
        self.down1 = DownBlock(img_ch, base_ch, t_dim)
        self.down2 = DownBlock(base_ch, base_ch*2, t_dim)

        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1),
            nn.ReLU()
        )

        # Upsampling
        self.up2 = UpBlock(base_ch*2, base_ch*2, base_ch, t_dim)
        self.up1 = UpBlock(base_ch, base_ch, base_ch, t_dim)

        # Output
        self.final = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, dim=64)

        d1, h1 = self.down1(x, t_emb)
        d2, h2 = self.down2(d1, t_emb)

        h = self.mid(d2)

        h = self.up2(h, h2, t_emb)
        h = self.up1(h, h1, t_emb)

        return self.final(h)
