import torch
import torch.nn as nn
from diffusion import sinusoidal_embedding

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)  # downsample
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(hidden_dim, out_ch*2)  # timestep embedding

    def forward(self, x, t_emb):
        scale_shift = self.mlp(t_emb).chunk(2, dim=1)
        scale, shift = [s[:,:,None,None] for s in scale_shift]

        h = self.conv1(x)
        h = h * scale + shift
        h = self.relu(h)

        h_down = self.conv2(h)
        h_down = h_down * scale + shift
        h_down = self.relu(h_down)

        return h_down, h  # downsampled + skip connection

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, hidden_dim):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv_fuse = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)  # fuse skip
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(hidden_dim, out_ch*2)

    def forward(self, x, skip, t_emb):
        x = self.conv_trans(x)
        x = torch.cat([x, skip], dim=1)  # concat skip connection

        scale_shift = self.mlp(t_emb).chunk(2, dim=1)
        scale, shift = [s[:,:,None,None] for s in scale_shift]

        x = self.conv_fuse(x)
        x = x * scale + shift
        x = self.relu(x)
        return x

class SmallUNet(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=32, max_timestep=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_timestep = max_timestep

        # --- Down ---
        self.down1 = DownBlock(img_channels, hidden_dim, hidden_dim)
        self.down2 = DownBlock(hidden_dim, hidden_dim*2, hidden_dim)

        # --- Bottleneck ---
        self.mid_conv = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1)
        self.relu_mid = nn.ReLU()

        # --- Up ---
        self.up2 = UpBlock(in_ch=hidden_dim*2, skip_ch=hidden_dim*2, out_ch=hidden_dim, hidden_dim=hidden_dim)
        self.up1 = UpBlock(in_ch=hidden_dim, skip_ch=hidden_dim, out_ch=hidden_dim, hidden_dim=hidden_dim)

        # --- Output ---
        self.conv_out = nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.hidden_dim)

        # --- Down ---
        d1, skip1 = self.down1(x, t_emb)
        d2, skip2 = self.down2(d1, t_emb)

        # --- Bottleneck ---
        h = self.mid_conv(d2)
        h = self.relu_mid(h)

        # --- Up ---
        h = self.up2(h, skip2, t_emb)
        h = self.up1(h, skip1, t_emb)

        # --- Output ---
        out = self.conv_out(h)
        return out
