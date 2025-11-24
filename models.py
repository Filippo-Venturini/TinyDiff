import torch
import torch.nn as nn
from diffusion import sinusoidal_embedding

class SmallUNet(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=32, max_timestep=1000):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), 
            nn.ReLU()
        )
        
        self.conv1 = nn.Conv2d(img_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv_out = nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        """
        x: [B, C, H, W] noisy image
        t: [B] timestep
        """
        t_emb = sinusoidal_embedding(t, self.hidden_dim)  # [B, hidden_dim]
        scale_shift = self.mlp(t_emb)  # [B, hidden_dim*2]
        scale, shift = scale_shift.chunk(2, dim=1)  # [B, hidden_dim]

        scale = scale[:,:,None,None]
        shift = shift[:,:,None,None]  # [B, hidden_dim, 1, 1]
        
         # --- first block ---
        h1 = self.conv1(x)
        h1 = h1 * scale + shift
        h1 = self.relu1(h1)

        # --- second block ---
        h2 = self.conv2(h1)
        h2 = h2 * scale + shift
        h2 = h2 = self.relu2(h2)

        # --- skip connection ---
        h = h1 + h2

        out = self.conv_out(h)
        return out