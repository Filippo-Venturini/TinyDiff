import torch
import torch.nn as nn

class SmallUNet(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=32, max_timestep=1000):
        super().__init__()
        
        self.t_embed = nn.Embedding(max_timestep, hidden_dim)
        
        self.conv1 = nn.Conv2d(img_channels + 1, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv_out = nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        """
        x: [B, C, H, W] noisy image
        t: [B] timestep
        """
        t_emb = self.t_embed(t)  # [B, hidden_dim]
        t_emb = t_emb[:,:,None,None]  # [B, hidden_dim, 1, 1]
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])  # [B, hidden_dim, H, W]
        
        x_in = torch.cat([x, t_emb[:,0:1]], dim=1)  # use only one channel of embedding
        
        h = self.relu1(self.conv1(x_in))
        h = self.relu2(self.conv2(h))
        out = self.conv_out(h)
        return out