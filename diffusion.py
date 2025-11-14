import torch

def add_noise_batch(x0_batch, t, beta=0.01):
    """
    x0_batch: [B, C, H, W]
    t:        [B] timesteps
    """
    B = x0_batch.shape[0]

    alpha = 1 - beta
    alpha_bar = alpha ** t   # [B]

    epsilon = torch.randn_like(x0_batch)

    # reshape alpha_bar for broadcasting
    alpha_bar = alpha_bar.view(B, 1, 1, 1)

    xt = torch.sqrt(alpha_bar) * x0_batch + torch.sqrt(1 - alpha_bar) * epsilon

    return xt, epsilon
