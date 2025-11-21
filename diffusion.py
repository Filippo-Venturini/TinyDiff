import torch
import math

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    """
    Linear beta schedule from beta_start to beta_end with T timesteps.
    Returns beta (T,), alpha (T,), alpha_cumprod (T,)
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)  # shape [T]
    alphas = 1.0 - betas  # [T]
    alpha_cumprod = torch.cumprod(alphas, dim=0)  # [T]
    return betas, alphas, alpha_cumprod

def sinusoidal_embedding(timesteps, dim):
    """
    timesteps: [B] int tensor
    dim: dimension of the embedding
    returns: [B, dim] float tensor
    """
    B = timesteps.shape[0]
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # [half_dim]
    emb = timesteps[:, None].float() * emb[None, :]               # [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)      # [B, dim]
    return emb

def add_noise_batch(x0_batch, t, alpha_cumprod):
    """
    x0_batch: [B, C, H, W], values in [0,1] (or normalized)
    t:        [B] timesteps in [0, T-1] (long tensor)
    alpha_cumprod: [T] tensor precomputed (alpha_1 * ... * alpha_t)
    returns: xt, epsilon (both same shape as x0_batch)
    """
    B = x0_batch.shape[0]
    device = x0_batch.device

    # gather alpha_bar for each example in batch
    # alpha_cumprod[t] -> [B], then reshape for broadcasting
    a_bar = alpha_cumprod[t].view(B, 1, 1, 1)  # [B,1,1,1]

    epsilon = torch.randn_like(x0_batch)

    xt = torch.sqrt(a_bar) * x0_batch + torch.sqrt(1.0 - a_bar) * epsilon

    return xt, epsilon