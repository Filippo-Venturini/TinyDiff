import torch
import torch.nn.functional as F

@torch.no_grad()
def sample(model, num_samples, img_size, max_timestep=1000, beta=0.01, device="cpu"):
    """
    Reverse diffusion sampling loop.
    Starts from x_T = pure noise and goes to x_0.
    """
    model.eval()

    beta = torch.tensor(beta, device=device)
    alpha = 1.0 - beta

    # Precompute alpha_bar[t] = alpha^t for all timesteps
    timesteps = torch.arange(0, max_timestep + 1, device=device).float()
    alpha_bar = alpha ** timesteps  # [T+1]

    # Start from pure Gaussian noise
    x_t = torch.randn(num_samples, 1, img_size, img_size, device=device)

    for t in reversed(range(1, max_timestep + 1)):
        # Build batch of timesteps
        t_tensor = torch.full((num_samples,), t-1, device=device, dtype=torch.long)

        # Predict noise ε
        eps_pred = model(x_t, t_tensor)

        # Get alpha_bar[t] and alpha_bar[t-1] for broadcasting
        a_bar_t = alpha_bar[t].view(1, 1, 1, 1)
        a_bar_prev = alpha_bar[t - 1].view(1, 1, 1, 1)

        # Predict x_0 using the DDPM formula
        x_0_pred = (x_t - torch.sqrt(1 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t)

        # Compute mean of q(x_{t-1} | x_t, x_0)
        mean = torch.sqrt(a_bar_prev) * x_0_pred + torch.sqrt(1 - a_bar_prev) * eps_pred

        # Sample noise if not final step
        if t > 1:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(beta) * noise
        else:
            x_t = x_0_pred  # last step, no noise

    # Clamp output to [0,1] for greyscale images
    x_t = x_t.clamp(0.0, 1.0)

    return x_t