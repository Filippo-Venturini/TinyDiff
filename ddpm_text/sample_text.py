import torch
from ddpm_core.diffusion import make_beta_schedule

@torch.no_grad()
def sample_text(model, text_emb, num_samples, img_size, max_timestep=1000,
                beta_start=1e-4, beta_end=0.02, device="cpu"):
    model.eval()
    betas, alphas, alpha_cumprod = make_beta_schedule(max_timestep, beta_start=beta_start, beta_end=beta_end, device=device)

    x_t = torch.randn(num_samples, 1, img_size, img_size, device=device)

    for t in reversed(range(0, max_timestep)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps_pred = model(x_t, t_tensor, text_emb)

        a_bar_t = alpha_cumprod[t].view(1,1,1,1)
        x_0_pred = (x_t - torch.sqrt(1 - a_bar_t) * eps_pred) / torch.sqrt(a_bar_t)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_cumprod[t]

        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)

        if t > 1:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(betas[t]) * noise
        else:
            x_t = x_0_pred

    x_t = (x_t + 1) / 2
    x_t = x_t.clamp(0, 1)
    return x_t
