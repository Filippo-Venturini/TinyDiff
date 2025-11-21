import torch
import matplotlib.pyplot as plt

from data_loader import get_mnist_dataloader
from diffusion import add_noise_batch, make_beta_schedule
from models import SmallUNet
from train import train
from sampler import sample
from utils import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_noisy_images():
    loader = get_mnist_dataloader(batch_size=16)
    batch = next(iter(loader))
    images, _ = batch

    timesteps = [0, 200, 400, 600, 800]
    fig, axs = plt.subplots(len(timesteps), 8, figsize=(12, 8))

    _, _, alpha_cumprod = make_beta_schedule(T=800, beta_start=1e-4, beta_end=0.02, device=device)

    for i, t in enumerate(timesteps):
        xt_batch, _ = add_noise_batch(images[0:8], torch.tensor([t]*8), alpha_cumprod=alpha_cumprod)
        for j in range(8):
            axs[i, j].imshow(xt_batch[j, 0].detach().numpy(), cmap='gray')
            axs[i, j].axis('off')
        axs[i, 0].set_ylabel(f"t={t}")
    plt.show()

def test_model_forward():
    model = SmallUNet().to(device)
    loader = get_mnist_dataloader(batch_size=8)
    batch = next(iter(loader))
    images, _ = batch
    images = images.to(device)
    timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)
    _, _, alpha_cumprod = make_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device=device)

    x_t, _ = add_noise_batch(images, timesteps, alpha_cumprod=alpha_cumprod)
    epsilon_pred = model(x_t, timesteps)
    print("Forward pass output shape:", epsilon_pred.shape)

def generate_samples(model_path="model.pt", num_samples=16, img_size=28):
    model = load_model(model_path, device=device)
    samples = sample(model, num_samples=num_samples, img_size=img_size, device=device)

    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(num_samples):
        axs[i // 4, i % 4].imshow(samples[i, 0].cpu().numpy(), cmap="gray")
        axs[i // 4, i % 4].axis("off")
    plt.show()

if __name__ == "__main__":
    # Train the model (weights will be saved automatically)
    # model = train(model_name="model_v3.pt")

    # Optional: visualize noisy images at different timesteps
    #visualize_noisy_images()

    # Optional: test a forward pass
    #test_model_forward()

    # Generate new MNIST samples using the trained model
    generate_samples("models/model_v3.pt")
