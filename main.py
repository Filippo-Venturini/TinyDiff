import torch
import matplotlib.pyplot as plt

from data_loader import get_mnist_dataloader
from diffusion import add_noise_batch
from models import SmallUNet
from train import train

def visualize_noisy_images():
    loader = get_mnist_dataloader(batch_size=16)
    batch = next(iter(loader))
    images, _ = batch

    timesteps = [0, 200, 400, 600, 800]
    fig, axs = plt.subplots(len(timesteps), 8, figsize=(12, 8))

    for i, t in enumerate(timesteps):
        xt_batch, _ = add_noise_batch(images[0:8], t)
        for j in range(8):
            axs[i,j].imshow(xt_batch[j,0].detach().numpy(), cmap='gray')
            axs[i,j].axis('off')
        axs[i,0].set_ylabel(f"t={t}")
    plt.show()

def test_model_forward():
    model = SmallUNet()
    loader = get_mnist_dataloader(batch_size=8)
    batch = next(iter(loader))
    images, _ = batch
    timesteps = torch.randint(0, 1000, (images.shape[0],))
    x_t, _ = add_noise_batch(images, timesteps[0].item())  # using same t for simplicity
    epsilon_pred = model(x_t, timesteps)
    print(epsilon_pred.shape)

if __name__ == "__main__":
    model = train(epochs=3, batch_size=16)