import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def add_noise_batch(x0_batch, t, beta=0.01):
    """
    x0_batch: images batch [B, C, H, W]
    t: single timestep
    """
    alpha = 1 - beta
    alpha_bar = alpha ** t
    epsilon = torch.randn_like(x0_batch)
    print(epsilon[0])
    xt = torch.sqrt(torch.tensor(alpha_bar)) * x0_batch + torch.sqrt(torch.tensor(1 - alpha_bar)) * epsilon
    return xt

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

batch = next(iter(train_loader))
images, labels = batch

timesteps = [0, 200, 400, 600, 800]
fig, axs = plt.subplots(len(timesteps), 8, figsize=(12, 8))

for i, t in enumerate(timesteps):
    xt_batch = add_noise_batch(images[0:8], t)
    for j in range(8):
        axs[i,j].imshow(xt_batch[j,0].detach().numpy(), cmap='gray')
        axs[i,j].axis('off')
    axs[i,0].set_ylabel(f"t={t}")
plt.show()

