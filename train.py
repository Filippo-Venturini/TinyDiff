import torch
import torch.nn as nn
import torch.optim as optim

from diffusion import add_noise_batch
from data_loader import get_mnist_dataloader
from models import SmallUNet


def train(
    epochs=3,
    batch_size=16,
    lr=1e-3,
    max_timestep=1000,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model = SmallUNet(max_timestep=max_timestep).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataloader = get_mnist_dataloader(batch_size=batch_size)

    model.train()

    for epoch in range(epochs):
        for images, _ in dataloader:

            images = images.to(device)

            # Sample different t for each image in the batch
            B = images.shape[0]
            timesteps = torch.randint(0, max_timestep, (B,), device=device)

            # Compute noisy image x_t and the true noise
            x_t, epsilon = add_noise_batch(images, timesteps)

            x_t = x_t.to(device)
            epsilon = epsilon.to(device)

            # Predict noise with the model
            epsilon_pred = model(x_t, timesteps)

            # MSE between predicted and real noise
            loss = criterion(epsilon_pred, epsilon)

            # Update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    return model
