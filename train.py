import torch
import torch.nn as nn
import torch.optim as optim

from diffusion import add_noise_batch
from data_loader import get_mnist_dataloader
from models import SmallUNet

def train(
    epochs=20,               
    batch_size=64,           
    lr=1e-4,                
    max_timestep=1000,
    beta=0.02,               
    device="cuda" if torch.cuda.is_available() else "cpu",
    print_interval=100       
):
    model = SmallUNet(max_timestep=max_timestep).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataloader = get_mnist_dataloader(batch_size=batch_size)

    model.train()

    for epoch in range(epochs):
        for batch_idx, (images, _) in enumerate(dataloader):

            images = images.to(device)

            # Sample different t for each image in the batch
            B = images.shape[0]
            timesteps = torch.randint(0, max_timestep, (B,), device=device)

            # Compute noisy image x_t and the true noise
            x_t, epsilon = add_noise_batch(images, timesteps, beta=beta)

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

            # Print diagnostics every print_interval batches
            if batch_idx % print_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                print(f"x_t min/max: {x_t.min():.4f}/{x_t.max():.4f}")
                print(f"epsilon min/max: {epsilon.min():.4f}/{epsilon.max():.4f}")
                print(f"epsilon_pred min/max: {epsilon_pred.min():.4f}/{epsilon_pred.max():.4f}")

        print(f"Epoch {epoch+1}/{epochs} completed. Last batch loss: {loss.item():.4f}")

    print("Training complete, saving model...")
    torch.save(model.state_dict(), "model_v2.pt")

    return model
