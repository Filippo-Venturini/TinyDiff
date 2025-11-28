import torch
import torch.nn as nn
import torch.optim as optim
from ddpm_core.data_loader import get_fashion_mnist_dataloader
from ddpm_text.tokenizer import SimpleTokenizer
from ddpm_text.text_encoder import SimpleTextEncoder
from ddpm_text.models_text import SmallUNetText
from ddpm_core.diffusion import make_beta_schedule, add_noise_batch
from ddpm_core.utils import save_model

device = "cuda" if torch.cuda.is_available() else "cpu"

label_to_text = {
    0: "t-shirt",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boot"
}

def train_text(
    epochs=10,
    batch_size=16,
    lr=1e-4,
    max_timestep=1000,
    t_dim=64,
    text_dim=64,
    model_name="text_ddpm.pt"
):
    dataloader = get_fashion_mnist_dataloader(batch_size=batch_size, train=True)
    tokenizer = SimpleTokenizer(max_len=1)
    text_encoder = SimpleTextEncoder(vocab_size=len(label_to_text)+2, token_emb_dim=text_dim)
    model = SmallUNetText(img_ch=1, base_ch=32, t_dim=t_dim, text_dim=text_dim).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(text_encoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    
    betas, alphas, alpha_cumprod = make_beta_schedule(max_timestep, device=device)
    
    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            B = images.shape[0]
            timesteps = torch.randint(0, max_timestep, (B,), device=device)
            
            texts = [label_to_text[int(l)] for l in labels]
            token_ids = [tokenizer.encode(t) for t in texts]
            token_ids = torch.tensor(token_ids, device=device)
            text_emb = text_encoder(token_ids)
            
            x_t, epsilon = add_noise_batch(images, timesteps, alpha_cumprod)
            
            epsilon_pred = model(x_t, timesteps, text_emb)
            
            loss = criterion(epsilon_pred, epsilon)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    save_model(model, model_name)
    print("Training complete and model saved:", model_name)
    return model, text_encoder
