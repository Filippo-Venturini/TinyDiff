import torch
from ddpm_text.text_encoder import SimpleTextEncoder
from ddpm_text.tokenizer import SimpleTokenizer
from ddpm_text.models_text import SmallUNetText

device = "cpu"

tokenizer = SimpleTokenizer()
encoder = SimpleTextEncoder(vocab_size=len(tokenizer.word2idx), token_emb_dim=64, out_dim=64).to(device)

texts = ["sneaker", "coat", "t-shirt", "unknownword"]
ids = [tokenizer.encode(t) for t in texts]
ids = torch.tensor(ids, dtype=torch.long, device=device)   # [B, T]

with torch.no_grad():
    text_emb = encoder(ids)       # [B, 64]
    print("text_emb shape:", text_emb.shape)

model = SmallUNetText(img_ch=1, base_ch=32, t_dim=64, text_dim=64).to(device)
B = len(texts)
dummy_x = torch.randn(B, 1, 28, 28, device=device)
timesteps = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)

with torch.no_grad():
    out = model(dummy_x, timesteps, text_emb)
    print("model out shape:", out.shape)  
