import torch
import torch.nn as nn

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size=5000, token_emb_dim=64, out_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(token_emb_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, token_ids):
        """
        token_ids: LongTensor [B, T]
        return: tensor [B, out_dim]
        """
        emb = self.embedding(token_ids)         # [B, T, token_emb_dim]
        emb = emb.mean(dim=1)                   # [B, token_emb_dim]  (avg pooling)
        out = self.fc(emb)                      # [B, out_dim]
        return out
