import torch.nn as nn

class CrossableTransformer(nn.Module):
    def __init__(self, emb_dim, n_heads=4, h_dim=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_fetch=None, mask=None):
        if x_fetch is not None:
            att, _ = self.attn(x, x_fetch, x_fetch, attn_mask=mask)
        else:
            att, _ = self.attn(x, x, x, attn_mask=mask)

        x = self.norm1(x + self.dropout(att))
        x_ff = self.ff(x)
        x = self.norm2(x + self.dropout(x_ff))

        return x
