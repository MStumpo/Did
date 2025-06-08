import torch
from torch import nn

from latent_embedder import LatentEmbedder
from crossable_transformer import CrossableTransformer
from memory_gater import MemoryGater
from simple_decoder import SimpleDecoder

class ExampleModel(nn.Module):
    def __init__(self, encoder=None, m_dim=128, token_dim=32, emb_dim=16, dict_size=1024, hidden_dim=64, n_stacks=3,
                 dropout=0.1, n_heads=8, n_convs=1):
        super().__init__()
        self.register_buffer('memory', torch.zeros(m_dim, emb_dim), persistent=True)

        if encoder is None:
            self.encoder = LatentEmbedder(token_dim, emb_dim, hidden_dim=hidden_dim, n_convs=n_convs)
        else:
            self.encoder = encoder

        self.self_stacks = nn.ModuleList([CrossableTransformer(emb_dim, n_heads, hidden_dim, dropout) for _ in range(n_stacks)])
        self.memory_stacks = nn.ModuleList([MemoryGater(emb_dim, m_dim, dropout) for _ in range(n_stacks)])
        self.cross_stacks = nn.ModuleList([CrossableTransformer(emb_dim, n_heads, hidden_dim, dropout) for _ in range(n_stacks)])
        self.logits = nn.Sequential(SimpleDecoder(emb_dim, token_dim, hidden_dim), nn.Linear(token_dim, dict_size))

    def forward(self, x):
        x = self.encoder(x)
        for i in range(len(self.self_stacks)):
            x = self.self_stacks[i](x)
            self.memory, x_fetch = self.memory_stacks[i](x, self.memory)
            x = self.cross_stacks[i](x, x_fetch)
        latent = x
        logits = self.logits(x)

        return latent, logits

    def reset_memory(self):
        self.memory.zero_()