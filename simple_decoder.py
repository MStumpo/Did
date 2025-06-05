import torch
from torch import nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    def __init__(self, emb_dim, token_dim, hidden_dim=128):
        super().__init__()

        self.linear1 = nn.Linear(emb_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, token_dim)

    def forward(self, x):
        x = self.linear1(x)
        #x = torch.mean(x, axis=1)
        x = F.relu(x)
        x = self.linear2(x)

        return x

