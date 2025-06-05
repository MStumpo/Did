import torch.nn as nn
import torch.nn.functional as F


class LatentEmbedder(nn.Module):
    def __init__(self, token_dim, emb_dim, kernel_size=3, stride=2, padding=1, hidden_dim=128, n_convs=1):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(token_dim if i == 0 else emb_dim, emb_dim, kernel_size=kernel_size, stride=stride, padding=padding) for i in range(n_convs)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(n_convs)])
        self.Linear1 = nn.Linear(emb_dim, hidden_dim)
        self.Linear2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
        x = x.transpose(1, 2)
        x = F.relu(self.Linear1(x))
        x = self.Linear2(x)
        return x