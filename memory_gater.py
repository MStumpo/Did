import torch
import torch.nn as nn

class MemoryGater(nn.Module):
    def __init__(self, emb_dim, m_dim, dropout=0.1):
        super().__init__()
        self.write_gate = nn.Linear(emb_dim, m_dim)
        self.erase_gate = nn.Linear(emb_dim, m_dim)
        self.read_gate = nn.Linear(emb_dim, m_dim)
        self.alloc = nn.Linear(emb_dim, m_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, m):
        w = torch.sigmoid(torch.mean(self.write_gate(x), dim=1))
        e = torch.sigmoid(torch.mean(self.erase_gate(x), dim=1))

        w, e = self.dropout(w), self.dropout(e)

        alloc = torch.transpose(torch.softmax(self.alloc(x), dim=0), 1, 2)
        write_template = torch.matmul(alloc, x)

        with torch.no_grad():
            m = (1-w-e).unsqueeze(-1) * m + w.unsqueeze(-1) * write_template

        r = torch.sigmoid(self.read_gate(x))
        x_fetch = torch.matmul(r, m)

        return m, x_fetch