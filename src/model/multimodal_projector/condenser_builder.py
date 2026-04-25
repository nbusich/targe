import torch
import torch.nn as nn
import re

class QTokenCondenser(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_lqv, device):
        super().__init__()
        self.lqv = nn.Parameter(torch.randn(1, num_lqv, embed_dim))

        self.mhca = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device)

        self.norm1 = nn.RMSNorm(embed_dim, device=device)
        self.norm2 = nn.RMSNorm(embed_dim, device=device)
        self.norm3 = nn.RMSNorm(embed_dim, device=device)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
            )
    def forward(self, q):
        B, N, D  = q.shape
        lqv = self.lqv.expand(B, -1, -1)
        x, _ = self.mhca(query=lqv, key=q, value=q)
        x_norm = self.norm1(x)
        x = x_norm + self.mha(query=x_norm, key=x_norm, value=x_norm)[0]
        x_norm = self.norm2(x)
        x = x_norm + self.mlp(x_norm)
        return x
    

def build_selector(config, delay_load=False, **kwargs):
    return QTokenCondenser(
        embed_dim=config.embed_dim,
        num_heads=config.num_selector_heads,
        hidden_dim = config.selector_hidden_dim,
        num_lqv = config.num_lqv,
        device = config.device
        )

# q = torch.randn((16, 400, 768))
# qtc = QTokenCondenser(12, 768, 3072, 64, device='cpu')
# lqv = qtc(q)
# print(lqv.shape)
# print("Should be: [16, 64, 768]")
