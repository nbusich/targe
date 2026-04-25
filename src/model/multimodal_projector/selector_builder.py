import torch
import torch.nn as nn
import re

class DenseAttentionSelector(nn.Module):
    def __init__(self, embed_dim, num_heads, device):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, device=device)
    
    def forward(self, x, k):
        B, N, D = x.shape
        out, attn = self.mha(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False
        )  # (B, H, N, N)

        s1 = torch.sum(attn, dim=1)
        s2 = torch.sum(s1, dim=1)

        _, idx = torch.topk(s2, k, dim=1)
        
        # selected top-k tokens
        k_idx = idx.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
        k_set = torch.gather(x, dim=1, index=k_idx)  # (B, k, D)

        # mask out selected tokens
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(dim=1, index=idx, value=False)  # False at selected indices

        # remaining tokens
        q_set = x[mask].view(B, N - k, D)  # (B, N-k, D)

        return k_set, q_set


def build_selector(config, delay_load=False, **kwargs):
    return DenseAttentionSelector(
        embed_dim = config.embed_dim, 
        num_heads = config.num_selector_heads, 
        device = config.device
        )

# x = torch.randn((16, 1024, 768))
# das = DenseAttentionSelector(768, 12, device='cpu')
# kset, qset = das(x, 100)
# print(kset.shape, qset.shape)