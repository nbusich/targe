import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class SmolVLMInstructConnectorConfig:
    embed_dim: int = 1152       # 1152 * (2^2) for scale_factor=2
    num_selector_heads: int = 8
    selector_hidden_dim: int = 2048
    num_lqv: int = 32           # Number of condensed latents
    k_select: int = 360         # Number of anchor tokens to keep
    device: str = "cuda"

class Idefics3SimpleMLP(nn.Module):
    """
    MLP designed for use without pixel shuffle
    """
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size # Make sure you aren't using pixel shuffle
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)

class DenseAttentionSelector(nn.Module):
    """
    This module estimates importance of each token and divides the
    sequence dimension into two sets

    1. Computes per-token attention scores (importance-score)
    2. Selects q lowest tokens by importance-score
    """
    def __init__(self, embed_dim, num_heads, device):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, device=device)
    
    def forward(self, x, k):
        B, N, D = x.shape

        # 1. Compute attention scores
        _, attn = self.mha(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False
        )  # (B, H, N, N)

        # 2. Compute Importance Scores (Density)
        # Sum across heads, then sum across keys to get per-token importance
        s1 = torch.sum(attn, dim=1)  # (B, N, N)
        s2 = torch.sum(s1, dim=1)    # (B, N)

        # 3. Sort indices by score (Descending)
        # This is the "Batch Secret": sorting keeps B intact and gives us a 
        # map for both Top-K and 'The Rest'.
        _, top_idx = torch.topk(s2, k, dim=1)
        _, sorted_indices = torch.sort(s2, dim=1, descending=True) # (B, N)

        # 4. Partition Indices
        k_idx_raw = sorted_indices[:, :k]      # (B, k)
        q_idx_raw = sorted_indices[:, k:]      # (B, N-k)

        # 5. Expand indices for the Embedding Dimension (D)
        # We need (B, k, D) and (B, N-k, D) for torch.gather
        k_idx = k_idx_raw.unsqueeze(-1).expand(-1, -1, D)
        q_idx = q_idx_raw.unsqueeze(-1).expand(-1, -1, D)

        # 6. Gather the sets
        k_set = torch.gather(x, dim=1, index=k_idx)
        q_set = torch.gather(x, dim=1, index=q_idx)

        return k_set, q_set, top_idx
    
class QTokenCondenser(nn.Module):
    """
    This module compresses the sequence dimension of the lowest importance-score
    tokens using cross attention
    """
    def __init__(self, embed_dim, num_heads, hidden_dim, num_lqv, device):
        super().__init__()
        self.lqv = nn.Parameter(torch.randn(1, num_lqv, embed_dim))

        self.mhca = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device)

        self.norm1 = nn.RMSNorm(embed_dim, device=device)
        self.norm2 = nn.RMSNorm(embed_dim, device=device)

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
    
class Idefics3SelectorConnector(nn.Module):
    """
    Combines the selector, compressor, and projector into a module that
    connects the vision encoder to LLM
    """
    def __init__(self, config, custom_params):
        super().__init__()
        self.modality_projection = Idefics3SimpleMLP(config)
        self.k_select = custom_params.k_select
        self.condensor = QTokenCondenser(embed_dim=custom_params.embed_dim,
                                        num_heads=custom_params.num_selector_heads,
                                        hidden_dim = custom_params.selector_hidden_dim,
                                        num_lqv = custom_params.num_lqv,
                                        device = custom_params.device
                                        )
        self.selector = DenseAttentionSelector(embed_dim=custom_params.embed_dim, 
                                              num_heads = custom_params.num_selector_heads, 
                                              device=custom_params.device
                                              )
    def _init_weights(self):
        """
        The Condenser's final MLP projector initially outputs values close to 0,
        relying heavily on the residual connection.
        """
        nn.init.zeros_(self.condensor.mlp[-1].weight)
        
        if self.condensor.mlp[-1].bias is not None:
            nn.init.zeros_(self.condensor.mlp[-1].bias)

        nn.init.xavier_uniform_(self.selector.mha.in_proj_weight)
        nn.init.zeros_(self.selector.mha.out_proj.weight)

    def forward(self, image_hidden_states):
        k_set, q_set, top_idx = self.selector(image_hidden_states, self.k_select) # Importance selection
        if not self.training:
            self.last_top_idx = top_idx
        condensed_q = self.condensor(q_set) # Compress lowest tokens
        image_hidden_states = torch.cat([k_set, condensed_q], dim=1) 
        image_hidden_states = self.modality_projection(image_hidden_states) # MLP
        return image_hidden_states
