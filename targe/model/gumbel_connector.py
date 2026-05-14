import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGumbelAttentionSelector(nn.Module):
    """
    This module estimates the importance of each token and divides the
    sequence dimension into two sets.

    1. Computes context-aware representations via MultiheadAttention.
    2. Projects to importance logits (Keep vs. Compress).
    3. Uses Gumbel-Softmax for differentiable binary masking.
    """
    def __init__(self, embed_dim, num_heads, device):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, device=device)
        
        # New: Maps contextualized tokens to two logits: [Keep, Compress]
        self.router = nn.Linear(embed_dim, 2, device=device)

    def forward(self, x, tau=1.0):
        B, N, D = x.shape

        # 1. Compute attention-enriched token representations
        # We don't need the raw attention weights anymore, just the output features
        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            need_weights=False
        ) 

        # Residual connection keeps original token identity intact for the router
        context_x = x + attn_out

        # 2. Compute Routing Logits
        logits = self.router(context_x) # Shape: (B, N, 2)

        # 3. Differentiable Binary Selection (Straight-Through Estimator)
        # hard=True forces the output to be exactly 0 or 1 for the forward pass,
        # but uses the continuous gradients for the backward pass.
        gumbel_out = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1) # (B, N, 2)

        # 4. Extract continuous-differentiable masks
        keep_mask = gumbel_out[:, :, 0:1]     # (B, N, 1)
        compress_mask = gumbel_out[:, :, 1:2] # (B, N, 1)

        # 5. Mask the sets (Zeroing out instead of gathering)
        # Sequence length N is maintained for training
        k_set = x * keep_mask
        q_set = x * compress_mask

        # 6. Extract soft probabilities for L1 Regularization in your training loop
        # We use standard softmax here because we want the true continuous probability
        keep_probs = F.softmax(logits, dim=-1)[:, :, 0] # (B, N)

        # We return the sets, the masks (for downstream attention blocking), and the probs
        return k_set, q_set, keep_mask, compress_mask, keep_probs

class SelectorCompressorPipeline(nn.Module):
    """
    1. Routes tokens using DenseGumbelAttentionSelector.
    2. Bypasses selected tokens.
    3. Compresses rejected tokens using Learnable Queries (Q-Former).
    4. Feeds the combined sequence into a downstream Transformer.
    """
    def __init__(self, embed_dim, num_heads, num_compressed_tokens, device):
        super().__init__()
        # 1. The Selector
        self.selector = DenseGumbelAttentionSelector(embed_dim, num_heads, device)
        
        # 2. The Compressor (Learnable Queries + Cross Attention)
        self.num_compressed_tokens = num_compressed_tokens
        self.compress_queries = nn.Parameter(torch.randn(1, num_compressed_tokens, embed_dim, device=device))
        self.compress_mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, device=device)
        
        # 3. Downstream Connector (Standard Transformer Layer)
        self.connector = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, device=device)
        
        self.tau = 1.0
        self.inference_threshold = 0.5
        self.latest_keep_probs = None

    def forward(self, x):
        """
        Automatically routes to the correct logic based on model.train() or model.eval()
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_inference(x)

    def _forward_train(self, x):
        B, N, D = x.shape
        
        # 1. Get routing masks (Shape: B, N, 1) and probs
        _, _, keep_mask, compress_mask, keep_probs = self.selector(x, self.tau)

        # 2. BYPASS PATH (Soft Masking)
        # Keep tensor size (B, N, D), zero out rejected tokens. 
        # Gradients flow through this multiplication.
        bypassed_tokens = x * keep_mask 

        # 3. COMPRESSION PATH (Soft Masking)
        queries = self.compress_queries.expand(B, -1, -1) # (B, M, D)
        tokens_to_compress = x * compress_mask            # (B, N, D)
        
        # We must tell PyTorch MHA to ignore the zeroed-out tokens.
        # key_padding_mask expects True for tokens to IGNORE.
        compress_padding_mask = (compress_mask.squeeze(-1) == 0) # (B, N)

        compressed_tokens, _ = self.compress_mha(
            query=queries, 
            key=tokens_to_compress, 
            value=tokens_to_compress, 
            key_padding_mask=compress_padding_mask
        ) # Output Shape: (B, M, D)

        # 4. RECOMBINE FOR CONNECTOR
        # Combine the padded sequence and the M compressed tokens
        combined_sequence = torch.cat([bypassed_tokens, compressed_tokens], dim=1) # (B, N + M, D)

        # Create a padding mask for the downstream connector
        # keep_mask is 1 (keep). Compressed tokens are always kept (1).
        compressed_mask_ones = torch.ones(B, self.num_compressed_tokens, device=x.device)
        connector_mask = torch.cat([keep_mask.squeeze(-1), compressed_mask_ones], dim=1) # (B, N + M)
        
        connector_padding_mask = (connector_mask == 0) # True for tokens to ignore

        # 5. Execute downstream connector
        final_output = self.connector(combined_sequence, src_key_padding_mask=connector_padding_mask)
        self.latest_keep_probs = keep_probs

        return final_output


    @torch.no_grad()
    def _forward_inference(self, x, threshold):
        """
        Assumes Batch Size = 1 for dynamic slicing.
        Uses hard boolean indexing to physically drop tokens and save compute.
        """
        B, N, D = x.shape
        
        # 1. Get raw probabilities (bypass Gumbel noise)
        logits = self.selector.router(x)
        keep_probs = F.softmax(logits, dim=-1)[:, :, 0] # (B, N)
        
        # 2. Hard thresholding to create boolean masks
        keep_bool_mask = keep_probs > threshold
        compress_bool_mask = ~keep_bool_mask
        
        # 3. PHYSICAL EXTRACTION (Sequence length dynamically shrinks here)
        bypassed_tokens = x[keep_bool_mask].unsqueeze(0)        # (1, K_dynamic, D)
        tokens_to_compress = x[compress_bool_mask].unsqueeze(0) # (1, C_dynamic, D)

        # 4. COMPRESSION PATH (Physical)
        queries = self.compress_queries.expand(B, -1, -1)
        
        # Only run cross-attention if there are actually tokens to compress
        if tokens_to_compress.size(1) > 0:
            # No padding mask needed because we physically removed the bypassed tokens!
            compressed_tokens, _ = self.compress_mha(
                query=queries, 
                key=tokens_to_compress, 
                value=tokens_to_compress
            ) # (1, M, D)
        else:
            # If all tokens were kept, compressor outputs zeros to maintain shape
            compressed_tokens = torch.zeros(1, self.num_compressed_tokens, D, device=x.device)

        # 5. RECOMBINE FOR CONNECTOR
        combined_sequence = torch.cat([bypassed_tokens, compressed_tokens], dim=1) # (1, K_dynamic + M, D)

        # 6. Execute downstream connector (No padding mask needed!)
        final_output = self.connector(combined_sequence)
        self.latest_keep_probs = None
        return final_output