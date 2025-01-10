import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Step 2: Prepare inputs for scoring
def prepare_inputs(pool, pack, card_to_idx, device):
    """
    Prepare inputs for the model
    """
    pool_indices = [card_to_idx.get(card, 0) for card in pool]
    pack_indices = [card_to_idx.get(card, 0) for card in pack]

    pool_tensor = torch.tensor(pool_indices, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dim
    pack_tensor = torch.tensor(pack_indices, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dim

    return pool_tensor, pack_tensor


class DraftPicker(nn.Module):
    def __init__(self, embedding_dim, num_cards):
        super(DraftPicker, self).__init__()
        self.internal_dims = 8
        
        self.card_embedding = nn.Embedding(num_cards, embedding_dim)
        
        # Extra layers
        self.win_embedding = nn.Linear(1, embedding_dim)
        self.rank_embedding = nn.Linear(1, embedding_dim)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        
        # Q, K, V projections and final output
        self.query_proj = nn.Linear(embedding_dim, self.internal_dims, bias=False)
        self.key_proj = nn.Linear(embedding_dim, self.internal_dims, bias=False)
        self.value_proj = nn.Linear(embedding_dim, self.internal_dims, bias=False)
        self.scorer = nn.Linear(self.internal_dims, 1)

        self.sqrt_d = math.sqrt(self.internal_dims)
    
    def forward(self, pool, pack, wins, ranks):
        
        # Embed pool, pack, wins, ranks
        pool_emb = self.card_embedding(pool)
        pack_emb = self.card_embedding(pack)
        win_emb = self.win_embedding(wins.unsqueeze(-1)).unsqueeze(1)
        rank_emb = self.rank_embedding(ranks.unsqueeze(-1)).unsqueeze(1)
        
        # Combine pool embeddings
        pool_emb = torch.cat([pool_emb, win_emb, rank_emb], dim=1)
        pool_emb = self.relu(pool_emb)

        # Cross-attention
        pack_q = self.query_proj(pack_emb)      # (B, n_pack, internal_dims)
        pool_k = self.key_proj(pool_emb)        # (B, n_pool, internal_dims)
        pool_v = self.value_proj(pool_emb)      # (B, n_pool, internal_dims)

        scores = torch.matmul(pack_q, pool_k.transpose(-1, -2)) / self.sqrt_d
        scores = self.relu(scores)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, pool_v)    # (B, n_pack, internal_dims)

        logits = self.scorer(self.leaky(context)).squeeze(-1)

        return logits
    
  