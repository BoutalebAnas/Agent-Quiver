"""OMIE interest extractor (adapted from DiffuMIN).

Input:
    - event_vectors : (batch, seq_len, input_dim)

Output:
    - interests : (batch, num_interests, interest_dim) normalized interest vectors
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OMIE(nn.Module):
    def __init__(self, input_dim: int = 128, num_interests: int = 4, interest_dim: int = 64):
        super().__init__()
        self.num_interests = num_interests
        self.interest_dim = interest_dim
        self.proj = nn.Linear(input_dim, num_interests * interest_dim)

    def forward(self, event_vectors):
        # Simple pooling over the sequence
        pooled = event_vectors.mean(dim=1) # (batch, input_dim)
        projected = self.proj(pooled) # (batch, num_interests * interest_dim)
        interests = projected.view(-1, self.num_interests, self.interest_dim)
        # L2 normalization for stability
        interests = F.normalize(interests, dim=-1)
        return interests # (batch, num_interests, interest_dim)