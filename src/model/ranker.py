"""DiffuMIN-style ranker: scores each candidate for each interest.

Input:
    - interests : (batch, num_interests, interest_dim)
    - candidates : (batch, num_candidates)

Output:
    - scores : (batch, num_candidates)
"""
import torch
import torch.nn as nn

class Ranker(nn.Module):
    def __init__(self, interest_dim: int = 64, num_repos: int = 10000, embed_dim: int = 64):
        super().__init__()
        self.repo_embedding = nn.Embedding(num_repos, embed_dim)
        self.score_proj = nn.Linear(interest_dim, embed_dim)

    def forward(self, interests, candidates):
        # Candidate embeddings
        cand_emb = self.repo_embedding(candidates) # (batch, num_cand, embed_dim)
        # Project interests into the candidate space
        interest_proj = self.score_proj(interests) # (batch, num_int, embed_dim)
        # Score = dot product between interest and candidate
        scores = torch.einsum("bid,bjd->bij", interest_proj, cand_emb) # (batch, num_int, num_cand)
        # Max over interests (a small interest can still score)
        scores = scores.max(dim=1).values # (batch, num_cand)
        return scores