"""Sequence encoder: turns the user history into event vectors.

Input:
    - history : (batch, seq_len) repo IDs
    - action_types : (batch, seq_len) 0=star, 1=fork, 2=contribution
    - time_gaps : (batch, seq_len) days since the action (float)

Output:
    - event_vectors : (batch, seq_len, hidden_dim)
"""
import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(self, num_repos: int = 10000, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.repo_embedding = nn.Embedding(num_repos, embed_dim)
        self.action_embedding = nn.Embedding(3, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, history, action_types, time_gaps):
        # Repo embedding + action type embedding + time gap projection
        x = (
            self.repo_embedding(history)
            + self.action_embedding(action_types)
            + self.time_proj(time_gaps.unsqueeze(-1))
        )
        out, _ = self.rnn(x)
        return out  # (batch, seq_len, hidden_dim)