"""MLE-Core model: sequence encoder -> multi-interest extractor (OMIE-style) -> max-over-interests retrieval/ranking.
Not yet executed by the author (no torch in the sandbox): run `python -m src.train` and report any error."""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class SeqEncoder(nn.Module):
    """item emb (+ optional text vector) + log-age embedding -> small Transformer. Padding is on the left."""
    def __init__(self, n_items, d=64, text_vecs=None, layers=2, heads=2, max_len=50, drop=0.2):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.text = None
        if text_vecs is not None:                       # (n_items+1, t) float tensor, row 0 = zeros
            self.register_buffer("text_vecs", text_vecs)
            self.text = nn.Linear(text_vecs.shape[1], d)
        self.pos = nn.Embedding(max_len, d)
        self.age = nn.Embedding(16, d)                  # log-bucketed age of the event (days)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def item_vectors(self):
        w = self.item.weight
        if self.text is not None:
            w = w + self.text(self.text_vecs)           # cold start: text vector alone still gives a vector
        return w

    def forward(self, items, gaps):
        B, L = items.shape
        x = self.item_vectors()[items]
        bucket = torch.clamp(torch.log2(gaps + 1).long(), 0, 15)
        x = x + self.age(bucket) + self.pos(torch.arange(L, device=items.device))[None]
        pad = items == 0
        h = self.enc(x, src_key_padding_mask=pad)
        return self.norm(h), pad


class MultiInterest(nn.Module):
    """c interest channels: each channel attends to the history with its own query (ComiRec-SA / OMIE spirit)."""
    def __init__(self, d=64, c=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(c, d) * 0.1)
        self.out = nn.Linear(d, d)

    def forward(self, h, pad):
        att = torch.einsum("cd,bld->bcl", self.q, h) / math.sqrt(h.shape[-1])
        att = att.masked_fill(pad[:, None, :], -1e9).softmax(-1)
        z = self.out(torch.einsum("bcl,bld->bcd", att, h))     # (B, c, d)
        return z, att


class InterestDiffusion(nn.Module):
    """Small DDPM on interest vectors: learns to generate extra interests from a user summary.
    Contrastive loss keeps generated interests close to the user's real ones (positives) and far from other users' (negatives)."""
    def __init__(self, d=64, T=10, temp=0.1):
        super().__init__()
        self.T, self.temp = T, temp
        betas = torch.linspace(1e-3, 0.2, T)
        self.register_buffer("ab", torch.cumprod(1 - betas, 0))
        self.temb = nn.Embedding(T, d)
        self.net = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def denoise(self, xt, t, cond):
        return self.net(torch.cat([xt, cond, self.temb(t)], -1))      # predicts x0

    def loss(self, z, cond):
        B, c, d = z.shape
        x0 = z[torch.arange(B), torch.randint(0, c, (B,), device=z.device)].detach()
        t = torch.randint(0, self.T, (B,), device=z.device)
        ab = self.ab[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
        x0h = self.denoise(xt, t, cond.detach())
        mse = F.mse_loss(x0h, x0)
        logits = F.normalize(x0h, dim=-1) @ F.normalize(x0, dim=-1).T / self.temp
        nce = F.cross_entropy(logits, torch.arange(B, device=z.device))
        return mse + nce

    @torch.no_grad()
    def sample(self, cond, n=2):
        B, d = cond.shape
        x = torch.randn(B, n, d, device=cond.device)
        cond = cond[:, None].expand(B, n, d)
        for t in reversed(range(self.T)):
            x0 = self.denoise(x, torch.full((B, n), t, device=cond.device), cond)
            x = x0 if t == 0 else self.ab[t - 1].sqrt() * x0 + (1 - self.ab[t - 1]).sqrt() * torch.randn_like(x0)
        return x


class AgentQuiver(nn.Module):
    def __init__(self, n_items, d=64, c=4, text_vecs=None, diffusion=False, n_gen=2, **kw):
        super().__init__()
        self.enc = SeqEncoder(n_items, d, text_vecs, **kw)
        self.mi = MultiInterest(d, c)
        self.diff = InterestDiffusion(d) if diffusion else None
        self.n_gen = n_gen
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))   # learned temperature for cosine scores

    def sim(self, z):
        """cosine similarity (x learned scale) between interests and ALL items -> (B,c,N+1).
        Cosine = popular items with long vectors no longer win every channel."""
        W = F.normalize(self.enc.item_vectors(), dim=-1)
        return self.log_scale.exp().clamp(max=50) * torch.einsum("bcd,nd->bcn", F.normalize(z, dim=-1), W)

    def interests(self, items, gaps):
        h, pad = self.enc(items, gaps)
        z, att = self.mi(h, pad)
        return z, att

    def scores(self, items, gaps):
        z, _ = self.interests(items, gaps)
        if self.diff is not None and not self.training:           # extra generated interests at inference only
            z = torch.cat([z, self.diff.sample(z.mean(1), self.n_gen)], 1)
        return self.sim(z).max(1).values                          # best match over interests

    def loss(self, items, gaps, tgt, ortho=0.1, w_diff=0.1):
        z, _ = self.interests(items, gaps)
        s = self.sim(z).max(1).values
        s = s.clone(); s[:, 0] = -1e9
        out = F.cross_entropy(s, tgt)
        zn = F.normalize(z, dim=-1)
        gram = zn @ zn.transpose(1, 2)
        out = out + ortho * ((gram - torch.eye(z.shape[1], device=z.device)) ** 2).mean()
        if self.diff is not None:
            out = out + w_diff * self.diff.loss(z, z.mean(1))
        return out


class Ranker(nn.Module):
    """DiffuMIN-style ranker: interests are rebuilt GUIDED BY the candidate item, then scored.
    The retriever's encoder is frozen (its history states are used as-is)."""
    def __init__(self, retriever, d=64, c=4):
        super().__init__()
        self.r = retriever
        self.q = nn.Parameter(torch.randn(c, d) * 0.1)
        self.guide = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, items, gaps, cands):
        with torch.no_grad():
            h, pad = self.r.enc(items, gaps)
            e = self.r.enc.item_vectors()[cands]                   # (B,K,d)
        d = h.shape[-1]
        q = self.q[None, None] + self.guide(e)[:, :, None]         # (B,K,c,d) target-guided queries
        att = torch.einsum("bkcd,bld->bkcl", q, h) / math.sqrt(d)
        att = att.masked_fill(pad[:, None, None, :], -1e9).softmax(-1)
        z = self.out(torch.einsum("bkcl,bld->bkcd", att, h))       # interests guided by the candidate
        w = torch.einsum("bkcd,bkd->bkc", z, e).softmax(-1)
        u = (w[..., None] * z).sum(2)
        return self.mlp(torch.cat([u, e, u * e], -1)).squeeze(-1)   # (B,K) logits
