"""MLE-Ops smoke training on synthetic data; not a GitHub benchmark."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import SequenceEncoder, OMIE, Ranker


class TrainableRecommender(nn.Module):
    def __init__(self, num_repos=64):
        super().__init__()
        self.num_repos = num_repos
        self.encoder = SequenceEncoder(num_repos, 32, 32)
        self.interests = OMIE(32, 4, 16)
        self.ranker = Ranker(16, num_repos, 16)

    def forward(self, history, actions, gaps, candidates):
        return self.ranker(self.interests(self.encoder(history, actions, gaps)), candidates)

    @torch.no_grad()
    def predict(self, history, actions, gaps, candidates, k=10):
        """Rank supplied unseen candidates. Scores are logits, not probabilities."""
        if k < 1:
            raise ValueError('k must be positive')
        self.eval()
        scores = self(history, actions, gaps, candidates)
        scores = scores.masked_fill(
            (candidates.unsqueeze(-1) == history.unsqueeze(1)).any(-1), -torch.inf
        )
        values, indices = scores.topk(min(k, candidates.shape[1]), dim=1)
        ids = candidates.gather(1, indices)
        # -1 marks positions with no eligible unseen candidate.
        return ids.masked_fill(~torch.isfinite(values), -1), values


def synthetic_data(n, seed, num_repos=64):
    """Independent examples with a learnable toy rule; never real user data.

    History contains a group anchor. The positive is the next repo in that
    group; negatives come from other groups. Candidate order is shuffled.
    """
    g = torch.Generator().manual_seed(seed)
    groups = num_repos // 8
    group = torch.randint(groups, (n,), generator=g)
    history = (group * 8).unsqueeze(1).repeat(1, 6)
    actions = torch.zeros_like(history)
    gaps = torch.arange(6, 0, -1).float().repeat(n, 1)
    candidates = torch.arange(groups).repeat(n, 1) * 8 + 1
    permutations = torch.rand(n, groups, generator=g).argsort(1)
    candidates = candidates.gather(1, permutations)
    targets = (permutations == group.unsqueeze(1)).long().argmax(1)
    return TensorDataset(history, actions, gaps, candidates, targets)


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for history, actions, gaps, candidates, targets in loader:
            loss = nn.functional.cross_entropy(model(history, actions, gaps, candidates), targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            total += loss.item() * len(targets)
            count += len(targets)
    return total / count


def load_checkpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = TrainableRecommender(saved['num_repos'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('artifacts/synthetic'))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    train = DataLoader(synthetic_data(256, args.seed), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(synthetic_data(128, args.seed + 1), batch_size=args.batch_size)
    model = TrainableRecommender()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    initial = run_epoch(model, val)
    best, records = float('inf'), []
    checkpoint = args.output / 'best.pt'
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train, optimizer)
        val_loss = run_epoch(model, val)
        records.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss))
        print(json.dumps(records[-1]))
        if val_loss < best:
            best = val_loss
            torch.save(dict(state_dict=model.state_dict(), num_repos=model.num_repos,
                            epoch=epoch, validation_loss=best, data_kind='synthetic'), checkpoint)
    restored = load_checkpoint(checkpoint)
    reloaded_loss = run_epoch(restored, val)
    assert abs(reloaded_loss - best) < 1e-6, 'Checkpoint reload mismatch'
    assert best < initial, 'Toy training did not improve validation loss'
    batch = next(iter(val))
    ids, scores = restored.predict(*batch[:4], k=3)
    assert ids.shape == (len(batch[-1]), 3)
    assert torch.isfinite(scores).all()
    summary = dict(data_kind='synthetic_only', initial_validation_loss=initial,
                   best_validation_loss=best, reloaded_validation_loss=reloaded_loss,
                   checkpoint_reload_passed=True, prediction_smoke_passed=True,
                   torch_version=str(torch.__version__), config={**vars(args), 'output': str(args.output)})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (args.output / 'losses.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
