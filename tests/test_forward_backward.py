import torch

from src.model.encoder import SequenceEncoder
from src.model.omie import OMIE
from src.model.ranker import Ranker

def test_forward_backward():
    # Mock dimensions
    batch_size = 4
    seq_len = 50
    num_repos = 10000
    num_candidates = 200

    # Mock tensors
    history = torch.randint(0, num_repos, (batch_size, seq_len))
    action_types = torch.randint(0, 3, (batch_size, seq_len))
    time_gaps = torch.rand(batch_size, seq_len)
    candidates = torch.randint(0, num_repos, (batch_size, num_candidates))

    # Instantiation
    encoder = SequenceEncoder(num_repos=num_repos)
    omie = OMIE()
    ranker = Ranker(num_repos=num_repos)

    # Forward pass
    event_vectors = encoder(history, action_types, time_gaps)
    interests = omie(event_vectors)
    scores = ranker(interests, candidates)

    assert scores.shape == (batch_size, num_candidates), f"Shape mismatch: {scores.shape}"
    assert not torch.isnan(scores).any(), "NaN in scores"
    print("Forward pass OK")

    # Backward pass (mock loss, no real loss)
    loss = scores.mean()
    loss.backward()

    for module_name, module in [("encoder", encoder), ("omie", omie), ("ranker", ranker)]:
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {module_name}.{name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {module_name}.{name}"

    print("Backward pass OK")


if __name__ == "__main__":
    test_forward_backward()