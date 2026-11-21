"""Manual check required by the plan: for 5 developers print what each interest channel looks at and recommends.
  python -m src.manual_check ckpt_loo_c4.pt"""
import sys, numpy as np, torch
from .data import load, make_samples, pad_batch
from .core_model import AgentQuiver
from .train import text_matrix

ck = torch.load(sys.argv[1]); cfg = ck["cfg"]
inter, items = load()
n_items = int(max(inter["item_id"].max(), items["item_id"].max())) + 1
tv = None if cfg["no_text"] else text_matrix(items, n_items)
m = AgentQuiver(n_items, cfg["d"], cfg["c"], tv, diffusion=cfg["diffusion"]); m.load_state_dict(ck["model"]); m.eval()
name = dict(zip(items["item_id"] + 1, items["item"])); S = make_samples(inter, cfg["protocol"])
for s in S["test"][:5]:
    it, gp, _ = pad_batch([s])
    with torch.no_grad():
        z, att = m.interests(torch.tensor(it), torch.tensor(gp))
        sc = m.sim(z)[0]
    print(f"\n=== user {s[0]} | target: {name.get(s[4])} ===")
    for k in range(z.shape[1]):
        a = att[0, k].numpy(); hist = [name.get(i, "?") for i in it[0][np.argsort(-a)[:3]] if i]
        sc[k, 0] = -1e9
        sc[k, list(s[5])] = -1e9
        rec = [name.get(int(j)) for j in sc[k].topk(3).indices]
        print(f" interest {k}: history -> {hist} | recommends -> {rec}")
