"""Are Lightning's routed experts close enough to store as deltas from one reference?

For delta coding to pay, ||W_i - ref|| must be well under ||W_i|| (bar: < 0.5; a 2-bit
delta beats a 4-bit weight only near < 0.25). Reads BF16 experts from the local checkpoint
by safetensors header offset. Raw comparison, plus a permutation-aware bound: experts may
match only after reordering their hidden neurons, so for sampled pairs each up_proj row of
expert i is matched to its best row in expert j (greedy max |cos|, an upper bound on what
any alignment achieves: many-to-one and sign flips allowed, so looser than a real
permutation) and the down_proj columns follow the same match.
env: S (snapshot dir), LAYERS (default: first, middle, last MoE layer), PAIRS (8), T (threads)

CPU only, in the serving image (it has torch); results/expert-delta/out.txt came from
  R=/srv/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
  nice -n 19 docker run --rm -i --network none --pull never --cpus 4 -e T=4 \
    -e S=/r/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9 -v $R:/r:ro \
    --entrypoint python3 vllm/vllm-openai:v0.29.0 - < probes/expert_delta.py
"""
import json, os, struct, random
import torch

S = os.environ.get("S", "/m")
torch.set_num_threads(int(os.environ.get("T", "4")))
idx = json.load(open(f"{S}/model.safetensors.index.json"))["weight_map"]
hdr = {}


def read(name):
    f = f"{S}/{idx[name]}"
    with open(f, "rb") as b:
        n = struct.unpack("<Q", b.read(8))[0]
        if f not in hdr:
            hdr[f] = json.loads(b.read(n))
        m = hdr[f][name]
        s, e = m["data_offsets"]
        b.seek(8 + n + s)
        raw = b.read(e - s)
    assert m["dtype"] == "BF16", (name, m["dtype"])
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().reshape(m["shape"])


moe = sorted({int(k.split(".")[2]) for k in idx if k.startswith("backbone.") and ".experts.0.up_proj" in k})
E = len({k.split(".")[5] for k in idx if k.startswith(f"backbone.layers.{moe[0]}.mixer.experts.")})
layers = [int(x) for x in os.environ["LAYERS"].split()] if os.environ.get("LAYERS") else [moe[0], moe[len(moe) // 2], moe[-1]]
PAIRS = int(os.environ.get("PAIRS", "8"))
print(f"{len(moe)} MoE layers, {E} experts each; testing layers {layers}")
rng = random.Random(20261005)
for l in layers:
    up = torch.stack([read(f"backbone.layers.{l}.mixer.experts.{i}.up_proj.weight") for i in range(E)])
    dn = torch.stack([read(f"backbone.layers.{l}.mixer.experts.{i}.down_proj.weight") for i in range(E)])
    zr = up.norm(dim=2) == 0
    print(f"\nlayer {l}: up {tuple(up.shape[1:])}, down {tuple(dn.shape[1:])}; "
          f"all-zero up rows {int(zr.sum())}, all-zero experts {int(zr.all(1).sum())}")
    for nm, W in (("up", up), ("down", dn)):
        F = W.flatten(1)
        nrm = F.norm(dim=1)
        r = (F - F.mean(0)).norm(dim=1) / nrm
        Fn = F / nrm[:, None]
        C = Fn @ Fn.T
        off = C[~torch.eye(E, dtype=torch.bool)].abs()
        near = []
        for i in range(E):
            d = (F - F[i]).norm(dim=1) / nrm[i]
            d[i] = float("inf")
            near.append(d.min().item())
        near = torch.tensor(near)
        print(f"  {nm:4} ||W-mean||/||W|| median {r.median():.3f} min {r.min():.3f} | "
              f"pairwise |cos| mean {off.mean():.3f} max {off.max():.3f} | "
              f"nearest other expert: delta ratio median {near.median():.3f} min {near.min():.3f}")
    # permutation-aware bound on sampled pairs: rows of up_proj are hidden neurons
    best = []
    for _ in range(PAIRS):
        i, j = rng.sample(range(E), 2)
        a = up[i] / up[i].norm(dim=1, keepdim=True).clamp_min(1e-12)  # dead rows stay 0, not NaN
        b = up[j] / up[j].norm(dim=1, keepdim=True).clamp_min(1e-12)
        cs = a @ b.T
        m, arg = cs.abs().max(dim=1)
        sgn = torch.sign(cs.gather(1, arg[:, None])).squeeze(1)
        sgn[sgn == 0] = 1
        upj = up[j][arg] * sgn[:, None]
        dnj = dn[j][:, arg] * sgn[None, :]
        ru = (up[i] - upj).norm() / up[i].norm()
        rd = (dn[i] - dnj).norm() / dn[i].norm()
        live = up[i].norm(dim=1) > 0
        best.append((m[live].mean().item(), ru.item(), rd.item()))
    t = torch.tensor(best)
    print(f"  aligned pairs ({PAIRS}): mean best-row |cos| {t[:, 0].mean():.3f}; "
          f"delta ratio after alignment up {t[:, 1].median():.3f} down {t[:, 2].median():.3f} (median)")
print("\ndone")
