"""KSTAGE=predict checks on the CPU, on a toy model whose residual stream is the same at every
MoE layer: then "scaled" (x_i * w_j / w_i) is exactly layer j's input, so it must predict layer
j's routing, "own" must reproduce the router, and every counter is recomputed here and compared.
usage: python3 predict_test.py   (in the vLLM image; no GPU)"""
import os, sys, types

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_kstage as ks  # noqa: E402

E, H, KR = 16, 32, 6
IDX = [1, 3, 5, 7, 9]
torch.manual_seed(0)


class Runner(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = object()
        self.routed_experts = types.SimpleNamespace(
            w13_weight=torch.zeros(E, 4), quant_method=types.SimpleNamespace(is_monolithic=False),
            forward_modular=lambda x, w, ids, *a, **k: ("fwd", ids))


def model():
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleDict()
    for i in IDX:
        dec = nn.Module()
        dec.norm = nn.Module()
        dec.norm.weight = nn.Parameter(torch.rand(H) + 0.5, requires_grad=False)
        dec.mixer = nn.Module()
        dec.mixer.gate = nn.Module()
        dec.mixer.gate.weight = nn.Parameter(torch.randn(E, H), requires_grad=False)
        dec.mixer.gate.e_score_correction_bias = nn.Parameter(torch.randn(E) * 0.1, requires_grad=False)
        dec.mixer.experts = Runner()
        root.model.layers[str(i)] = dec
    return root


def route(dec, h):
    x = h / h.pow(2).mean(-1, keepdim=True).sqrt() * dec.norm.weight
    g = dec.mixer.gate
    return x, ((x @ g.weight.t()).sigmoid() + g.e_score_correction_bias).topk(KR, dim=-1).indices.int()


def main():
    m = model()
    layers = ks._layers(m)
    assert [i for i, _, _ in layers] == IDX, layers
    counts = {i: torch.randint(0, 1000, (E,)).tolist() for i in IDX}
    cold_n = {i: 4 for i in IDX}
    host = ks._install_predict(layers, m, cold_n, counts, device="cpu")
    cold = {}
    for i in IDX:
        perm = sorted(range(E), key=lambda e: (-counts[i][e], -e))
        cold[i] = set(perm[E - 4:])
    decs = [m.model.layers[str(i)] for i in IDX]
    exp = torch.zeros_like(host)  # the counters, recomputed for the slots that can be: own, d1/d2 scaled
    runs = {0: 0, 1: 0}
    with torch.inference_mode():
        for M in (1, 4, 16, 64, 65, 300, 7, 200):
            ph = 0 if M <= 64 else 1
            runs[ph] += 1
            h = torch.randn(M, H)
            routed = []
            for li, dec in enumerate(decs):
                x, ids = route(dec, h)
                routed.append(ids)
                out = dec.mixer.experts.routed_experts.forward_modular(x, None, ids)
                assert out[0] == "fwd" and out[1] is ids, "the wrapped call changed"
                act = [set(r) for r in ids.tolist()]
                ua = set().union(*act)
                for si in [0] + ([2] if li >= 1 else []) + ([4] if li >= 2 else []):
                    for ki in range(len(ks.PRED_K)):  # exact predictions: top6 is the routed set itself
                        e = exp[li, si, ki, ph]
                        e[0] += 1; e[1] += M; e[2] += M * KR; e[3] += len(ua); e[4] += len(ua)
                        e[6] += len(ua & cold[IDX[li]]); e[7] += len(ua & cold[IDX[li]])
    got = host
    for si, name in ((0, "own"), (2, "d1 scaled"), (4, "d2 scaled")):
        for ki, k in enumerate(ks.PRED_K):
            for ph in (0, 1):
                g, e = got[:, si, ki, ph], exp[:, si, ki, ph]
                # steps, tokens, union of routed ids, cold routed: exact; hits: exact at top6, at least at top10
                for f in (0, 1, 3, 6):
                    assert torch.equal(g[:, f], e[:, f]), f"{name} top{k} ph{ph} {ks.PRED_FIELDS[f]}: {g[:, f]} != {e[:, f]}"
                cmp = torch.equal if k == KR else (lambda a, b: bool((a >= b).all()))
                for f in (2, 4, 7):
                    assert cmp(g[:, f], e[:, f]), f"{name} top{k} ph{ph} {ks.PRED_FIELDS[f]}: {g[:, f]} vs {e[:, f]}"
                if k == KR:
                    assert torch.equal(g[:, 5], e[:, 3]), f"{name} top6: fetched set != routed set"
    raw = got[:, 1, 0, :, 2].sum() / (got[:, 1, 0, :, 1].sum() * KR)
    assert got[0, 1:].sum() == 0 and got[1, 3:].sum() == 0, "layers too early for d1/d2 counted"
    assert raw < 0.9, f"raw d1 recall {raw:.3f}: the toy should not make raw exact"
    print("\n".join(ks._predict_lines(got.sum(0))))
    print(f"runs {runs}; own and scaled exact, raw d1 top6 recall {raw:.3f}; all predict checks passed")


if __name__ == "__main__":
    main()
