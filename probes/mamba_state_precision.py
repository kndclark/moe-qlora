"""Mamba SSM state storage precision, simulated on CPU (no GPU, no activations).

vLLM decode reads each layer's SSM state from the cache, updates it in fp32
(h = exp(dt*A) h + dt x B^T), emits y = h C + D x, and writes h back. vLLM pins
the cache to fp32 for NemotronH. This replays that loop with the checkpoint's
real A_log, dt_bias and D for every Mamba layer and synthetic x, B, C, dt_raw,
storing the state in each candidate format between steps, and compares y with
the fp32 run. Heads are independent in the recurrence, so per-head errors also
score a per-head mixed allocation. Rows (head_dim) are independent under
per-row scales, so P rows of 64 are simulated.

A pass here is necessary, not sufficient: inputs are Gaussian, not captured
activations. Prefill quantizes once per chunk, not per token, so the per-token
store here is the harsher case.

env: S (snapshot dir holding model.safetensors.index.json, default /m),
STEPS (4096), MU/SD (dt_raw ~ N(MU, SD)), P (16), T (threads),
SCHEMES (space-separated subset of fp16 bf16 fp8row int8row int8row-sr; all if unset)

CPU only, in the serving image (it has torch); results/mamba-state/ came from
  R=/srv/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
  docker run --rm -i --network none --pull never --cpus 12 -e STEPS=4096 -e MU=0 \
    -e S=/r/snapshots/bee7596271d1495f6992ae224aefde4410e816b8 -v $R:/r:ro \
    --entrypoint python3 vllm/vllm-openai:v0.29.0 - < probes/mamba_state_precision.py
(full-mu0: all schemes; long16k-mu0: STEPS=16384, mu-1: MU=-1, both SCHEMES='fp16 bf16')
"""
import json, os, struct, time
import torch

S = os.environ.get("S", "/m")
STEPS = int(os.environ.get("STEPS", "4096"))
MU, SD = float(os.environ.get("MU", "0")), float(os.environ.get("SD", "1"))
P, N, G = int(os.environ.get("P", "16")), 128, 8
torch.set_num_threads(int(os.environ.get("T", "12")))
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


Ls = sorted({int(k.split(".")[2]) for k in idx if k.startswith("backbone.") and k.endswith(".mixer.A_log")})
A = -torch.stack([read(f"backbone.layers.{l}.mixer.A_log") for l in Ls]).exp()
dtb = torch.stack([read(f"backbone.layers.{l}.mixer.dt_bias") for l in Ls])
Dp = torch.stack([read(f"backbone.layers.{l}.mixer.D") for l in Ls])
L, H = A.shape
rate = torch.nn.functional.softplus(dtb + MU) * A.abs()  # per-step decay at the dt_raw mean
qs = torch.quantile(rate.flatten(), torch.tensor([0, 0.1, 0.5, 0.9, 1.0]))
print(f"{L} mamba layers x {H} heads, {P} of 64 rows; dt_raw ~ N({MU}, {SD}); {STEPS} steps")
print("decay rate dt*|A| at the dt_raw mean, quantiles 0/10/50/90/100%: "
      + " ".join(f"{x:.1e}" for x in qs)
      + f"; heads below 1e-2: {(rate < 1e-2).sum().item()}, below 1e-3: {(rate < 1e-3).sum().item()} of {L * H}")


def rowscale(h, qmax):
    s = (h.abs().amax(-1, keepdim=True) / qmax).half().float()  # fp16 scale per row
    return torch.where(s > 0, s, torch.ones_like(s))


def int8(h):
    s = rowscale(h, 127)
    return (h / s).round().clamp(-127, 127) * s


def int8sr(h):
    s = rowscale(h, 127)
    return (h / s + torch.rand_like(h)).floor().clamp(-127, 127) * s


def fp8(h):
    s = rowscale(h, 448)
    return (h / s).to(torch.float8_e4m3fn).float() * s


Q = {"fp16": lambda h: h.half().float(), "bf16": lambda h: h.bfloat16().float(),
     "fp8row": fp8, "int8row": int8, "int8row-sr": int8sr}
if os.environ.get("SCHEMES"):
    Q = {k: Q[k] for k in os.environ["SCHEMES"].split()}
BYTES = {"fp16": 2, "bf16": 2, "fp8row": 1 + 2 / N, "int8row": 1 + 2 / N, "int8row-sr": 1 + 2 / N}
g = torch.Generator().manual_seed(20261004)
torch.manual_seed(1)
hs = {k: torch.zeros(L, H, P, N) for k in ["ref", *Q]}
acc = {k: torch.zeros(L, H) for k in Q}
accref = torch.zeros(L, H)
peak = {k: 0.0 for k in Q}
hmax = 0.0
gi = H // G
t0 = time.time()
for t in range(1, STEPS + 1):
    x = torch.randn(L, H, P, generator=g)
    Bm = torch.randn(L, G, N, generator=g).repeat_interleave(gi, 1)[:, :, None, :]
    Cm = torch.randn(L, G, N, generator=g).repeat_interleave(gi, 1)[:, :, None, :]
    dt = torch.nn.functional.softplus(MU + SD * torch.randn(L, H, generator=g) + dtb)
    dA = torch.exp(dt * A)[..., None, None]
    inj = (dt[..., None] * x)[..., None] * Bm
    dx = Dp[..., None] * x
    h = dA * hs["ref"] + inj
    yr = (h * Cm).sum(-1) + dx
    hs["ref"] = h
    hmax = max(hmax, h.abs().max().item())
    nr = (yr ** 2).sum(-1)
    tail = t > STEPS - 256
    if tail:
        accref += nr
    for k, q in Q.items():
        h = dA * hs[k] + inj
        e = (((h * Cm).sum(-1) + dx - yr) ** 2).sum(-1)
        hs[k] = q(h)
        peak[k] = max(peak[k], (e.sum(-1) / nr.sum(-1)).sqrt().max().item())
        if tail:
            acc[k] += e
    if t in (64, 256, 1024, 2048, 4096, 8192, STEPS):
        print(f"t={t:5} {time.time() - t0:4.0f}s  peak worst-layer rel err of y so far: "
              + "  ".join(f"{k} {peak[k]:.1e}" for k in Q), flush=True)

print(f"max |h| in fp32: {hmax:.1f} (fp16 max 65504)")
print("\nlast 256 steps, relative error of y:  worst layer | heads under 1e-2 | under 3e-2 | worst head")
rh = {k: (acc[k] / accref).sqrt() for k in Q}
for k in Q:
    lay = (acc[k].sum(-1) / accref.sum(-1)).sqrt()
    print(f"  {k:10} {lay.max():.1e} | {(rh[k] < 1e-2).sum().item():4} | {(rh[k] < 3e-2).sum().item():4} | {rh[k].max():.1e}")

print("\nby decay rate (dt*|A| at the dt_raw mean): heads, and share under 1e-2 per format")
edges = [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1e9]
for lo, hi in zip(edges, edges[1:]):
    m = (rate >= lo) & (rate < hi)
    if m.sum() == 0:
        continue
    print(f"  [{lo:.0e}, {hi:.0e}) {m.sum().item():4} heads  "
          + "  ".join(f"{k} {(rh[k][m] < 1e-2).float().mean().item():.0%}" for k in Q))

for lo_fmt in [k for k in ("int8row", "int8row-sr", "fp8row") if k in Q and "fp16" in Q]:
    # static rule: a head stores 1 byte if its error passes, else fp16 if that passes, else fp32
    one = rh[lo_fmt] < 1e-2
    two = ~one & (rh["fp16"] < 1e-2)
    four = ~one & ~two
    e = torch.where(one, acc[lo_fmt], torch.where(two, acc["fp16"], torch.zeros_like(acc["fp16"])))
    b = (one.sum() * BYTES[lo_fmt] + two.sum() * 2 + four.sum() * 4).item() / (L * H)
    lay = (e.sum(-1) / accref.sum(-1)).sqrt()
    print(f"\nmixed {lo_fmt}/fp16/fp32 per head (in-sample choice): {one.sum().item()}/{two.sum().item()}/"
          f"{four.sum().item()} heads, {b:.2f} bytes per element vs 4 ({4 / b:.1f}x), worst layer {lay.max():.1e}")
print(f"\ndone in {time.time() - t0:.0f}s")
