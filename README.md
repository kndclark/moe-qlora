# moe-qlora

Can NVIDIA Nemotron 3.5 Lightning (30B-A3B, a hybrid Mamba/attention
mixture-of-experts model) be LoRA-trained on one 24 GB GPU by holding its
routed experts in 4-bit, and does the result beat a Qwen3-8B adapter on a
held-out research-assistant eval?

Everything was run on a two-node home lab: an RTX 5090 Laptop (sm_120) and an
RTX 3090 desktop (sm_86), joined by a direct 2.5 GbE link. The serving,
pooling and eval infrastructure lives in the companion repo,
[GPU-Lab](https://github.com/kndclark/GPU-Lab).

## Where to read

- `docs/plan.md` is the record. Every gate is pre-registered (question,
  run, threshold, expectations) before it runs, and its result is written
  underneath with the files it came from. Claims are tagged MEASURED,
  SOURCED, ARITHMETIC, INFERENCE or UNKNOWN.
- `docs/evidence.md` is the Phase 0 evidence the plan starts from.

## Layout

- `probes/`: every script a gate ran: memory and fidelity probes, training
  (`g6_train.py`), dataset builders (`g6*_build.py`), eval and serving
  runners (`g6_eval.sh`, `g7*`), noise studies and comparisons.
- `results/`: the outputs those scripts wrote: eval JSON and logs, datasets,
  traces, serve logs. Adapter weights are not committed.
- `probes/ssu_sm120/`: a side quest on FlashInfer's Mamba kernel on sm_120,
  with a local header patch.

## License

MIT (`LICENSE`), except the third-party material listed in `NOTICE`:
databricks-dolly-15k records (CC BY-SA 3.0) and FlashInfer headers
(Apache-2.0).
