# vLLM v0.29.0 plus PR #37190 (MoE expert LRU cache), ported to v0.29.0. The port differs
# from the PR in three places: moe_runner.py keeps v0.29.0's cast() wrappers, fp8.py imports
# Fp8MoeBackend where the PR's property names it, and online/fp8.py gains the cache hooks so
# a BF16 checkpoint quantized at load (--quantization fp8) can use it.
FROM vllm/vllm-openai:v0.29.0
COPY v029-pr37190.patch /tmp/
RUN patch -p1 -d /usr/local/lib/python3.12/dist-packages < /tmp/v029-pr37190.patch \
 && rm /tmp/v029-pr37190.patch
