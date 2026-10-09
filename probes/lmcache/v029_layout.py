"""Patches the LMCache 0.5.4 inside vllm/vllm-openai:v0.29.0 so LMCacheMPConnector can register KV caches.

LMCache asks vLLM for the KV layout through vllm.v1.attention.backends.utils.get_kv_cache_layout(),
which vLLM 0.29.0 removed: the engine core now resolves one layout per model and records it on
cache_config.kv_cache_layout under new names (LBNHC is the old NHD, LBHNC the old HND). The import
fails, the lookup returns None, and registration stops with "Unsupported kv_layout: none" (d45).
The connector holds the vLLM config, so it records the resolved layout where LMCache's lookup reads
it; the heartbeat's re-registration reads the same value. Second (d46): vLLM 0.29 hands each Mamba
layer over as an int8 [blocks, 1, 1, state bytes] view, and LMCache re-views that as block_size
tokens of state bytes / block_size each, which fails when the state does not divide (Lightning's is
2,134,016 bytes against 4,176-token blocks). The padded page does divide, since vLLM pads Mamba pages
to the attention page (4,176 x 512 bytes with fp8 KV), so the view covers the whole page. Third (d47): LMCache sorts a view's dims by stride before it
reads the format, so the size-1 dim must not carry a stride above the block tokens', or the page
reads as 1 token of 4,176 heads ("Number of heads (4176) exceeds max threads per block").
Edits the container's own copy, in place.
Usage (inside the container, before vllm serve): python3 v029_layout.py
"""
import importlib.util
import pathlib
import sys

MARK = "v029_layout.py"
root = pathlib.Path(importlib.util.find_spec("lmcache").submodule_search_locations[0]) / "integration/vllm"
edits = {
    "utils.py": [
        ('ENGINE_NAME = "vllm-instance"\n',
         'ENGINE_NAME = "vllm-instance"\n'
         f"RESOLVED_LAYOUT = None  # {MARK}: set by LMCacheMPConnector.register_kv_caches\n"),
        ("    try:\n        # Third Party\n        from vllm.v1.attention.backends.utils import (",
         f"    if RESOLVED_LAYOUT is not None:  # {MARK}\n        return RESOLVED_LAYOUT\n"
         "    try:\n        # Third Party\n        from vllm.v1.attention.backends.utils import ("),
    ],
    "kv_cache_group_edits.py": [
        ('        kv_layout = layout_hints.get("kv_layout", "none")\n'
         '        if kv_layout == "NHD":\n'
         "            return kv_cache.view(kv_cache.shape[0], spec.block_size, 1, -1)\n",
         f"        # {MARK}: vLLM 0.29 views a Mamba page as int8 [B, 1, 1, state bytes] whose block\n"
         "        # stride is the padded page; the state alone need not divide by block_size\n"
         "        # (Lightning: 2,134,016 bytes, 4,176 tokens), so view the whole padded page\n"
         "        # (conv | ssm | pad), as _MambaPageViewEdit does for older vLLM\n"
         '        kv_layout = layout_hints.get("kv_layout", "none")\n'
         "        page = spec.page_size_bytes // kv_cache.element_size()\n"
         '        if kv_layout in ("NHD", "HND") and kv_cache.shape[1] == kv_cache.shape[2] == 1 \\\n'
         "                and kv_cache.stride(-1) == 1 and page % spec.block_size == 0:\n"
         "            c = page // spec.block_size\n"
         '            n, h = (spec.block_size, 1) if kv_layout == "NHD" else (1, spec.block_size)\n'
         "            return kv_cache.as_strided((kv_cache.shape[0], n, h, c),\n"
         "                                       (kv_cache.stride(0), h * c, c, 1))\n"
         '        if kv_layout == "NHD":\n'
         "            return kv_cache.view(kv_cache.shape[0], spec.block_size, 1, -1)\n"),
    ],
    "lmcache_mp_connector.py": [
        ("        layout_hints = vllm_layout_hints()\n",
         f"        # {MARK}: vLLM 0.29 resolves the layout once and records it on cache_config\n"
         "        import lmcache.integration.vllm.utils as _u\n"
         '        _u.RESOLVED_LAYOUT = {"LBNHC": "NHD", "LBHNC": "HND"}[\n'
         "            self._vllm_config.cache_config.kv_cache_layout]\n"
         "        layout_hints = vllm_layout_hints()\n"),
    ],
}
for name, pairs in edits.items():
    path = root / name
    text = path.read_text()
    if MARK in text:
        print(f"{MARK}: {name} already patched")
        continue
    for old, new in pairs:
        if text.count(old) != 1:
            sys.exit(f"{MARK}: {name}: expected one anchor, found {text.count(old)}: {old[:60]!r}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"{MARK}: patched {name}")
