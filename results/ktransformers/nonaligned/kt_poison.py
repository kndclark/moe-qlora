import sys
sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[:1]
import importlib
mod = importlib.import_module("per_commit.test_moe_gptq_int4_accuracy")
name, cls, thr = [b for b in mod.available_backends() if b[0] == "AVXVNNI256GPTQInt4Packed_MOE"][0]
mod.hidden_size, mod.group_size = 544, 32
try:
    mod.run_backend_accuracy_test(name, cls, thr, [(1, False), (16, False)]); print("NONALIGNED PASS")
except AssertionError as e:
    print("NONALIGNED FAIL", e)
