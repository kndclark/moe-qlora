# One process: big gated packed run (Lightning-like shapes), then the
# nonaligned case. Same build, no relu2 code involved.
import sys
sys.path.insert(0, sys.argv[1])  # test dir
sys.argv = sys.argv[:1]
import importlib
mod = importlib.import_module("per_commit.test_moe_gptq_int4_accuracy")
name, cls, thr = [b for b in mod.available_backends() if b[0] == "AVXVNNI256GPTQInt4Packed_MOE"][0]
mod.expert_num, mod.hidden_size, mod.intermediate_size = 8, 2048, 1024
mod.num_experts_per_tok, mod.group_size = 2, 64
print("== big gated", flush=True)
mod.run_backend_accuracy_test(name, cls, thr, [(16, False)])
mod.expert_num, mod.hidden_size, mod.intermediate_size = 8, 544, 512
mod.num_experts_per_tok, mod.group_size = 2, 32
print("== nonaligned", flush=True)
try:
    mod.run_backend_accuracy_test(name, cls, thr, [(1, False), (16, False)])
    print("NONALIGNED PASS")
except AssertionError as e:
    print("NONALIGNED FAIL", e)
