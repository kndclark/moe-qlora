"""KSTAGE_PFFIX checks on the CPU, without vLLM: a model of PrefetchOffloader's schedule.
Module i reads slot i % s; after its forward, it starts the copy of module (i + s) % n. Each
copy is taken to land the moment it starts (the worst case for overwriting a slot), and every
read must find its own module in the slot, over several passes. Stock vLLM must fail exactly
when n % s != 0; _pf_fix must pass for every n and s, and still start one copy per module.
usage: python3 pffix_test.py   (in the vLLM image, or anywhere torch imports)"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_kstage as ks  # noqa: E402


def make_cls():
    class Fake:
        def __init__(self, n, s):
            self.module_offloaders = list(range(n))
            self.prefetch_step = s
            self.slot = [None] * s
            self.started = []

        def _start_prefetch(self, j):  # stock: the copy lands at once
            self.slot[j % self.prefetch_step] = j
            self.started.append(j)
    return Fake


def run(cls, n, s, passes=4):
    o = cls(n, s)
    for j in range(min(s, n)):  # post_init's first copies
        o.slot[j % s] = j
    bad = 0
    for _ in range(passes):
        for i in range(n):
            bad += o.slot[i % s] != i
            o._start_prefetch((i + s) % n)
    return bad, o.started


def main():
    stock, fixed = make_cls(), make_cls()
    ks._pf_fix(fixed)
    for n in range(1, 25):
        for s in range(1, min(n, 6) + 1):
            b0, _ = run(stock, n, s)
            b1, st = run(fixed, n, s)
            assert (b0 > 0) == (n % s != 0), f"n={n} s={s}: stock bad reads {b0}"
            assert b1 == 0, f"n={n} s={s}: fixed bad reads {b1}"
            assert sorted(st) == sorted(list(range(n)) * 4), f"n={n} s={s}: copies {st}"
    b0, _ = run(stock, 7, 2)
    print(f"Lightning G4/1 (n=7) step 2: stock {b0} wrong reads in 4 passes, fixed 0")
    print("all pffix checks passed")


if __name__ == "__main__":
    main()
