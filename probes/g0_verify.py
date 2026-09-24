"""G0: is the downloaded Lightning snapshot complete and byte-identical to the pinned revision?

Checks every file in results/g0-manifest.json (the HF tree API at rev a9904d24):
size, then sha256 for LFS files and the git blob sha1 for the rest. Also records
owner and mode, since a 0600 file owned by root is unreadable from the laptop
over NFS (root squash). Run on the laptop, so the reads take the same NFS path
G4 will use. Standard library only, so it runs on the host:
  python3 probes/g0_verify.py results/g0-manifest.json \
    /srv/model-cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/<rev> \
    results/g0-verify.json
"""
import hashlib
import json
import os
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor

manifest_path, snap, out_path = sys.argv[1:4]
man = json.load(open(manifest_path))


def check(item):
    rel, want = item
    p = os.path.join(snap, rel)
    r = {"file": rel}
    if not os.path.exists(p):
        return {**r, "ok": False, "error": "missing"}
    real = os.path.realpath(p)
    st = os.stat(real)
    r.update(size=st.st_size, uid=st.st_uid, mode=oct(stat.S_IMODE(st.st_mode)))
    if st.st_size != want["size"]:
        return {**r, "ok": False, "error": f"size {st.st_size} != {want['size']}"}
    if want["sha256"]:
        h, kind, expect = hashlib.sha256(), "sha256", want["sha256"]
    else:
        h, kind, expect = hashlib.sha1(), "git_blob_sha1", want["git_oid"]
        h.update(f"blob {st.st_size}\0".encode())
    t = time.time()
    with open(real, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    got = h.hexdigest()
    return {**r, "hash": kind, "ok": got == expect, "got": got, "expect": expect,
            "seconds": round(time.time() - t, 1)}


t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:  # hashlib drops the GIL on large updates
    rows = list(ex.map(check, sorted(man["files"].items())))
bad = [r for r in rows if not r["ok"]]
unreadable = [r["file"] for r in rows if r.get("mode") and int(r["mode"], 8) & 0o004 == 0]
res = {"repo": man["repo"], "revision": man["revision"], "snapshot": snap,
       "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.time() - t0, 1),
       "n_files": len(rows), "n_expected": len(man["files"]),
       "bytes": sum(r.get("size", 0) for r in rows), "bytes_expected": man["total_bytes"],
       "pass": not bad and len(rows) == len(man["files"]), "failures": bad,
       "not_world_readable": unreadable, "files": rows}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "files"}, indent=1))
