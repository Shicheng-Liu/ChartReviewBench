"""Regenerate rebuttal/B_humaneval200.json from work/runtime/B.json.

42 single-, 95 double- and 63 triple-flaw tasks (proportional to 400/900/600), spread as
evenly as possible over the flaw scenarios of each arity, with random.Random(42).
Run from the visaudit folder after materializing work/runtime.
"""
import collections
import json
import random
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "work/runtime/B.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "rebuttal/B_humaneval200.json"
d = json.load(open(src))
rng = random.Random(42)
by = collections.defaultdict(list)
for x in d["items"]:
    by[(x["arity"], x["category"])].append(x)
pick = []
for arity, n in {"single": 42, "pair": 95, "triple": 63}.items():
    keys = sorted(k for k in by if k[0] == arity)
    base, extra = divmod(n, len(keys))
    bonus = set(rng.sample(keys, extra))
    for k in keys:
        pick += rng.sample(sorted(by[k], key=lambda x: x["instance"]), base + (k in bonus))
assert len(pick) == 200 and len({x["instance"] for x in pick}) == 200
out = dict(d)
out["items"], out["total"], out["name"] = pick, 200, d["name"] + "-humaneval200-seed42"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst, collections.Counter(x["arity"] for x in pick))
