#!/usr/bin/env python3
"""
Dump the actual similarity/scoring code from the INSTALLED perch_hoplite (v1.0.2),
so we know exactly how nearest-neighbour scores are computed in the version we run.

Run on spark:  python3 tools/read_hoplite_scoring.py
"""
import os
import perch_hoplite.db.sqlite_usearch_impl as m

dbdir = os.path.dirname(m.__file__)
print("perch_hoplite.db dir:", dbdir)
print("py files:", sorted(f for f in os.listdir(dbdir) if f.endswith(".py")))

for name in ("score_functions.py", "brutalism.py", "search_results.py"):
    p = os.path.join(dbdir, name)
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)
    if os.path.exists(p):
        txt = open(p).read()
        # keep it readable — print, but note length
        print(f"[{len(txt)} chars]\n")
        print(txt)
    else:
        print("(not present)")
