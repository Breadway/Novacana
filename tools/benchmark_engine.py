#!/usr/bin/env python3
"""Compare seeded release binaries from engine/examples/training_bench.rs."""
import argparse
import json
import resource
import statistics
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("before", type=Path)
p.add_argument("after", type=Path)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--rounds", type=int, default=4)
args = p.parse_args()
records = []
for mode, count in [("expert", 12), ("advanced", 200), ("view", 500000), ("legal", 500000), ("clone", 500000)]:
    for repeat in range(args.rounds):
        seed = 1000 + repeat * 100
        pair = {}
        # Alternate execution order to reduce bias from thermal/load drift.
        order = ["before", "after"] if repeat % 2 == 0 else ["after", "before"]
        for version in order:
            start = resource.getrusage(resource.RUSAGE_CHILDREN)
            proc = subprocess.run([str(getattr(args, version).resolve()), mode, str(count), str(seed)],
                                  text=True, capture_output=True, check=True)
            end = resource.getrusage(resource.RUSAGE_CHILDREN)
            row = json.loads(proc.stdout)
            row.update(version=version, repeat=repeat,
                       cpu_seconds=end.ru_utime + end.ru_stime - start.ru_utime - start.ru_stime)
            pair[version] = row
            records.append(row)
        assert pair["before"]["checksum"] == pair["after"]["checksum"], pair
        assert pair["before"]["turns"] == pair["after"]["turns"], pair
        args.out.write_text(json.dumps(records, indent=2) + "\n")
        print(f'{mode} {repeat+1}/{args.rounds}: before={pair["before"]["seconds"]:.3f}s '
              f'after={pair["after"]["seconds"]:.3f}s', flush=True)
    rows = [r for r in records if r["mode"] == mode]
    timings = {v: statistics.median(r["seconds"] for r in rows if r["version"] == v)
               for v in ["before", "after"]}
    print(f'{mode}: median throughput speedup {timings["before"]/timings["after"]:.3f}x', flush=True)

for version in ["before", "after"]:
    row = json.loads(subprocess.check_output([str(getattr(args, version).resolve()), "expert", "12", "1000", "alloc"], text=True))
    records.append(dict(row, version=version, allocation_run=True))
args.out.write_text(json.dumps(records, indent=2) + "\n")
