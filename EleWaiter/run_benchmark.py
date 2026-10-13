#!/usr/bin/env python3
"""Benchmark all dispatchers over many random seeds and print a results table.

Usage:
  python run_benchmark.py                       # hospital, all scenarios, 10 test seeds
  python run_benchmark.py --building "12-floor Office" --scenarios up_peak down_peak mixed
  python run_benchmark.py --intensity 1.5 --seeds 20 --out results.csv
"""
import argparse

import pandas as pd

from elewaiter.benchmark import CORE_ALGORITHMS, ALGORITHMS, run_benchmark, summarize, win_rate
from elewaiter.buildings import PRESETS
from elewaiter.traffic import HOSPITAL_SCENARIOS

ap = argparse.ArgumentParser()
ap.add_argument("--building", default="10-floor Hospital", choices=list(PRESETS))
ap.add_argument("--scenarios", nargs="*", default=None, help="default: hospital scenarios (or all general ones for non-hospitals)")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--first-seed", type=int, default=2000, help="test seeds start here (training used 1000-1002)")
ap.add_argument("--duration", type=float, default=600.0)
ap.add_argument("--intensity", type=float, default=1.5)
ap.add_argument("--ablation", action="store_true", help="also run 'AI (no emergency logic)'")
ap.add_argument("--out", default=None)
a = ap.parse_args()

cfg = PRESETS[a.building]()
scen = a.scenarios or (HOSPITAL_SCENARIOS if cfg.kind == "hospital" else ["up_peak", "down_peak", "mixed"])
algos = ALGORITHMS if a.ablation else CORE_ALGORITHMS
seeds = range(a.first_seed, a.first_seed + a.seeds)

df = run_benchmark(cfg, scen, algos, seeds, a.duration, a.intensity)
pd.set_option("display.width", 250, "display.max_columns", 30)
cols = ["scenario", "algorithm", "avg_wait", "p95_wait", "max_wait", "wait_std", "starvation_count",
        "left_behind", "em_avg_wait", "em_p95_wait", "em_within_target", "routine_avg_wait"]
s = summarize(df)
print(f"\n{a.building} | intensity {a.intensity} | {a.seeds} seeds | mean per scenario\n")
print(s[[c for c in cols if c in s.columns]].round(2).to_string(index=False))

for metric in ("em_avg_wait", "avg_wait"):
    if metric == "em_avg_wait" and cfg.kind != "hospital":
        continue
    print(f"\nSeeds where EleWaiter AI has LOWER {metric} than each rival:")
    for rival in [x for x in algos if x != "EleWaiter AI"]:
        w = win_rate(df, "EleWaiter AI", rival, metric)
        print(f"  vs {rival:24s}", "  ".join(f"{r.scenario}:{r.wins}/{r.seeds}" for r in w.itertuples()))
if a.out:
    df.to_csv(a.out, index=False)
    print("\nraw results ->", a.out)
