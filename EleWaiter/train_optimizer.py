#!/usr/bin/env python3
"""Tune the AI dispatcher's weights with a GA and with simulated annealing.

Protocol
  train      seeds 1000-1002  (used by the optimizers)
  validation seeds 3000-3003  (picks the winner among default / GA / SA)
  test       seeds 2000-2009  (never used for tuning; run run_benchmark.py)

Usage:  python train_optimizer.py [--pop 14] [--gens 10] [--sa-iters 120]
Writes building-specific hospital-v2 weights and training logs under elewaiter/data/.
"""
import argparse
import json
import os
import time

from elewaiter.buildings import PRESETS
from elewaiter.dispatchers.ai import DEFAULT_WEIGHTS
from elewaiter.optimizer import fitness, genetic_algorithm, simulated_annealing

VAL_KW = dict(scenarios=("emergency_surge", "mass_casualty", "visiting_hours", "lift_failure"),
              seeds=(3000, 3001, 3002, 3003), intensities=(1.0, 1.8), duration=300.0,
              alert_coverage=1.0, transfer_space=3)

ap = argparse.ArgumentParser()
ap.add_argument("--pop", type=int, default=14)
ap.add_argument("--gens", type=int, default=10)
ap.add_argument("--sa-iters", type=int, default=120)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--building", choices=list(PRESETS), default="10-floor Hospital")
args = ap.parse_args()

cfg = PRESETS[args.building]()
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elewaiter", "data")
os.makedirs(here, exist_ok=True)
t0 = time.time()

print("GA ...", flush=True)
ga_w, ga_hist = genetic_algorithm(cfg, args.pop, args.gens, args.seed,
                                  progress=lambda g, f: print(f"  gen {g:2d} best {f:7.2f}  ({time.time()-t0:5.0f}s)", flush=True))
print("SA ...", flush=True)
sa_w, sa_hist = simulated_annealing(cfg, args.sa_iters, seed=args.seed,
                                    progress=lambda i, f: print(f"  it {i:3d} best {f:7.2f}  ({time.time()-t0:5.0f}s)", flush=True) if i % 20 == 0 else None)

cands = {"default": dict(DEFAULT_WEIGHTS), "GA": ga_w, "SA": sa_w}
val = {k: fitness(w, cfg, **VAL_KW) for k, w in cands.items()}
winner = min(val, key=val.get)
print("validation fitness:", {k: round(v, 2) for k, v in val.items()}, "->", winner, flush=True)

suffix = "10" if args.building == "10-floor Hospital" else "15"
with open(os.path.join(here, f"tuned_weights_hospital_v2_{suffix}.json"), "w") as fh:
    json.dump({k: round(v, 4) for k, v in cands[winner].items()}, fh, indent=2)
with open(os.path.join(here, f"training_log_hospital_v2_{suffix}.json"), "w") as fh:
    json.dump({"ga_history": ga_hist, "sa_history": sa_hist, "validation": val, "winner": winner,
               "ga_weights": ga_w, "sa_weights": sa_w, "default_weights": DEFAULT_WEIGHTS,
               "seconds": round(time.time() - t0)}, fh, indent=2)
print("done in", round(time.time() - t0), "s")
