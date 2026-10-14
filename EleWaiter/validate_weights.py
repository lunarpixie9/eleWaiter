#!/usr/bin/env python3
"""Compare default and saved AI weights on separate hospital layouts and seeds.

This is read-only: it never overwrites tuned weights. Use training seeds only in
train_optimizer.py; this script defaults to held-out seeds 4000 onward.
"""
import argparse
import json
import os
from statistics import mean

from elewaiter.benchmark import run_once
from elewaiter.buildings import PRESETS
from elewaiter.dispatchers import DEFAULT_WEIGHTS, load_tuned_weights
from elewaiter.optimizer import composite
from elewaiter.traffic import HOSPITAL_SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--intensity", type=float, default=1.0)
    ap.add_argument("--alerts", type=float, default=1.0)
    ap.add_argument("--transfer-space", type=int, default=3)
    args = ap.parse_args()
    if args.seeds < 1 or not 0 <= args.alerts <= 1:
        ap.error("--seeds must be positive and --alerts must be between 0 and 1")

    for name, build in PRESETS.items():
        cfg = build()
        candidates = {"default": dict(DEFAULT_WEIGHTS)}
        saved = load_tuned_weights(name)
        if saved:
            candidates["hospital-v2"] = saved
        if name == "10-floor Hospital":
            legacy_path = os.path.join(os.path.dirname(__file__), "elewaiter", "data", "tuned_weights.json")
            if os.path.exists(legacy_path):
                with open(legacy_path, encoding="utf-8") as fh:
                    candidates["legacy weights"] = json.load(fh)
        print(f"\n{name} | {args.seeds} held-out seeds | alerts {args.alerts:.0%} | "
              f"transfer space {args.transfer_space}")
        for label, weights in candidates.items():
            results = [run_once(cfg, scenario, "EleWaiter AI", seed, args.duration,
                                args.intensity, weights, alert_coverage=args.alerts,
                                transfer_space=args.transfer_space)
                       for scenario in HOSPITAL_SCENARIOS
                       for seed in range(4000, 4000 + args.seeds)]
            print(f"  {label:14s} score {mean(map(composite, results)):6.1f} | "
                  f"emergency full trip {mean(r.em_avg_journey for r in results if r.em_avg_journey is not None):5.1f}s | "
                  f"routine wait {mean(r.routine_avg_wait for r in results if r.routine_avg_wait is not None):5.1f}s | "
                  f"unserved {sum(r.n_unserved for r in results)}")
        if not saved:
            print("  No hospital-v2 tuned weights yet. Run train_optimizer.py --building "
                  f'"{name}" to tune this layout.')


if __name__ == "__main__":
    main()
