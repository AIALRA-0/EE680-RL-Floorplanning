"""Run a declared seed/weight grid with equal update budgets and retain failures."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from src.experiments.evaluate import write_json


def run_one(args, seed, weight):
    directory = args.output / f"seed-{seed}-weight-{weight:g}"
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "src.experiments.train", "--input", *map(str, args.input),
               "--output", str(directory), "--seed", str(seed), "--area-weight", str(weight),
               "--updates", str(args.updates), "--episodes-per-update", str(args.episodes_per_update),
               "--eval-every", str(args.updates), "--eval-episodes", str(args.eval_episodes), "--no-preview"]
    with (directory / "execution.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    row = {"seed": seed, "area_weight": weight, "command": command[1:],
           "returncode": result.returncode, "directory": str(directory)}
    if result.returncode == 0:
        progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
        initial = json.loads((directory / "initial-evaluation.json").read_text(encoding="utf-8"))
        row.update({"training_seconds": progress["training_seconds"],
                    "initial": initial["circuits"], "final": progress["evaluations"]})
    print(json.dumps({"seed": seed, "area_weight": weight, "returncode": result.returncode}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weights", type=float, nargs="+", default=[0, 0.5, 1])
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    grid = [(seed, weight) for seed in args.seeds for weight in args.weights]
    write_json(args.output / "plan.json", {"seeds": args.seeds, "weights": args.weights,
                                          "updates": args.updates, "episodes_per_update": args.episodes_per_update,
                                          "eval_episodes": args.eval_episodes, "inputs": list(map(str, args.input))})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_one, args, seed, weight) for seed, weight in grid]
        results = [future.result() for future in pending]
    write_json(args.output / "summary.json", {"runs": results})
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
