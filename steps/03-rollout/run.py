"""Collect actual masked actions from a checkpoint without updating its weights."""

import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.parse_blknet import load_circuit
from src.experiments.evaluate import write_json
from src.ppo.algorithm import collect_episodes, load_checkpoint, rollout_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("Episodes must be positive")
    torch.set_num_threads(1)
    model, _, config, saved = load_checkpoint(args.checkpoint)
    torch.manual_seed(args.seed)
    metadata = saved["metadata"]
    records, summaries = collect_episodes(model, [load_circuit(p) for p in args.input], args.episodes,
                                          metadata["width"], metadata["height"], metadata["area_weight"])
    targets = rollout_tensors(records, config)
    samples = [{**{k: v for k, v in r.items() if k != "encoded"},
                "encoding": {k: v.tolist() for k, v in r["encoded"].items()},
                "return": float(targets["returns"][i]),
                "advantage_normalized": float(targets["advantages"][i])} for i, r in enumerate(records)]
    write_json(args.output, {"seed": args.seed, "episodes": summaries, "transitions": samples})
    print(f"Saved {len(samples)} real transitions from {len(summaries)} complete episodes")


if __name__ == "__main__":
    main()
