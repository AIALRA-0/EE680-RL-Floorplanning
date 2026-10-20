"""Record an isolated, fully inspectable PPO update from a saved checkpoint."""

import argparse
from pathlib import Path
import torch

from src.data.parse_blknet import load_circuit
from src.experiments.evaluate import encoder_detail, policy_detail, write_json
from src.experiments.train import export_rollout
from src.ppo.algorithm import (collect_episodes, load_checkpoint, rollout_tensors,
                               save_checkpoint, update_policy)
from src.ppo.encoding import collate


def observed_decision(model, record):
    with torch.no_grad():
        distribution, value = model.distribution(collate([record["encoded"]]))
        detail = policy_detail(record["encoded"], distribution, value, record["action"])
        detail["encoder"] = encoder_detail(model, record["encoded"])
    return detail


def record_teaching(args):
    torch.set_num_threads(1)
    model, optimizer, config, saved = load_checkpoint(args.checkpoint)
    torch.manual_seed(args.seed)
    meta = saved["metadata"]
    circuits = [load_circuit(path) for path in args.input]
    records, episodes = collect_episodes(model, circuits, args.episodes, meta["width"], meta["height"], meta["area_weight"])
    targets = rollout_tensors(records, config, include_details=True)
    before = observed_decision(model, records[0])
    stats = update_policy(model, optimizer, records, config)
    after = observed_decision(model, records[0])
    payload = {"source_update": meta["update"], "seed": args.seed, "config": vars(config),
               "board": {"width": meta["width"], "height": meta["height"]}, "circuits": circuits,
               "episodes": episodes, "transitions": export_rollout(records, targets, config),
               "advantage_mean": float(targets["advantage_mean"]), "advantage_std": float(targets["advantage_std"]),
               "first_decision_before": before, "first_decision_after": after, "update_stats": stats}
    write_json(args.output, payload)
    save_checkpoint(args.output.with_suffix(".pt"), model, optimizer, config,
                    {**meta, "teaching_source_update": meta["update"], "isolated_teaching_update": True})
    print(f"Recorded {len(records)} real transitions and one isolated PPO update from checkpoint {meta['update']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("Episodes must be positive")
    record_teaching(args)


if __name__ == "__main__":
    main()
