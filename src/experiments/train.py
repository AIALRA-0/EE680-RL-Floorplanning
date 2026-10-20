"""Train masked PPO on supplied circuits and save reproducible local evidence."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import torch

from src.data.parse_blknet import load_circuit
from src.experiments.evaluate import evaluate_many, run_episode, write_json
from src.ppo.algorithm import (PPOConfig, collect_episodes, load_checkpoint, rollout_tensors,
                               save_checkpoint, update_policy)
from src.ppo.model import ActorCritic


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--area-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path, help="Load model weights into a new training experiment")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=32)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--no-preview", action="store_true", help="Keep the learning page on its current run")
    args = parser.parse_args()
    if min(args.updates, args.episodes_per_update, args.eval_every, args.eval_episodes, args.threads) < 1:
        parser.error("Counts must be positive")
    if args.resume and args.initialize_from:
        parser.error("Choose --resume or --initialize-from")
    if min(args.epochs, args.minibatch) < 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("Epochs/minibatch must be positive; GAE lambda must be in [0, 1]")
    return args


def prepare(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = PPOConfig(learning_rate=args.learning_rate, entropy_coefficient=args.entropy,
                       epochs=args.epochs, minibatch=args.minibatch, gae_lambda=args.gae_lambda)
    model = ActorCritic()
    if args.initialize_from:
        model, _, _, _ = load_checkpoint(args.initialize_from)
        torch.manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    offset, history, previous_seconds = 0, [], 0.0
    if args.resume:
        model, optimizer, config, saved = load_checkpoint(args.resume)
        for key in ("width", "height", "area_weight"):
            if saved["metadata"][key] != getattr(args, key):
                raise ValueError(f"Resume must preserve {key}")
        torch.set_rng_state(saved["torch_rng"])
        offset = saved["metadata"]["update"]
        history = saved["metadata"]["history"]
        previous_seconds = saved["metadata"]["training_seconds"]
    return model, optimizer, config, offset, history, previous_seconds


def evaluate_inputs(args, model, circuits, update):
    rng = torch.get_rng_state()
    torch.manual_seed(args.seed + 100000)
    summaries = []
    for index, (path, circuit) in enumerate(zip(args.input, circuits)):
        summary = evaluate_many(circuit, model, args.eval_episodes, args.width, args.height, args.area_weight)
        greedy = run_episode(circuit, model, args.width, args.height, args.area_weight)
        greedy["update"] = update
        write_json(args.output / f"samples-{index}-update-{update}.json", summary.pop("samples"))
        summary.update({"input": path.name, "index": index, "update": update, "greedy": {k: greedy[k] for k in
                        ("metrics", "reward", "reason", "inference_seconds", "validation")}})
        write_json(args.output / f"replay-{index}.json", greedy)
        write_json(args.output / f"replay-{index}-update-{update}.json", greedy)
        summaries.append(summary)
    torch.set_rng_state(rng)
    write_json(args.output / "evaluation.json", {"update": update, "circuits": summaries})
    return summaries


def export_rollout(records, targets, config):
    rows = []
    for i, record in enumerate(records):
        next_value = records[i+1]["value"] if i+1 < len(records) else 0.0
        encoded = record["encoded"]
        current = int(encoded["nodes"][:, 6].argmax())
        rows.append({**{key: value for key, value in record.items() if key != "encoded"},
                     "sample_index": i, "next_value": 0.0 if record["terminated"] else next_value,
                     "delta": record["reward"] + config.gamma * next_value * (not record["terminated"]) - record["value"],
                     "advantage_raw": float(targets["advantages_raw"][i]),
                     "legal_count": int(encoded["mask"].sum()),
                     "current_node": encoded["nodes"][current].tolist(),
                     "nodes": encoded["nodes"].tolist(),
                     "selected_features": encoded["candidates"][record["action"]].tolist(),
                     "advantage_normalized": float(targets["advantages"][i]),
                     "return": float(targets["returns"][i])})
    return rows


def training_step(args, model, optimizer, config, circuits, update):
    records, episodes = collect_episodes(model, circuits, args.episodes_per_update, args.width,
                                         args.height, args.area_weight,
                                         start=(update - 1) * args.episodes_per_update)
    stats = update_policy(model, optimizer, records, config)
    targets = rollout_tensors(records, config, include_details=True)
    exported = export_rollout(records, targets, config)
    payload = {"update": update, "transitions": exported, "circuits": circuits,
               "advantage_mean": float(targets["advantage_mean"]), "advantage_std": float(targets["advantage_std"])}
    write_json(args.output / "rollout.json", payload)
    if update % args.eval_every == 0:
        write_json(args.output / f"rollout-update-{update}.json", payload)
    stats.update({"update": update, "transitions": len(records),
                  "mean_reward": sum(e["reward"] for e in episodes) / len(episodes),
                  "legal_completion_rate": sum(e["complete"] for e in episodes) / len(episodes)})
    return stats


def save_progress(args, model, optimizer, config, history, evaluations, started, status):
    metadata = {"width": args.width, "height": args.height, "area_weight": args.area_weight,
                "update": history[-1]["update"] if history else 0, "history": history,
                "seed": args.seed, "inputs": [str(p) for p in args.input],
                "training_seconds": time.perf_counter() - started,
                "episodes_per_update": args.episodes_per_update, "eval_episodes": args.eval_episodes,
                "initialize_from": str(args.initialize_from) if args.initialize_from else None,
                "evaluation_update": evaluations[0]["update"] if evaluations else None}
    save_checkpoint(args.output / "checkpoint.pt", model, optimizer, config, metadata)
    if metadata["update"] == 0 or metadata["update"] % args.eval_every == 0 or status == "completed":
        save_checkpoint(args.output / f"checkpoint-{metadata['update']}.pt", model, optimizer, config, metadata)
    progress = {**metadata, "config": asdict(config), "status": status, "evaluations": evaluations,
                "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "device": "cpu", "threads": torch.get_num_threads()}}
    write_json(args.output / "progress.json", progress)
    root = Path(__file__).resolve().parents[2]
    if not args.no_preview and args.output.resolve().is_relative_to(root):
        write_json(root / "steps/build/ppo-latest.json", {"directory": args.output.resolve().relative_to(root).as_posix()})


def main():
    args = arguments()
    circuits = [load_circuit(path) for path in args.input]
    args.output.mkdir(parents=True, exist_ok=True)
    model, optimizer, config, offset, history, previous_seconds = prepare(args)
    started = time.perf_counter() - previous_seconds
    evaluations = evaluate_inputs(args, model, circuits, offset)
    if not args.resume:
        write_json(args.output / "initial-evaluation.json", {"circuits": evaluations})
    save_progress(args, model, optimizer, config, history, evaluations, started, "running")
    for update in range(offset + 1, offset + args.updates + 1):
        model.train()
        row = training_step(args, model, optimizer, config, circuits, update)
        history.append(row)
        if update % args.eval_every == 0 or update == offset + args.updates:
            model.eval()
            evaluations = evaluate_inputs(args, model, circuits, update)
            print(json.dumps({**row, "evaluation": evaluations}), flush=True)
        save_progress(args, model, optimizer, config, history, evaluations, started, "running")
    save_progress(args, model, optimizer, config, history, evaluations, started, "completed")


if __name__ == "__main__":
    main()
