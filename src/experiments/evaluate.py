"""Evaluate checkpoints and simple controls through the same environment."""

import argparse
import base64
import json
from pathlib import Path
import random
import statistics
import time
import struct
import zlib
import torch

from integration.evaluation.placement import evaluate_placement
from src.data.parse_blknet import load_circuit
from src.environment.placement_env import PlacementEnv
from src.ppo.algorithm import load_checkpoint
from src.ppo.encoding import collate, encode_observation


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def run_episode(circuit, model=None, width=64, height=64, area_weight=0.5,
                deterministic=True, seed=0, baseline=None, trace=True, policy_trace=True):
    generator = random.Random(seed)
    env = PlacementEnv(circuit, width, height, area_weight)
    steps, started = [], time.perf_counter()
    while not env.terminated:
        mask, observation = env.legal_mask(), env.observation()
        detail = {}
        if model is None:
            legal = [i for i, valid in enumerate(mask) if valid]
            action = generator.choice(legal) if baseline == "random" else legal[0]
        else:
            encoded = encode_observation(observation, mask)
            distribution, value = model.distribution(collate([encoded]))
            action = int(distribution.probs.argmax(-1) if deterministic else distribution.sample())
            if trace and policy_trace:
                detail = policy_detail(encoded, distribution, value, action)
                detail["encoder"] = encoder_detail(model, encoded)
        _, reward, done, truncated, info = env.step((action % width, action // width))
        if trace:
            steps.append({"observation": observation, "action_index": action,
                          "action": [action % width, action // width], "reward": reward,
                          "terminated": done, "truncated": truncated, "info": info, **detail})
    runtime = time.perf_counter() - started
    return episode_result(circuit, env, steps, runtime)


def policy_detail(encoded, distribution, value, action):
    probabilities = distribution.probs[0]
    top = probabilities.topk(min(10, int(encoded["mask"].sum())))
    return {"value": float(value), "log_prob": float(distribution.log_prob(torch.tensor([action]))),
            "entropy": float(distribution.entropy()), "legal_count": int(encoded["mask"].sum()),
            "nodes": encoded["nodes"].tolist(), "incidence": encoded["incidence"].tolist(),
            "selected_features": encoded["candidates"][action].tolist(),
            **grid_payload(probabilities.tolist(), encoded["mask"].tolist()),
            "top_actions": [{"index": int(i), "probability": float(p)}
                            for i, p in zip(top.indices, top.values)]}


def grid_payload(probabilities, mask):
    if len(mask) <= 16384:
        return {"probabilities": probabilities, "legal_mask": mask}
    runs, start = [], None
    for index, valid in enumerate(mask + [False]):
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            runs.append([start, index])
            start = None
    packed = struct.pack(f"<{len(probabilities)}f", *probabilities)
    return {"probability_grid": {"encoding": "zlib-float32-le", "length": len(mask),
                                 "data": base64.b64encode(zlib.compress(packed, 1)).decode("ascii")},
            "legal_mask_runs": runs}


@torch.no_grad()
def encoder_detail(model, encoded):
    context, intermediate = model.encode_context(collate([encoded]))
    return {"node_shape": list(intermediate["node_embeddings"].shape[1:]),
            "net_shape": list(intermediate["net_embeddings"].shape[1:]),
            "pooled_first8": intermediate["pooled"][0, :8].tolist(),
            "current_first8": intermediate["current"][0, :8].tolist(),
            "context_first8": context[0, :8].tolist()}


def episode_result(circuit, env, steps, runtime):
    checked = evaluate_placement(circuit, env.placements, env.width, env.height)
    if checked["complete"] and (not checked["legal"] or any(
            not abs(checked[key] - env.metrics()[key]) < 1e-6 for key in ("area", "hpwl"))):
        raise AssertionError("Independent evaluator disagrees with the environment")
    return {"circuit": circuit, "board": {"width": env.width, "height": env.height},
            "area_weight": env.area_weight, "placements": env.placements,
            "metrics": env.metrics(), "reward": env.reward, "reason": env.reason,
            "validation": checked, "inference_seconds": runtime, "steps": steps}


def evaluate_many(circuit, model, episodes, width, height, area_weight, seed=10000):
    results = [run_episode(circuit, model, width, height, area_weight,
                           deterministic=False, seed=seed + i, trace=False) for i in range(episodes)]
    legal = [r for r in results if r["validation"]["legal"]]
    return {"episodes": episodes, "legal_completion_rate": len(legal) / episodes,
            "mean_reward": statistics.mean(r["reward"] for r in results),
            "mean_area_complete": statistics.mean(r["metrics"]["area"] for r in legal) if legal else None,
            "mean_hpwl_complete": statistics.mean(r["metrics"]["hpwl"] for r in legal) if legal else None,
            "samples": [{"episode": i, **{key: r[key] for key in
                         ("placements", "metrics", "reward", "reason", "validation", "inference_seconds")}}
                        for i, r in enumerate(results)]}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=("first_legal", "random"))
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--width", type=int, help="Override the saved canvas width for a new input")
    parser.add_argument("--height", type=int, help="Override the saved canvas height for a new input")
    parser.add_argument("--summary-only", action="store_true", help="Skip large per-action traces")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.baseline):
        parser.error("Supply exactly one of --checkpoint and --baseline")
    return args


def main():
    args = arguments()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model, width, height, weight = None, 64, 64, 0.5
    if args.checkpoint:
        model, _, _, saved = load_checkpoint(args.checkpoint)
        width, height, weight = (saved["metadata"][k] for k in ("width", "height", "area_weight"))
        model.eval()
    width = width if args.width is None else args.width
    height = height if args.height is None else args.height
    result = run_episode(load_circuit(args.input), model, width, height, weight,
                         not args.sample, args.seed, args.baseline, trace=not args.summary_only)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("metrics", "reason", "inference_seconds", "validation")}))
    if not result["validation"]["legal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
