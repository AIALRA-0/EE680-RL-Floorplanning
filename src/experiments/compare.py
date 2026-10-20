"""Compare frozen PPO checkpoints and adapted SA-SP under one declared protocol."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time
import torch

from cpp.sa_sp.adapter import run_original
from cpp.sa_sp.aligned import run_aligned
from integration.evaluation.protocol import comparison_protocol, score_layout
from src.data.parse_blknet import load_circuit
from src.experiments.evaluate import run_episode, write_json
from src.ppo.algorithm import load_checkpoint
from src.ppo.encoding import collate, encode_observation
from src.environment.placement_env import PlacementEnv

ROOT = Path(__file__).resolve().parents[2]


def case_plan(count):
    base = ROOT / "steps/build"
    if not (base / "ppo-5-10-seed0/progress.json").exists():
        base = ROOT / "steps/evidence"
    folder = base / "ppo-5-10-seed0" if count <= 10 else base / "ppo-scale-20261019" / f"case-{[20,50,100].index(count)}-macros-{count}"
    progress = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
    return {"macros": count, "input": f"benchmarks/alarik-synthetic/blknet_{count}.txt",
            "checkpoint": (folder / "checkpoint.pt").relative_to(ROOT).as_posix(),
            "checkpoint_update": progress["update"], "training_seed": progress["seed"],
            "training_seconds": progress["training_seconds"],
            "shared_pretraining_seconds": json.loads((base / "ppo-5-10-seed0/progress.json").read_text(encoding="utf-8"))["training_seconds"] if count > 10 else None,
            "protocol": comparison_protocol(progress["width"], progress["height"], progress["area_weight"])}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("steps/build/comparison-20261020"))
    parser.add_argument("--cases", nargs="+", type=int, choices=[5,10,20,50,100], default=[5,10,20,50,100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--stages", type=int, default=180)
    args = parser.parse_args()
    if args.stages < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Stages must be positive and seeds must be distinct")
    if (args.output / "plan.json").exists():
        parser.error("Choose a new output directory to preserve the existing experiment")
    return args


def warm_policy(model, circuit, protocol):
    env = PlacementEnv(circuit, protocol["width"], protocol["height"], protocol["area_weight"])
    with torch.no_grad():
        model.distribution(collate([encode_observation(env.observation(), env.legal_mask())]))


def ppo_trial(model, circuit, protocol, seed, deterministic):
    torch.manual_seed(seed)
    started, cpu = time.perf_counter(), time.process_time()
    replay = run_episode(circuit, model, protocol["width"], protocol["height"], protocol["area_weight"],
                         deterministic=deterministic, trace=True, policy_trace=False)
    validation = score_layout(circuit, replay["placements"], protocol)
    return {"algorithm": "ppo_argmax" if deterministic else "ppo_sample", "seed": seed,
            "placements": replay["placements"], "validation": validation, "replay": replay,
            "timing": {"wall_seconds": time.perf_counter()-started,
                       "process_cpu_seconds": time.process_time()-cpu,
                       "includes": "environment, policy inference, lightweight step trace and independent validation"}}


def trial_summary(result, relative):
    return {key: result[key] for key in ("algorithm", "seed", "validation", "timing")} | {"file": relative}


def aggregate(trials):
    result = []
    for algorithm in dict.fromkeys(t["algorithm"] for t in trials):
        selected = [t for t in trials if t["algorithm"] == algorithm]
        legal = [t for t in selected if t["validation"]["legal"]]
        row = {"algorithm": algorithm, "episodes": len(selected), "legal": len(legal),
               "mean_reward_all": statistics.mean(t["validation"]["reward"] for t in selected)}
        for metric in ("area", "hpwl", "normalized_cost"):
            values = [t["validation"][metric] for t in legal]
            row[f"mean_{metric}_complete"] = statistics.mean(values) if values else None
            row[f"std_{metric}_complete"] = statistics.stdev(values) if len(values) > 1 else None
        for metric in ("wall_seconds", "process_cpu_seconds"):
            row[f"mean_{metric}"] = statistics.mean(t["timing"][metric] for t in selected)
        result.append(row)
    return result


def save_trial(directory, result, circuit, protocol):
    name = f"{result['algorithm']}-seed-{result['seed']}.json"
    result.update({"circuit": circuit, "protocol": protocol})
    path = directory / name
    write_json(path, result)
    return trial_summary(result, path.resolve().relative_to(ROOT).as_posix())


def run_case(args, plan, status):
    circuit = load_circuit(ROOT / plan["input"])
    directory = args.output / f"macros-{plan['macros']}"
    loaded = time.perf_counter()
    model, _, _, _ = load_checkpoint(ROOT / plan["checkpoint"])
    model.eval()
    load_seconds = time.perf_counter()-loaded
    warm_policy(model, circuit, plan["protocol"])
    trials = [save_trial(directory, ppo_trial(model, circuit, plan["protocol"], 0, True), circuit, plan["protocol"])]
    for seed in args.seeds:
        trials.append(save_trial(directory, ppo_trial(model, circuit, plan["protocol"], seed, False), circuit, plan["protocol"]))
    for seed in args.seeds:
        def update(stage):
            if stage["stage"] % 30 == 0:
                status.update({"current": {"macros": plan["macros"], "algorithm": "sa_sp_aligned", "seed": seed, **stage}})
                write_json(args.output / "summary.json", status)
                print(json.dumps(status["current"]), flush=True)
        result = run_aligned(circuit, plan["protocol"], seed, args.stages, on_stage=update)
        trials.append(save_trial(directory, result, circuit, plan["protocol"]))
    return {**plan, "model_load_seconds": load_seconds, "trials": trials, "aggregate": aggregate(trials)}


def main():
    args = arguments()
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Keep comparison evidence inside the ignored repository build directory")
    torch.set_num_threads(1)
    plans = [case_plan(n) for n in args.cases]
    status = {"status": "running", "cases": [], "plan": {"cases": plans, "seeds": args.seeds,
              "sa_stages": args.stages, "sa_moves_per_temperature": "max(20*N, 200)",
              "ppo": f"frozen checkpoints; one argmax and {len(args.seeds)} independently seeded sampled inference runs",
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "device": "cpu", "torch_threads": 1}, "created_utc": datetime.now(timezone.utc).isoformat()}}
    write_json(args.output / "plan.json", status["plan"])
    write_json(args.output / "summary.json", status)
    write_json(ROOT / "steps/build/comparison-latest.json", {"directory": args.output.resolve().relative_to(ROOT).as_posix()})
    original = run_original(load_circuit(ROOT / "benchmarks/alarik-synthetic/blknet_5.txt"))
    write_json(args.output / "original-sa-5-seed-0.json", original)
    status["original_sa_smoke"] = {"area": original["final"]["validation"]["area"],
                                   "hpwl": original["final"]["validation"]["hpwl"], "moves": original["moves"]}
    for plan in plans:
        status["cases"].append(run_case(args, plan, status))
        write_json(args.output / "summary.json", status)
    status.update({"status": "completed", "current": None})
    write_json(args.output / "summary.json", status)
    print(json.dumps({"status": "completed", "cases": len(status["cases"])}), flush=True)


if __name__ == "__main__":
    main()
