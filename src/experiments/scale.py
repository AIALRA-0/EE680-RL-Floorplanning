"""Fine-tune the shared model on larger inputs using declared per-circuit canvases."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys

from src.data.parse_blknet import load_circuit
from src.experiments.evaluate import write_json

ROOT = Path(__file__).resolve().parents[2]


def case_plan(args, path, index):
    circuit = load_circuit(path)
    n = len(circuit["blocks"])
    total = sum(w*h for w,h in circuit["blocks"].values())
    side = math.ceil(max(math.sqrt(total / 0.5), max(max(d) for d in circuit["blocks"].values())) / 16) * 16
    directory = args.output / f"case-{index}-macros-{n}"
    command = [sys.executable, "-m", "src.experiments.train", "--input", str(path), "--output", str(directory),
               "--initialize-from", str(args.initialize_from), "--width", str(side), "--height", str(side),
               "--updates", str(args.updates), "--episodes-per-update", str(2 if n > 50 else 4),
               "--epochs", str(3 if n <= 20 else 2), "--minibatch", str(16 if n <= 20 else 8),
               "--gae-lambda", "1", "--learning-rate", "0.0003", "--eval-every", "25",
               "--eval-episodes", "8", "--seed", str(args.seed), "--no-preview"]
    return {"input": str(path), "macros": n, "macro_area": total, "width": side, "height": side,
            "utilization": total / side**2, "directory": str(directory), "command": command}


def run_case(plan):
    directory = Path(plan["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "execution.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(plan["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    row = {**plan, "returncode": result.returncode}
    if result.returncode == 0:
        row["initial"] = json.loads((directory / "initial-evaluation.json").read_text(encoding="utf-8"))["circuits"]
        progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
        row.update({"final": progress["evaluations"], "training_seconds": progress["training_seconds"]})
    print(json.dumps({"macros": plan["macros"], "returncode": result.returncode}), flush=True)
    return row


def register_runs(plans):
    path = ROOT / "steps/build/ppo-runs.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    pointer = ROOT / "steps/build/ppo-latest.json"
    if pointer.exists():
        existing["runs"].append({**json.loads(pointer.read_text(encoding="utf-8")), "label": "Original shared 5/10-macro training"})
    for plan in plans:
        directory = Path(plan["directory"]).resolve()
        if directory.is_relative_to(ROOT):
            existing["runs"].append({"directory": directory.relative_to(ROOT).as_posix(),
                                      "label": f"{plan['macros']} macros: fine-tuning on {plan['width']} x {plan['height']}"})
    unique = {item["directory"]: item for item in existing["runs"]}
    write_json(path, {"runs": list(unique.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    plans = [case_plan(args, path, i) for i, path in enumerate(args.input)]
    write_json(args.output / "plan.json", {"canvas_rule": "Round sqrt(total macro area / 0.5) up to 16, respecting largest dimension", "cases": plans})
    register_runs(plans)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_case, plan) for plan in sorted(plans, key=lambda p: -p["macros"])]
        for future in as_completed(pending):
            results.append(future.result())
            write_json(args.output / "summary.json", {"complete": len(results) == len(plans), "runs": results})
    if any(row["returncode"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
