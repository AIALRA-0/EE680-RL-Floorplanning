"""Replay real deterministic placement transitions before adding PPO."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from integration.visualization.preview import placement_trace
from src.environment.placement_env import PlacementEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "steps/00-input/build/circuit.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    args = parser.parse_args()
    circuit = json.loads(args.input.read_text(encoding="utf-8"))
    trace = placement_trace(PlacementEnv(circuit, args.width, args.height))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trace.json").write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    final = trace["states"][-1]
    print(json.dumps({"policy": trace["policy"], "transitions": len(trace["states"]) - 1,
                      **final["info"], "reward": final["reward"]}, indent=2))
    print("For every candidate check and actual execution: run python steps/serve.py")


if __name__ == "__main__":
    main()
