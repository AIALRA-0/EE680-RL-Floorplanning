"""Parse a circuit and export the data used by the next learning step."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.parse_blknet import load_circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "benchmarks/alarik-synthetic/blknet_5.txt")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "build")
    args = parser.parse_args()
    circuit = load_circuit(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "circuit.json").write_text(json.dumps(circuit, indent=2), encoding="utf-8")
    print(f"Parsed {len(circuit['blocks'])} blocks and {len(circuit['nets'])} nets")
    print(f"Data: {(args.output / 'circuit.json').resolve()}")
    print("Maintained learning page: run python steps/serve.py")


if __name__ == "__main__":
    main()
