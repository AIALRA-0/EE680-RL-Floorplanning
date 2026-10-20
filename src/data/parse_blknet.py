"""Read BLOCKS/NETS text into a circuit without creating a placement."""

import argparse
import math
from pathlib import Path


def load_circuit(path):
    """Return blocks as ID-to-size pairs and nets as ordered lists of IDs."""
    path = Path(path)
    blocks = {}
    nets = []
    section = None

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        location = f"{path.name}:{line_number}"
        header = line.upper()
        if header == "BLOCKS":
            if section is not None:
                raise ValueError(f"{location}: BLOCKS must appear once, before NETS")
            section = "BLOCKS"
            continue
        if header == "NETS":
            if section != "BLOCKS":
                raise ValueError(f"{location}: NETS must appear once, after BLOCKS")
            section = "NETS"
            continue

        values = line.split()
        if section == "BLOCKS":
            if len(values) != 3:
                raise ValueError(f"{location}: expected block ID, width, and height")
            name, width_text, height_text = values
            if name in blocks:
                raise ValueError(f"{location}: duplicate block ID {name!r}")
            try:
                width, height = float(width_text), float(height_text)
            except ValueError as exc:
                raise ValueError(f"{location}: block dimensions must be numbers") from exc
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise ValueError(f"{location}: block dimensions must be finite and positive")
            blocks[name] = (width, height)
        elif section == "NETS":
            for name in values:
                if name not in blocks:
                    raise ValueError(f"{location}: net references unknown block {name!r}")
            nets.append(values)
        else:
            raise ValueError(f"{location}: expected BLOCKS before circuit data")

    if not blocks:
        raise ValueError(f"{path.name}: circuit must contain at least one block")
    if section != "NETS":
        raise ValueError(f"{path.name}: missing NETS section")
    return {"blocks": blocks, "nets": nets}


def main():
    """Print a small summary so the parsed input can be inspected directly."""
    parser = argparse.ArgumentParser(description="Read a BLOCKS/NETS circuit file")
    parser.add_argument("input", type=Path, help="Path to the circuit text file")
    args = parser.parse_args()
    try:
        circuit = load_circuit(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    blocks, nets = circuit["blocks"], circuit["nets"]
    print(f"{len(blocks)} blocks, {len(nets)} nets")
    first_block = next(iter(blocks))
    print(f"{first_block} = {blocks[first_block]}")
    if nets:
        print(f"Net 0 = {nets[0]}")


if __name__ == "__main__":
    main()
