"""Load Alarik's unchanged Python solver and export its actual returned layouts."""

import importlib.util
from pathlib import Path
import random

from integration.evaluation.placement import evaluate_placement


def load_original():
    path = Path(__file__).with_name("seqpair_usc.py")
    spec = importlib.util.spec_from_file_location("ee680_received_sa_sp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_positions(circuit, names, x, y, widths, heights):
    positions = {}
    for index, name in enumerate(names):
        original = tuple(circuit["blocks"][name])
        dimensions = (float(widths[index]), float(heights[index]))
        if dimensions not in (original, original[::-1]):
            raise ValueError(f"Returned dimensions disagree with input: {name}")
        positions[name] = {"x": float(x[index]), "y": float(y[index]),
                           "rotated": dimensions != original}
    return positions


def original_snapshot(circuit, arrays, width, height, pair):
    positions = export_positions(circuit, list(circuit["blocks"]), *arrays)
    checked = evaluate_placement(circuit, positions, width, height)
    if not checked["legal"]:
        raise AssertionError(checked["errors"])
    return {"placements": positions, "bounds": [width, height],
            "sequence_pair": pair, "validation": checked}


def run_original(circuit, seed=0, alpha=0.5):
    module = load_original()
    module.random = random.Random(seed)
    names = list(circuit["blocks"])
    result = module.simulated_annealing(circuit["blocks"], names, circuit["nets"],
                                       {name: i for i, name in enumerate(names)}, alpha, 1-alpha, verbose=False)
    initial = original_snapshot(circuit, result[:4], result[4], result[5], [result[8], result[9]])
    final = original_snapshot(circuit, result[11:15], result[15], result[16], [result[19], result[20]])
    for state, area, hpwl in [(initial, result[6], result[7]), (final, result[17], result[18])]:
        if state["validation"]["area"] != area or state["validation"]["hpwl"] != hpwl:
            raise AssertionError("The independent scorer disagrees with the original solver")
    return {"algorithm": "sa_sp_original", "seed": seed, "initial": initial, "final": final,
            "runtime_seconds": result[22], "moves": result[23], "accepted": result[24],
            "rotation": True, "normalization": "initial native Area and HPWL",
            "source": "cpp/sa_sp/seqpair_usc.py"}
