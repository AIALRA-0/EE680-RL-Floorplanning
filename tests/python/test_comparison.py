"""Check shared physical rules, original export and adapted search decisions."""
import json
import math
from pathlib import Path
import random
import unittest

from cpp.sa_sp.adapter import export_positions, load_original, run_original
from cpp.sa_sp.aligned import accept_proposal, decode, run_aligned, shelf_sequences
from integration.evaluation.protocol import comparison_protocol, score_layout
from src.data.parse_blknet import load_circuit

ROOT = Path(__file__).resolve().parents[2]


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.circuit = {"blocks": {"A": [2, 2], "B": [1, 2]}, "nets": [["A", "B"]]}
        self.protocol = comparison_protocol(10, 10)
        self.positions = {"A": {"x": 3, "y": 2}, "B": {"x": 5, "y": 2}}

    def test_shared_cost_uses_tight_bbox_and_lower_left_endpoints(self):
        result = score_layout(self.circuit, self.positions, self.protocol)
        self.assertTrue(result["legal"])
        self.assertEqual((result["area"], result["hpwl"], result["per_net_hpwl"]), (6, 2, [2]))
        self.assertAlmostEqual(result["normalized_cost"], 0.08)
        self.assertAlmostEqual(result["reward"], -0.08)

    def test_rotation_fractional_anchors_and_incomplete_outputs_are_rejected(self):
        for change in ({"rotated": True}, {"x": 5.5}):
            positions = {**self.positions, "B": {**self.positions["B"], **change}}
            result = score_layout(self.circuit, positions, self.protocol)
            self.assertFalse(result["legal"])
            self.assertIsNone(result["normalized_cost"])
        result = score_layout(self.circuit, {"A": self.positions["A"]}, self.protocol)
        self.assertFalse(result["complete"])
        self.assertEqual(result["reward"], -2)

    def test_export_preserves_rotation_and_rejects_unexplained_dimensions(self):
        result = export_positions(self.circuit, ["A", "B"], [0, 2], [0, 0], [2, 2], [2, 1])
        self.assertFalse(result["A"]["rotated"])
        self.assertTrue(result["B"]["rotated"])
        with self.assertRaises(ValueError):
            export_positions(self.circuit, ["A", "B"], [0, 2], [0, 0], [2, 8], [2, 1])

    def test_received_circuits_have_independently_legal_shelf_initializations(self):
        module = load_original()
        for count, side in [(5,64), (10,64), (20,96), (50,176), (100,256)]:
            circuit = load_circuit(ROOT / f"benchmarks/alarik-synthetic/blknet_{count}.txt")
            protocol = comparison_protocol(side, side)
            state = decode(module, circuit, protocol, shelf_sequences(circuit, side))
            x, y, widths, heights = state["arrays"]
            positions = export_positions(circuit, list(circuit["blocks"]), x, y, widths, heights)
            self.assertTrue(score_layout(circuit, positions, protocol)["legal"], count)

    def test_metropolis_acceptance_matches_the_draw_and_hard_board_rejection(self):
        for seed in (0, 1):
            draw = random.Random(seed).random()
            accepted, probability, actual = accept_proposal(1, 1, True, random.Random(seed))
            self.assertAlmostEqual(probability, math.exp(-1))
            self.assertEqual((accepted, actual), (draw < probability, draw))
        self.assertEqual(accept_proposal(-1, 1, True, random.Random(0)), (True, 1, None))
        self.assertEqual(accept_proposal(-1, 1, False, random.Random(0)), (False, 0, None))

    def test_aligned_trace_follows_acceptance_and_retains_best_cost(self):
        result = run_aligned(self.circuit, self.protocol, seed=2, stages=3, moves_per_temperature=12)
        costs = [stage["best_cost"] for stage in result["history"]]
        self.assertEqual(costs, sorted(costs, reverse=True))
        self.assertEqual(result["moves"], 36)
        self.assertTrue(result["validation"]["legal"])
        for event in result["trace"]:
            self.assertEqual(event["after"], event["candidate"] if event["accepted"] else event["before"])
        self.assertEqual(result["trace"][-1]["iteration"], 36)

    def test_unchanged_native_five_macro_run_reproduces_the_exported_metrics(self):
        circuit = load_circuit(ROOT / "benchmarks/alarik-synthetic/blknet_5.txt")
        result = run_original(circuit, seed=0)
        self.assertEqual((result["final"]["validation"]["area"], result["final"]["validation"]["hpwl"]), (880,84))
        self.assertEqual(result["moves"], 36000)


if __name__ == "__main__":
    unittest.main()
