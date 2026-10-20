"""Check recorded actions, candidate scoring, and early failure without a browser."""

from pathlib import Path
import unittest

from integration.visualization.preview import placement_trace
from src.data.parse_blknet import load_circuit
from src.environment.placement_env import PlacementEnv


class PlacementTraceTests(unittest.TestCase):
    def test_five_block_trace_has_expected_layout_and_replays_exactly(self):
        root = Path(__file__).resolve().parents[2]
        circuit = load_circuit(root / "benchmarks/alarik-synthetic/blknet_5.txt")
        trace = placement_trace(PlacementEnv(circuit))
        states = trace["states"]
        self.assertEqual([s["next_action"] for s in states],
                         [[0, 0], [25, 0], [30, 0], [43, 0], [30, 12], None])
        self.assertEqual(states[-1]["info"]["metrics"]["area"], 1624)
        self.assertEqual(states[-1]["info"]["metrics"]["hpwl"], 207)
        env = PlacementEnv(circuit)
        for state in states:
            self.assertEqual(state["observation"], env.observation())
            self.assertEqual(state["info"], env.info())
            self.assertEqual(state["reward"], env.reward)
            for index in (0, 25, 64, 798, 4095):
                preview = env.preview(index % 64, index // 64)
                expected = [preview["metrics"][key] for key in ("area", "hpwl")] if preview["legal"] else None
                self.assertEqual(state["candidates"][index], expected)
            if state["next_action"] is not None:
                env.step(state["next_action"])

    def test_failed_initial_state_exports_without_an_action(self):
        trace = placement_trace(PlacementEnv({"blocks": {"A": [5, 1]}, "nets": []}, 4, 4))
        self.assertEqual(len(trace["states"]), 1)
        state = trace["states"][0]
        self.assertIsNone(state["next_action"])
        self.assertEqual(state["info"]["reason"], "no_legal_action")
        self.assertEqual(state["reward"], -2)

if __name__ == "__main__":
    unittest.main()
