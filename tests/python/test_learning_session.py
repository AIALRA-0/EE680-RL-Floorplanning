"""Verify that finding a legal anchor never skips or silently executes a candidate."""

from pathlib import Path
import unittest

from integration.visualization.learning_session import LearningSession
from src.data.parse_blknet import load_circuit


def sample():
    return load_circuit(Path(__file__).resolve().parents[2] / "benchmarks/alarik-synthetic/blknet_5.txt")


class LearningSessionTests(unittest.TestCase):
    def test_b1_rejects_every_anchor_before_25_and_waits_before_applying(self):
        session = LearningSession(sample())
        session.command("module")
        before = session.env.observation()
        state = session.command("choose")
        self.assertEqual(state["phase"], "choose")
        self.assertEqual(state["chosen"], [25, 0])
        self.assertEqual(session.env.observation(), before)
        self.assertEqual(len(state["attempts"]), 26)
        self.assertTrue(all(a["legal"] is False for a in state["attempts"][:25]))
        self.assertTrue(all(a["error"] == "Macro would overlap B0" for a in state["attempts"][:25]))
        self.assertTrue(state["attempts"][25]["legal"])
        last_rejected = session.attempt(1, 24)["report"]["checks"][-1]
        first_accepted = session.attempt(1, 25)["report"]["checks"][-1]
        self.assertEqual(last_rejected["comparisons"][0], [24, 25])
        self.assertFalse(last_rejected["passed"])
        self.assertEqual(first_accepted["comparisons"][0], [25, 25])
        self.assertTrue(first_accepted["passed"])
        session.command("advance")
        self.assertEqual(session.env.placements["B1"], {"x": 25, "y": 0, "rotated": False})

    def test_single_advances_show_bounds_overlap_mask_rejection_and_increment(self):
        session = LearningSession(sample())
        session.command("module")
        before = session.env.observation()
        phases = [session.command("advance")["phase"] for _ in range(6)]
        self.assertEqual(phases, ["candidate", "bounds", "overlap", "mask", "reject", "candidate"])
        self.assertEqual(session.current["index"], 1)
        self.assertEqual(session.env.observation(), before)
        self.assertEqual(session.snapshot()["attempts"][-1]["legal"], None)

    def test_mask_is_unknown_until_written_and_is_rebuilt_for_the_next_macro(self):
        session = LearningSession(sample())
        self.assertEqual(session.snapshot()["mask"], [None] * 4096)
        session.command("advance")
        session.command("advance")
        self.assertIsNone(session.snapshot()["mask"][0])
        session.command("advance")
        self.assertTrue(session.snapshot()["mask"][0])
        self.assertTrue(all(value is None for value in session.snapshot()["mask"][1:]))
        session.command("module")
        self.assertTrue(all(value is None for value in session.snapshot()["mask"]))
        for _ in range(4):
            session.command("advance")
        self.assertFalse(session.snapshot()["mask"][0])

    def test_b4_search_records_row_wrap_and_full_rectangle_bounds(self):
        session = LearningSession(sample())
        for _ in range(4):
            session.command("module")
        state = session.command("choose")
        self.assertEqual(state["chosen"], [30, 12])
        self.assertEqual(len(state["attempts"]), 799)
        self.assertEqual([state["attempts"][64][key] for key in ("x", "y")], [0, 1])
        outside = session.attempt(4, 37)["report"]
        self.assertEqual(outside["error"], "Macro would extend outside the board")
        self.assertEqual(outside["checks"][0]["right"], 65)
        self.assertFalse(outside["checks"][0]["passed"])
        self.assertEqual(len(outside["checks"]), 1)

    def test_metrics_and_feedback_are_unfolded_after_actual_execution(self):
        session = LearningSession(sample())
        session.command("choose")
        phases = [session.command("advance")["phase"] for _ in range(4)]
        self.assertEqual(phases, ["apply", "metrics", "reward", "next_observation"])
        transition = session.transitions[-1]
        self.assertEqual(transition["observation"]["placements"], {})
        self.assertEqual(transition["before_info"]["metrics"]["area"], 0)
        self.assertIsNone(transition["before_details"]["bounds"])
        self.assertEqual(transition["action"], [0, 0])
        self.assertEqual(transition["reward"], 0)
        self.assertEqual(transition["next_observation"]["current_macro"], "B1")
        details = session.env.metric_details()
        self.assertEqual(details["bounds"], {"left": 0, "bottom": 0, "right": 25, "top": 8})
        self.assertEqual(sum(net["hpwl"] for net in details["nets"]), 0)

    def test_completed_episode_and_replayed_commands_match_every_event(self):
        session = LearningSession(sample())
        for _ in range(5):
            session.command("module")
        self.assertEqual(session.phase, "done")
        self.assertEqual(session.env.metrics()["area"], 1624)
        self.assertEqual(session.env.metrics()["hpwl"], 207)
        self.assertAlmostEqual(session.env.reward, -0.3599609375)
        self.assertEqual(len(session.transitions), 5)
        self.assertEqual(sum(net["hpwl"] for net in session.env.metric_details()["nets"]), 207)
        restored = LearningSession(sample())
        for command in session.commands:
            restored.command(command)
        self.assertEqual(restored.events, session.events)
        self.assertEqual(restored.snapshot(), session.snapshot())

    def test_failure_is_preserved_and_unknown_command_changes_nothing(self):
        session = LearningSession({"blocks": {"A": [2, 2], "B": [2, 2]}, "nets": []}, 2, 2)
        session.command("module")
        self.assertEqual(session.phase, "done")
        self.assertEqual(session.env.reason, "no_legal_action")
        self.assertEqual(session.transitions[-1]["reward"], -2)
        before = session.snapshot()
        with self.assertRaises(ValueError):
            session.command("invented")
        self.assertEqual(session.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
