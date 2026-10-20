"""Use hand-calculated layouts to check transitions, geometry, and reward."""

import unittest

from src.environment.placement_env import PlacementEnv


def toy_circuit():
    return {"blocks": {"A": (2, 2), "B": (1, 2), "C": (2, 1)},
            "nets": [["A", "B", "C"]]}


class PlacementEnvironmentTests(unittest.TestCase):
    def test_reset_exposes_geometry_connections_and_empty_placement(self):
        circuit = toy_circuit()
        env = PlacementEnv(circuit, 4, 4)
        observation, info = env.reset()
        self.assertEqual(observation["circuit"], circuit)
        self.assertEqual(observation["current_macro"], "A")
        self.assertEqual(observation["placements"], {})
        self.assertEqual(info["metrics"]["area"], 0)
        observation["circuit"]["blocks"]["A"] = (99, 99)
        self.assertEqual(env.circuit["blocks"]["A"], (2, 2))

    def test_mask_uses_full_rectangle_and_row_major_indices(self):
        env = PlacementEnv(toy_circuit(), 4, 4)
        mask = env.legal_mask()
        self.assertEqual(len(mask), 16)
        self.assertEqual(sum(mask), 9)
        self.assertTrue(mask[2 * 4 + 2])
        self.assertFalse(mask[3])
        self.assertFalse(mask[3 * 4])

    def test_touching_edges_are_legal_but_positive_overlap_is_not(self):
        env = PlacementEnv(toy_circuit(), 4, 4)
        env.step((0, 0))
        self.assertTrue(env.preview(2, 0)["legal"])
        self.assertFalse(env.preview(1, 0)["legal"])
        self.assertFalse(env.preview(-1, 0)["legal"])
        self.assertFalse(env.preview(4, 0)["legal"])

    def test_rejected_action_and_candidate_preview_do_not_change_state(self):
        env = PlacementEnv(toy_circuit(), 4, 4)
        env.step((0, 0))
        before = env.observation()
        self.assertEqual(env.preview(2, 0)["metrics"]["area"], 6)
        with self.assertRaisesRegex(ValueError, "overlap"):
            env.step((1, 0))
        self.assertEqual(env.observation(), before)
        for action in ((True, 0), (0.5, 0)):
            with self.assertRaisesRegex(ValueError, "integers"):
                env.step(action)

    def test_hand_calculated_complete_layout_and_terminal_reward(self):
        env = PlacementEnv(toy_circuit(), 4, 4)
        for action in ((0, 0), (2, 0)):
            _, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["metrics"]["complete"])
        observation, reward, terminated, truncated, info = env.step((0, 2))
        self.assertEqual(info["metrics"], {"area": 9, "hpwl": 4, "placed": 3, "total": 3, "complete": True})
        self.assertAlmostEqual(reward, -0.53125)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(observation["current_macro"])
        self.assertFalse(any(env.legal_mask()))
        with self.assertRaisesRegex(ValueError, "ended"):
            env.step((0, 0))

    def test_no_legal_continuation_is_recorded_as_failure(self):
        env = PlacementEnv({"blocks": {"A": (2, 2), "B": (2, 2)}, "nets": [["A", "B"]]}, 2, 2)
        _, reward, terminated, _, info = env.step((0, 0))
        self.assertTrue(terminated)
        self.assertEqual(info["reason"], "no_legal_action")
        self.assertEqual(reward, -2)
        self.assertFalse(info["metrics"]["complete"])

    def test_oversized_first_macro_has_no_legal_initial_action(self):
        env = PlacementEnv({"blocks": {"A": (5, 1)}, "nets": []}, 4, 4)
        self.assertTrue(env.terminated)
        self.assertEqual(env.reason, "no_legal_action")
        self.assertFalse(any(env.legal_mask()))

    def test_partial_hpwl_is_labeled_incomplete(self):
        env = PlacementEnv(toy_circuit(), 4, 4)
        env.step((0, 0))
        env.step((2, 0))
        self.assertEqual(env.metrics()["hpwl"], 2)
        self.assertFalse(env.metrics()["complete"])

    def test_translated_layout_keeps_same_bounding_box_area(self):
        env = PlacementEnv(toy_circuit(), 8, 8)
        env.step((3, 3))
        env.step((5, 3))
        env.step((3, 5))
        self.assertEqual(env.metrics()["area"], 9)
        self.assertEqual(env.metrics()["hpwl"], 4)

    def test_reset_removes_old_episode_and_preserves_original_circuit(self):
        circuit = toy_circuit()
        env = PlacementEnv(circuit, 4, 4)
        env.step((0, 0))
        observation, _ = env.reset()
        self.assertEqual(observation["current_macro"], "A")
        self.assertEqual(observation["placements"], {})
        self.assertEqual(circuit, toy_circuit())


if __name__ == "__main__":
    unittest.main()
