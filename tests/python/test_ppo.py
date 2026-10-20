"""Check actual masking, graph inputs, GAE, PPO gradients and checkpoint replay."""

from pathlib import Path
import random
import tempfile
import unittest
import torch
from torch.distributions import Categorical

from integration.evaluation.placement import evaluate_placement
from src.environment.placement_env import PlacementEnv
from src.experiments.evaluate import run_episode
from src.ppo.algorithm import (PPOConfig, collect_episodes, generalized_advantages,
                               load_checkpoint, ppo_loss, save_checkpoint, update_policy)
from src.ppo.encoding import collate, encode_observation
from src.ppo.model import ActorCritic, masked_distribution


def circuit():
    return {"blocks": {"A": (2, 2), "B": (1, 2), "C": (2, 1)}, "nets": [["A", "B", "C"]]}


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_fast_mask_matches_scalar_geometry_with_fractional_sizes(self):
        rng = random.Random(8)
        for _ in range(30):
            sample = {"blocks": {str(i): (rng.choice([1, 1.5, 2, 2.25]),
                                           rng.choice([1, 1.5, 2, 2.25])) for i in range(4)}, "nets": []}
            env = PlacementEnv(sample, 8, 7)
            while not env.terminated:
                expected = [env.inspect_action(x, y)["legal"] for y in range(7) for x in range(8)]
                self.assertEqual(env.legal_mask(), expected)
                action = rng.choice([i for i, legal in enumerate(expected) if legal])
                env.step((action % 8, action // 8))
            self.assertFalse(any(env.legal_mask()))

    def test_candidate_features_match_independent_candidate_previews(self):
        env = PlacementEnv(circuit(), 6, 5)
        env.step((1, 1))
        encoded = encode_observation(env.observation(), env.legal_mask())
        self.assertEqual(tuple(encoded["nodes"].shape), (3, 8))
        self.assertEqual(encoded["incidence"].tolist(), [[1, 1, 1]])
        for i, legal in enumerate(env.legal_mask()):
            if legal:
                score = env.preview(i % 6, i // 6)["metrics"]
                self.assertAlmostEqual(float(encoded["candidates"][i, 2]), score["area"] / 30, places=6)
                self.assertAlmostEqual(float(encoded["candidates"][i, 3]), score["hpwl"] / 11, places=6)

    def test_masked_actions_have_exactly_zero_probability_and_gradient(self):
        logits = torch.tensor([[1.0, 999.0, -1.0]], requires_grad=True)
        mask = torch.tensor([[True, False, True]])
        distribution = masked_distribution(logits, mask)
        self.assertEqual(float(distribution.probs[0, 1]), 0)
        self.assertNotIn(1, distribution.sample((1000,)).flatten().tolist())
        (-distribution.log_prob(torch.tensor([0]))).backward()
        self.assertEqual(float(logits.grad[0, 1]), 0)
        with self.assertRaisesRegex(ValueError, "No legal"):
            masked_distribution(logits, torch.zeros_like(mask))

    def test_variable_graph_sizes_and_no_net_graph_are_supported(self):
        env = PlacementEnv(circuit(), 6, 5)
        small = PlacementEnv({"blocks": {"renamed": (1, 1)}, "nets": []}, 6, 5)
        batch = collate([encode_observation(e.observation(), e.legal_mask()) for e in (env, small)])
        model = ActorCritic()
        distribution, values = model.distribution(batch)
        self.assertEqual(tuple(distribution.probs.shape), (2, 30))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertTrue(torch.isfinite(values).all())
        with torch.no_grad():
            alone = model.distribution(collate([encode_observation(small.observation(), small.legal_mask())]))
        torch.testing.assert_close(distribution.probs[1], alone[0].probs[0])

    def test_graph_connections_and_positions_reach_the_network(self):
        first = PlacementEnv(circuit(), 6, 5)
        first.step((0, 0))
        changed = PlacementEnv({"blocks": circuit()["blocks"], "nets": [["A", "C"]]}, 6, 5)
        changed.step((0, 0))
        model = ActorCritic()
        a, b = [model(collate([encode_observation(e.observation(), e.legal_mask())])) for e in (first, changed)]
        self.assertGreater(float((a[0] - b[0]).abs().max().detach()), 0)
        self.assertGreater(float((a[1] - b[1]).abs().max().detach()), 0)

    def test_gae_matches_hand_returns_and_bootstraps_time_limits(self):
        adv, returns = generalized_advantages(torch.tensor([1., 2.]), torch.tensor([.5, .25]),
                                              torch.tensor([.25, 0.]), torch.tensor([False, True]),
                                              torch.tensor([False, True]), gamma=1., lam=1.)
        torch.testing.assert_close(adv, torch.tensor([2.5, 1.75]))
        torch.testing.assert_close(returns, torch.tensor([3., 2.]))
        adv, _ = generalized_advantages(torch.tensor([1., 9.]), torch.tensor([.5, 0.]),
                                        torch.tensor([7., 0.]), torch.tensor([False, True]),
                                        torch.tensor([True, True]), gamma=1., lam=1.)
        self.assertAlmostEqual(float(adv[0]), 7.5)

    def test_clipped_surrogate_matches_positive_and_negative_hand_examples(self):
        distribution = Categorical(probs=torch.tensor([[.3, .7]]))
        for advantage, expected in ((2., -2.4), (-2., 3.)):
            loss, _ = ppo_loss(distribution, torch.zeros(1), torch.tensor([0]),
                               torch.tensor([.2]).log(), torch.tensor([advantage]), torch.zeros(1),
                               PPOConfig(value_coefficient=0, entropy_coefficient=0))
            self.assertAlmostEqual(float(loss), expected, places=5)

    def test_update_changes_parameters_and_loaded_model_replays_identically(self):
        model, config = ActorCritic(), PPOConfig(epochs=2, minibatch=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        records, episodes = collect_episodes(model, [circuit()], 8, 6, 5)
        self.assertEqual(len(episodes), 8)
        self.assertTrue(all(r["boundary"] == r["terminated"] for r in records))
        stats = update_policy(model, optimizer, records, config)
        self.assertGreater(stats["parameter_delta_l2"], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, config, {"width": 6, "height": 5})
            restored, restored_optimizer, _, saved = load_checkpoint(path)
            a, b = [run_episode(circuit(), m, 6, 5) for m in (model, restored)]
            self.assertEqual(a["placements"], b["placements"])
            self.assertEqual(a["metrics"], b["metrics"])
            self.assertTrue(restored_optimizer.state_dict()["state"])
            self.assertEqual(saved["metadata"]["width"], 6)

    def test_resume_restores_rng_and_reproduces_the_next_training_update(self):
        model, config = ActorCritic(), PPOConfig(epochs=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        records, _ = collect_episodes(model, [circuit()], 4, 6, 5)
        update_policy(model, optimizer, records, config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, config, {})
            original_rng = torch.get_rng_state()
            records, _ = collect_episodes(model, [circuit()], 4, 6, 5)
            update_policy(model, optimizer, records, config)
            restored, opt, cfg, saved = load_checkpoint(path)
            torch.set_rng_state(saved["torch_rng"])
            self.assertTrue(torch.equal(original_rng, saved["torch_rng"]))
            repeated, _ = collect_episodes(restored, [circuit()], 4, 6, 5)
            self.assertEqual([r["action"] for r in records], [r["action"] for r in repeated])
            update_policy(restored, opt, repeated, cfg)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_independent_evaluator_detects_overlap_and_scores_hand_layout(self):
        positions = {"A": {"x": 0, "y": 0}, "B": {"x": 2, "y": 0}, "C": {"x": 0, "y": 2}}
        result = evaluate_placement(circuit(), positions, 4, 4)
        self.assertTrue(result["legal"])
        self.assertEqual((result["area"], result["hpwl"], result["per_net_hpwl"]), (9, 4, [4]))
        positions["B"]["x"] = 1
        self.assertFalse(evaluate_placement(circuit(), positions, 4, 4)["legal"])

    def test_weight_ablation_uses_the_same_metrics_and_different_reward(self):
        results = []
        for weight in (0., .5, 1.):
            env = PlacementEnv(circuit(), 4, 4, weight)
            for action in ((0, 0), (2, 0), (0, 2)):
                env.step(action)
            results.append(env.reward)
        self.assertEqual(results, [-.5, -.53125, -.5625])


if __name__ == "__main__":
    unittest.main()
