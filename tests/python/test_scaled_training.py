"""Verify larger-grid traces, new-run initialization and inspectable updates."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
import struct
import tempfile
import unittest
import zlib
import torch

from src.experiments.evaluate import grid_payload
from src.experiments.scale import case_plan
from src.experiments.train import export_rollout, prepare
from src.ppo.algorithm import PPOConfig, collect_episodes, rollout_tensors, save_checkpoint, update_policy
from src.ppo.model import ActorCritic


class ScaledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)
        self.circuit = {"blocks": {"A": [2,2], "B": [1,2], "C": [2,1]}, "nets": [["A","B","C"]]}

    def test_large_grid_compression_preserves_float32_probabilities_and_mask(self):
        mask = [(i % 100) < 45 for i in range(17000)]
        probabilities = [1/sum(mask) if valid else 0 for valid in mask]
        data = grid_payload(probabilities, mask)
        raw = zlib.decompress(base64.b64decode(data["probability_grid"]["data"]))
        recovered = struct.unpack('<17000f', raw)
        torch.testing.assert_close(torch.tensor(recovered), torch.tensor(probabilities), rtol=0, atol=0)
        actual = [False]*len(mask)
        for start, end in data["legal_mask_runs"]:
            actual[start:end] = [True]*(end-start)
        self.assertEqual(actual, mask)

    def test_canvas_rule_has_hand_checked_capacity_for_the_received_larger_inputs(self):
        root=Path(__file__).resolve().parents[2]
        args=SimpleNamespace(output=Path('steps/build/test-plan'), initialize_from=Path('model.pt'), updates=100, seed=0)
        for i,(count,area,side) in enumerate([(20,4448,96),(50,15101,176),(100,29344,256)]):
            plan=case_plan(args,root/f'benchmarks/alarik-synthetic/blknet_{count}.txt',i)
            self.assertEqual((plan['macro_area'],plan['width'],plan['height']),(area,side,side))
            self.assertLessEqual(plan['utilization'],0.5)

    def test_initialization_reuses_only_weights_and_allows_a_new_canvas(self):
        config, model = PPOConfig(), ActorCritic()
        opt=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        records,_=collect_episodes(model,[self.circuit],2,6,5)
        update_policy(model,opt,records,config)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'checkpoint.pt'
            save_checkpoint(checkpoint,model,opt,config,{'width':6,'height':5,'area_weight':.5,'update':7,'history':[],'training_seconds':2.0})
            args=SimpleNamespace(threads=1,seed=3,learning_rate=.0003,entropy=.001,epochs=2,minibatch=8,
                                 gae_lambda=1.,initialize_from=checkpoint,resume=None,width=12,height=10,area_weight=.5)
            new,new_opt,new_cfg,offset,history,seconds=prepare(args)
            for before,after in zip(model.parameters(),new.parameters()):
                torch.testing.assert_close(before,after,rtol=0,atol=0)
            self.assertFalse(new_opt.state_dict()['state'])
            self.assertEqual((offset,history,seconds,new_cfg.gae_lambda),(0,[],0.,1.))
            args.initialize_from,args.resume=None,checkpoint
            with self.assertRaisesRegex(ValueError,'width'):
                prepare(args)

    def test_exported_gae_numbers_and_transitions_match_the_actual_episode(self):
        model,config=ActorCritic(),PPOConfig(gae_lambda=1,epochs=1,minibatch=8)
        records,_=collect_episodes(model,[self.circuit],3,6,5)
        targets=rollout_tensors(records,config,include_details=True)
        rows=export_rollout(records,targets,config)
        for i,row in enumerate(rows):
            self.assertIn(row['macro'],row['placements_after'])
            self.assertNotIn(row['macro'],row['placements_before'])
            self.assertAlmostEqual(row['delta'],row['reward']+row['next_value']-row['value'])
            self.assertAlmostEqual(row['return'],row['advantage_raw']+row['value'],places=6)
            if not row['boundary']:
                self.assertEqual(row['placements_after'],rows[i+1]['placements_before'])
                self.assertAlmostEqual(row['advantage_raw'],row['delta']+rows[i+1]['advantage_raw'],places=6)

    def test_recorded_optimizer_example_matches_the_loss_and_changed_probability(self):
        model,config=ActorCritic(),PPOConfig(epochs=1,minibatch=8)
        opt=torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        records,_=collect_episodes(model,[self.circuit],4,6,5)
        stats=update_policy(model,opt,records,config)
        example=stats['example']
        self.assertAlmostEqual(example['selected_objective'],min(example['unclipped_objective'],example['clipped_objective']))
        self.assertLessEqual(example['gradient_norm_after_clip'],config.max_grad_norm+1e-5)
        self.assertNotEqual(example['log_prob_before_step'],example['log_prob_after_step'])
        self.assertTrue(example['minibatch_stats'])


if __name__=='__main__':
    unittest.main()
