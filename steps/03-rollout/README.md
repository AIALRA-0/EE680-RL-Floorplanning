# What each PPO rollout contains

Chapter **03** has 14 explanations and displays real random samples, separately
from chapter 02's deterministic evaluation replay. Each sample contains the old
state/mask, action, log probability, value, actual reward and episode boundaries.
GAE computes returns before normalizing advantages.

To export a new bounded rollout without changing the checkpoint:

```powershell
# Sample four episodes, preserving their actual graph inputs and masks.
python steps/03-rollout/run.py --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/03-rollout/build/rollout.json --episodes 4 --seed 10000
```

The implementation is `src/ppo/algorithm.py`. Its collector finishes episodes;
an impossible continuation ends with a failure reward and is retained in the
records. A failed initial state raises a clear error because there is no action
from which to train.

For complete teaching records matched to every saved checkpoint, run:

```powershell
python -m src.experiments.study --run steps/build/ppo-scale-20261019/case-0-macros-20 --teaching-episodes 2 --eval-episodes 32
```

This loads each official checkpoint, records an isolated teaching update and
saves it separately as `teaching-update-N.json` and `.pt`. It never overwrites an
official model. The page labels the source checkpoint and shows each sample's
before/after positions, reward, terminal flag, delta, raw advantage, return and
normalized advantage. Its sample selector follows actual episode order.
