# What changed in the 20/50/100-macro experiments

Experiment date: 2026-10-19, America/Los_Angeles. The runs use Alarik's supplied
synthetic inputs and the shared 5/10-macro update-300 weights. Each larger-input
run starts a fresh Adam optimizer, RNG stream and update count.

## Which conditions were held fixed

Canvas planning targets roughly 50% utilization, rounds the square side up to a
multiple of 16 and ensures the largest macro fits. Before/after evaluations keep
their own selected board fixed.

| Macros | Total macro area | Board | Utilization | Updates | Episodes/update | Epochs | Minibatch |
|---|---:|---|---:|---:|---:|---:|---:|
| 20 | 4448 | 96 x 96 | 48.26% | 100 | 4 | 3 | 16 |
| 50 | 15101 | 176 x 176 | 48.75% | 100 | 4 | 2 | 8 |
| 100 | 29344 | 256 x 256 | 44.78% | 100 | 2 | 2 | 8 |

Common settings: training seed 0, area weight 0.5, learning rate 0.0003, gamma 1,
GAE lambda 1, PPO clip 0.2, entropy coefficient 0.001, gradient norm limit 0.5 and
approximate-KL threshold 0.03. Runtime: Python 3.12.7 and PyTorch 2.6.0+cpu on
Windows, one PyTorch thread per process, at most two concurrent training processes.

The three runs completed 1000 training episodes and 48000 transitions; all these
training episodes completed. Their training wall times, including evaluation and
saving, were 148.5, 963.5 and 1993.3 seconds. Shared small-input pretraining took
348.6 seconds, reported separately.

## How do sampled evaluations change

Each stage samples 32 episodes at evaluation seed 100000. Before fine-tuning, the
policy already has small-input training weights. Completion counts and reward
retain failures; Area and HPWL means include complete legal layouts.

| Macros | Stage | Legal/total | Mean reward | Mean Area | Mean HPWL |
|---|---|---:|---:|---:|---:|
| 20 | Before | 32/32 | -0.586113 | 6717.906 | 1702.219 |
| 20 | Update 100 | 32/32 | -0.572330 | 6427.344 | 1717.438 |
| 50 | Before | 32/32 | -0.548048 | 22709.188 | 6388.344 |
| 50 | Update 100 | 32/32 | -0.533880 | 21845.313 | 6380.469 |
| 100 | Before | 32/32 | -0.531940 | 44616.625 | 19613.938 |
| 100 | Update 100 | 32/32 | -0.516925 | 43476.500 | 18967.125 |

The 20-macro sampled mean Area improves while HPWL increases slightly. Both means
improve for 50 and 100 macros. These fixed-input runs use one training seed;
formal unseen-circuit and multiple-training-seed studies remain next work.

## Why inspect greedy replay and the failed probe separately

Argmax chooses the largest-probability legal action; sampled evaluation draws
from the distribution. Their trajectories can differ with the same weights.

| Macros | Before Area / HPWL | After Area / HPWL | Independent check |
|---|---:|---:|---|
| 20 | 5642 / 1666 | 5796 / 1831 | Both complete and legal |
| 50 | 19436 / 5825 | 19314 / 5756 | Both complete and legal |
| 100 | 36864 / 17615 | 34304 / 17658 | Both complete and legal |

The 20-macro greedy result worsens on both metrics. The 100-macro greedy Area
improves while HPWL increases. The viewer retains these actual layouts.

The 100-macro 192 x 192 probe ran only two updates. Sampled evaluation completed
0/32 before and after, with reward -2, while one greedy layout was legal (Area
35150, HPWL 16951). Incomplete sampled trajectories remain visible. The expanded
256 x 256 run is a separate experiment.

## How do saved checkpoints connect to the explanations

Larger runs retain checkpoints 0/25/50/75/100; the failed probe retains 0/2 and the
original 5/10 experiment retains update 300. Eighteen isolated teaching updates
contain 2077 transitions. Each loads an official checkpoint, samples separately,
performs one real PPO update and saves separate JSON/model files without changing
the official checkpoint.

Records expose positions before/after, features, old log probability, value,
reward, termination, delta, GAE, returns and normalized advantages. The first
minibatch records clipping, losses, gradient norms and the same action's
probability/value before and after one Adam step. Full masks and float32
probability grids use lossless encoding.

Start `python steps/serve.py` and open `/steps/reading.html`. Chapters 02-04 embed
the experiment selector. Published evidence works without retraining. For a new
reproduction, use a new output directory:

```powershell
python -m src.experiments.scale --input benchmarks/alarik-synthetic/blknet_20.txt benchmarks/alarik-synthetic/blknet_50.txt benchmarks/alarik-synthetic/blknet_100.txt --initialize-from steps/evidence/ppo-5-10-seed0/checkpoint.pt --output steps/build/reproduction-scale --updates 100 --workers 2 --seed 0
python -m src.experiments.study --run steps/build/reproduction-scale/case-0-macros-20 --teaching-episodes 2 --eval-episodes 32
python steps/check_course.py
```

`--initialize-from` reuses weights only. `--resume` restores the original optimizer,
RNG and progress and requires matching canvas/reward settings. Read the
[first run](ppo-first-run.md), [PPO design](ppo-design.md),
[learning workflow](learning-workflow.md) and [comparison](ppo-sa-sp-comparison.md)
for related evidence.
