# How PPO updates are run and inspected

Chapter **04** reads real update statistics and plots mean training return and
legal completion rate. It shows the parameter-change norm, losses, approximate
KL and the gradient norm measured before clipping. The [design](../../docs/ppo-design.md)
connects these fields to the original PPO and GAE papers.

From the repository root, use Python 3.12 and PyTorch 2.6.0. The verified local
PyTorch build is CPU-only. Install dependencies in your own environment when needed:

```powershell
# Install the declared PPO dependency; parser and geometry need only Python.
python -m pip install -r requirements.txt
# Train one shared model on both inputs with a fixed seed.
python steps/04-ppo-update/run.py --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --output steps/build/ppo-5-10-seed0 --updates 300 --episodes-per-update 16 --eval-every 25 --eval-episodes 32 --seed 0
# Open the maintained learning page and select 02, 03 or 04.
python steps/serve.py
```

The run writes `progress.json`, `rollout.json`, `initial-evaluation.json`,
`evaluation.json`, `replay-0.json`, `replay-1.json` and `checkpoint.pt` inside the
ignored output directory. Evaluation checkpoints also retain numbered model and
replay files; the page's checkpoint selector reads only snapshots actually saved.
While training is running the page refreshes its
statistics every five seconds; after completion use the refresh button.

Resume with `--resume <checkpoint>` and a new output directory, retaining the
same board and reward weight. `--updates` then means additional updates.
Try `--area-weight 0`, `0.5` or `1` in separate directories for reward ablations;
retain every run and seed. These controls compare reward preferences, while
formal SA-SP comparison still requires aligned rotation and area conventions.

To run the proposal's predeclared three-seed / three-weight experiment:

```powershell
# Use equal update and sampling budgets for all nine runs and retain failures.
python -m src.experiments.ablate --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --output steps/build/ppo-ablation-100 --seeds 0 1 2 --weights 0 0.5 1 --updates 100 --episodes-per-update 16 --eval-episodes 32 --workers 2
```

Each run keeps its command, log, checkpoint and initial/final evaluations;
`summary.json` contains every run, including nonzero exit codes. Training time
under concurrent execution should not be compared with an isolated inference
timing as a hardware speedup measurement.

The maintained chapter has 16 explanations. It distinguishes a sample's loss
contribution, the first minibatch, one Adam step and the complete PPO update.
The isolated teaching records show actual probabilities and values before/after
updating, both gradient norms and the exact clipping operands. Labeled hand
examples are separate from experiment measurements.

To fine-tune on the larger supplied circuits with declared canvases:

```powershell
python -m src.experiments.scale --input benchmarks/alarik-synthetic/blknet_20.txt benchmarks/alarik-synthetic/blknet_50.txt benchmarks/alarik-synthetic/blknet_100.txt --initialize-from steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/build/ppo-scale-20261019 --updates 100 --workers 2 --seed 0
```

`--initialize-from` loads model weights into a new experiment with a fresh Adam
optimizer and RNG stream; it permits a new canvas. `--resume` restores the same
experiment's optimizer, RNG, update history, canvas and reward convention.
The [larger-input report](../../docs/ppo-scaled-training.md) records the actual
budgets and results. Runtime records and models stay in ignored `steps/build/`.

Loading the checkpoint and evaluating a saved input is a separate process:

```powershell
# Independently load and validate the ten-macro layout.
python -m src.experiments.evaluate --input benchmarks/alarik-synthetic/blknet_10.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/build/ppo-5-10-seed0/loaded-replay-10.json
```
