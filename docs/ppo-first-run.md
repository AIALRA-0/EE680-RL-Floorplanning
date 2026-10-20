# What the first small-circuit PPO experiment produced

## What was executed

Date: October 19, 2026. Local branch: `feat/ppo-training-loop`.
Python 3.12.7, PyTorch 2.6.0+cpu, Windows, one PyTorch CPU thread.
Inputs: existing `benchmarks/alarik-synthetic/blknet_5.txt` and `blknet_10.txt`.
No third-party dataset or source was downloaded for this experiment.

```powershell
# Train on both inputs with the same policy and retain the full update history.
python -m src.experiments.train --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --output steps/build/ppo-5-10-seed0 --updates 300 --episodes-per-update 16 --eval-every 25 --eval-episodes 32
# Load the saved checkpoint in separate processes and score each result independently.
python -m src.experiments.evaluate --input benchmarks/alarik-synthetic/blknet_5.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/build/ppo-5-10-seed0/loaded-replay-5.json
python -m src.experiments.evaluate --input benchmarks/alarik-synthetic/blknet_10.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/build/ppo-5-10-seed0/loaded-replay-10.json
```

The main run completed 300 PPO updates and 4,800 sampled training episodes in
348.577 seconds, including periodic evaluation and saving. At update 300 the
parameter-change L2 norm was 0.305727, policy loss -0.000815, value loss 0.051300
and approximate KL 0.000129. Training statistics varied by update and are retained
in full; the final sampled training batch completed 15/16 episodes.

## How did the fixed sampled evaluations change

Each entry uses 32 stochastic evaluation episodes per circuit and the same
evaluation RNG stream before and after training. Area/HPWL means below are
conditional on complete legal layouts; mean reward includes failures at -2.

| Input | Stage | Complete legal episodes | Mean reward | Mean Area, complete only | Mean HPWL, complete only |
|---|---|---:|---:|---:|---:|
| 5 macros | Initial model | 32/32 | -0.555882 | 2764.781 | 279.531 |
| 5 macros | Update 300 | 32/32 | -0.249372 | 1234.656 | 126.281 |
| 10 macros | Initial model | 7/32 | -1.712086 | 3862.000 | 543.714 |
| 10 macros | Update 300 | 29/32 | -0.664491 | 2988.586 | 413.483 |

The ten-macro sampled evaluation reached 32/32 at some earlier checkpoints and
fell back to 29/32 at update 300. The final result is reported as observed;
successful prefixes can still block later macros even though every selected
action satisfies its current legal mask.

## What did the independently loaded model generate

These deterministic argmax results were reproduced after loading the checkpoint
in separate processes. The independent scorer agreed with the environment on
legality, occupied bounding-box Area and lower-left endpoint HPWL.

| Input | Complete legal layout | Area | HPWL |
|---|---|---:|---:|
| 5 macros | Yes | 1221 | 135 |
| 10 macros | Yes | 2744 | 434 |

Inference includes encoding, mask construction, network calls, environment
transitions and detailed trace construction. It excludes checkpoint loading and
writing the result file. Actual per-run timings are retained in the JSON rather
than treated as a stable hardware benchmark.

## Which verification commands passed

```powershell
# Run the actual automated tests.
python -m unittest discover -s tests/python -p 'test_*.py' -v
# Compile all maintained Python source and tests without executing experiments.
python -m compileall -q src integration steps tests
# Export real samples without changing model parameters.
python steps/03-rollout/run.py --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/03-rollout/build/rollout.json --episodes 4 --seed 10000
# Resume a two-update smoke checkpoint for one additional update.
python -m src.experiments.train --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --output steps/build/ppo-resume-check --updates 1 --episodes-per-update 4 --eval-every 1 --eval-episodes 2 --resume steps/build/ppo-smoke/checkpoint.pt --no-preview
```

Observed: 46 tests passed, syntax checks passed, the sample export retained 30
transitions from four episodes, and the resumed run completed update 3. A separate
test restored RNG and optimizer state and reproduced the next update's parameters
exactly. Geometry-mask parity includes fractional-size cases. GAE tests cover
true termination and time-limit bootstrap; PPO clipping tests include positive
and negative advantages.

Browser checks exercised 02/03/04, source excerpts, circuit selection, action
tables, replay and training curves at widths 1440, 736 and 320 in both light and
dark modes: six views passed without JavaScript errors or page-wide overflow.
Reading these chapters preserved the separate 01 environment exactly. The 01
regression check also passed: 40 recorded states, 195 explanations, 451 actual
source references, and six theme/width views. Review scripts and screenshots are
local QA artifacts, while the maintained viewer and tests are repository source.

## What did the three-seed reward ablation show

The predeclared grid used seeds 0/1/2 and area weights 0/0.5/1. Each of the nine
runs used 100 updates, 16 training episodes per update and 32 final sampled
evaluation episodes per input. All nine processes exited with code 0. Their
`plan.json`, commands, logs, checkpoints and complete results remain under
`steps/build/ppo-ablation-100/`. This is a separate, equal-budget experiment;
its 100-update numbers are distinct from the main 300-update run.

Each value below is the mean of three seed-level evaluation means, plus/minus
the sample standard deviation across seeds. Within each seed, Area/HPWL include
only complete legal layouts. Completion rates count all 32 episodes per seed.

| Macros | Area weight | Legal rate, % | Area, input units squared | HPWL, input units |
|---:|---:|---:|---:|---:|
| 5 | 0 | 100.00 +/- 0.00 | 1823.20 +/- 196.23 | 213.94 +/- 37.34 |
| 5 | 0.5 | 100.00 +/- 0.00 | 1411.58 +/- 148.15 | 182.40 +/- 22.95 |
| 5 | 1 | 100.00 +/- 0.00 | 1269.23 +/- 51.00 | 211.32 +/- 6.99 |
| 10 | 0 | 85.42 +/- 4.77 | 3556.91 +/- 49.04 | 494.44 +/- 53.86 |
| 10 | 0.5 | 95.83 +/- 7.22 | 3252.57 +/- 190.62 | 495.83 +/- 48.74 |
| 10 | 1 | 97.92 +/- 1.80 | 3193.19 +/- 161.73 | 508.85 +/- 16.28 |

Higher area weight reduced the observed mean Area in this short sweep. For the
five-macro input, weight 0.5 achieved lower observed HPWL than either extreme;
for ten macros, HPWL differences were small relative to seed variation. Reward
values use different objectives across weights, so comparisons use the explicit
Area, HPWL and completion columns.

```powershell
# Execute the predeclared grid without replacing the viewer's main run.
python -m src.experiments.ablate --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --output steps/build/ppo-ablation-100 --seeds 0 1 2 --weights 0 0.5 1 --updates 100 --episodes-per-update 16 --eval-episodes 32 --workers 2
# Export actual single-run curves, across-seed curves and labeled PPO floorplans.
python -m src.experiments.plot --run steps/build/ppo-5-10-seed0 --ablation steps/build/ppo-ablation-100 --output steps/build/ppo-figures
```

The plotting command succeeded with Matplotlib 3.10.9 and produced three PNGs.
The ablation plot shows mean and sample standard deviation across the three
seeds; the main-run plot explicitly identifies n=1. Labels, axes and the two
equally scaled PPO floorplans were visually inspected. The separate PPO/SA-SP
comparison viewer still awaits aligned baseline output.

## What happened on the larger supplied inputs

The same 5/10-trained checkpoint was loaded without retraining or source edits.
Canvas dimensions were supplied through `--width`/`--height`. The first tests
used 80/128/192-square canvases for 20/50/100 macros. After the 20/50 failures,
the two failed cases were also tested on larger canvases. All results are retained.

| Macros | Canvas | Placed | Complete and legal | Complete Area | Complete HPWL |
|---:|---|---:|---|---:|---:|
| 20 | 80 x 80 | 17/20 | No | - | - |
| 20 | 96 x 96 | 20/20 | Yes | 5642 | 1666 |
| 50 | 128 x 128 | 41/50 | No | - | - |
| 50 | 176 x 176 | 50/50 | Yes | 19436 | 5825 |
| 100 | 192 x 192 | 100/100 | Yes | 35150 | 16951 |

For example, the successful twenty-macro command was:

```powershell
# Generate a compact independent evaluation record for an input absent from training.
python -m src.experiments.evaluate --input benchmarks/alarik-synthetic/blknet_20.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --width 96 --height 96 --summary-only --output steps/build/ppo-5-10-seed0/unseen-20-board96.json
```

The failed cases returned exit code 2 and retained their failure reason and partial
placements. Successful complete layouts passed independent scoring. These are
single deterministic evaluations of additional supplied circuits; dedicated
large-circuit training, repeated evaluation and formal wildcard tests remain.

## Which files changed in this implementation phase

Existing staged parser, SA-SP imports and earlier learning work were preserved.
This phase added the following functional files:

```text
docs/ppo-design.md
docs/ppo-first-run.md
integration/evaluation/placement.py
integration/visualization/ppo_data.py
requirements.txt
src/ppo/encoding.py
src/ppo/model.py
src/ppo/algorithm.py
src/experiments/train.py
src/experiments/evaluate.py
src/experiments/ablate.py
src/experiments/plot.py
steps/02-policy/README.md
steps/02-policy/run.py
steps/03-rollout/README.md
steps/03-rollout/run.py
steps/04-ppo-update/README.md
steps/04-ppo-update/run.py
steps/ppo.js
tests/python/test_ppo.py
```

Modified: `README.md`, `src/environment/placement_env.py`,
`steps/01-placement/lesson.js`, `steps/README.md`, `steps/index.html`,
`steps/learning.css`, `steps/learning.js`, `steps/serve.py` and
`tests/python/test_learning_server.py`. Removed only redundant empty markers:
`src/ppo/.gitkeep`, `src/experiments/.gitkeep` and
`integration/evaluation/.gitkeep`.

`git diff --check` and the existing import-compatible
`git -c core.whitespace=cr-at-eol diff --cached --check` passed. The unmodified
v1.0.0 README audit still reports its documented English-only language findings
and nine pre-existing received-image metadata findings at that phase. The later
delivery publishes selected actual JSON, checkpoints and plots under `steps/evidence/`;
the [delivery report](delivery-report.md) records its expanded validation results.

## What remains before team acceptance

C12/C13 now have code, numerical tests, actual updates and replayable trajectories.
C15 has a reproducible first experiment and a checkpoint producing legal layouts
on the two supplied small inputs. Teammate review and shared-checklist acceptance
have not been recorded by these local runs. Subsequent work completed
[larger-circuit fine-tuning](ppo-scaled-training.md) and the first
[aligned SA-SP comparison](ppo-sa-sp-comparison.md). Python baseline/scoring review, formal
wildcard experiments and ECE-server reproduction remain separate milestones.
The [team handoff](team-handoff.md) identifies review items eligible for closure;
this repository does not automatically update the shared checklist.
