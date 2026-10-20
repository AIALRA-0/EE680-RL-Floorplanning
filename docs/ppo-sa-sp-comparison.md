# How do frozen PPO and adapted SA-SP compare

Experiment date: 2026-10-20. Current recorded status: **completed**.

## Which conditions are shared

Both solvers use the same supplied synthetic circuit, original unrotated dimensions,
integer lower-left anchors, hard canvas, legal touching and no positive overlap.
Area is the tight occupied bounding-box area; each net contributes its lower-left
endpoint x-span plus y-span to HPWL. An independent validator checks completeness,
all board edges, every rectangle pair, rotation and integer anchors. No snapping
or post-hoc repair is applied. Area weight is 0.5 for every comparison trial.

```text
cost = alpha*Area/(W*H) + (1-alpha)*HPWL/max(1, M*(W+H))
reward = -cost for complete legal output; reward = -2 for invalid/incomplete output
```

| Macros | Board | Frozen PPO update | Training seed | Offline training seconds | Shared pretraining seconds |
|---|---|---:|---:|---:|---:|
| 5 | 64 x 64 | 300 | 0 | 348.577 | Included in shared 5/10 run |
| 10 | 64 x 64 | 300 | 0 | 348.577 | Included in shared 5/10 run |
| 20 | 96 x 96 | 100 | 0 | 148.503 | 348.577 |
| 50 | 176 x 176 | 100 | 0 | 963.491 | 348.577 |
| 100 | 256 x 256 | 100 | 0 | 1993.255 | 348.577 |

The 5- and 10-macro rows use the same joint training run; do not add its time twice.
Larger circuits each initialize from those weights, then fine-tune separately.
All models use training seed 0. Three PPO inference seeds represent action
randomness from each frozen checkpoint, not three independent training replicates.

## How was the original SA-SP reused

The received `src/sa_sp/seqpair_usc.py` stays unchanged. `native.py` exports actual
returned coordinates/dimensions and independently checks its metrics. The unchanged
5-macro seed-0 smoke run reproduces **Area 880, HPWL 84, 36000 moves**. Original
result images remain under `docs/sa-sp-results/alarik/`. Native rotation, unrestricted
canvas and initial-layout cost normalization differ from this comparison.

`solver.py` is a separate Python adaptation reusing the original sequence-pair
decoder and HPWL calculation. It disables rotation, uses a height-sorted first-fit
shelf initialization, proposes a swap in either sequence, rejects out-of-board
candidates and uses the shared normalized cost. Temperature probing uses the same
normalized units (maximum observed feasible absolute delta divided by log(2),
floor 1e-8). Search resets to the shelf state after probing. Metropolis accepts
improvements and accepts other feasible proposals when a uniform draw is below
exp(-delta/T). It retains the best feasible state.

Each SA seed uses 180 temperature stages, cooling factor 0.95 and max(20*N,200)
proposals per stage. Budgets are 36000/36000/72000/180000/360000 proposals for
5/10/20/50/100 macros. Each circuit has one PPO argmax run, sampled PPO seeds
0/1/2 and independent SA seeds 0/1/2. All recorded comparison inference/search is
CPU; PyTorch uses one thread.

## What did the actual trials produce

Quality means and sample standard deviations use complete legal outputs only.
Legal/total counts retain failed trials. A single argmax run has no sample SD.

| Macros | Algorithm | Legal/total | Area mean +/- SD | HPWL mean +/- SD | Cost mean +/- SD | Wall s mean | CPU s mean |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | ppo_argmax | 1/1 | 1221.000000 | 135.000000 | 0.254517 | 0.043280 | 0.031250 |
| 5 | ppo_sample | 3/3 | 1232.000000 +/- 0.000000 | 126.333333 +/- 25.813433 | 0.249089 +/- 0.020167 | 0.029717 | 0.031250 |
| 5 | sa_sp_aligned | 3/3 | 924.000000 +/- 0.000000 | 100.000000 +/- 0.000000 | 0.190918 +/- 0.000000 | 7.874437 | 7.515625 |
| 10 | ppo_argmax | 1/1 | 2744.000000 | 434.000000 | 0.504492 | 0.068408 | 0.062500 |
| 10 | ppo_sample | 3/3 | 2992.000000 +/- 241.016597 | 407.666667 +/- 12.662280 | 0.524479 +/- 0.027353 | 0.064443 | 0.062500 |
| 10 | sa_sp_aligned | 3/3 | 2600.000000 +/- 28.844410 | 268.000000 +/- 17.058722 | 0.422070 +/- 0.005724 | 11.883448 | 11.697917 |
| 20 | ppo_argmax | 1/1 | 5796.000000 | 1831.000000 | 0.552865 | 0.128600 | 0.125000 |
| 20 | ppo_sample | 3/3 | 6233.333333 +/- 348.794113 | 1680.000000 +/- 75.544689 | 0.556930 +/- 0.014654 | 0.140385 | 0.140625 |
| 20 | sa_sp_aligned | 3/3 | 5175.000000 +/- 154.696477 | 1318.666667 +/- 107.546889 | 0.452463 +/- 0.017849 | 38.734746 | 38.411458 |
| 50 | ppo_argmax | 1/1 | 19314.000000 | 5756.000000 | 0.475280 | 0.787998 | 0.781250 |
| 50 | ppo_sample | 3/3 | 22209.333333 +/- 746.108124 | 6104.333333 +/- 455.176156 | 0.531911 +/- 0.024795 | 0.796875 | 0.796875 |
| 50 | sa_sp_aligned | 3/3 | 16615.666667 +/- 248.347203 | 4009.333333 +/- 186.752064 | 0.382104 +/- 0.004818 | 211.041794 | 208.729167 |
| 100 | ppo_argmax | 1/1 | 34304.000000 | 17658.000000 | 0.434160 | 3.327664 | 3.078125 |
| 100 | ppo_sample | 3/3 | 41642.666667 +/- 1625.818358 | 18425.000000 +/- 245.220309 | 0.497640 +/- 0.014569 | 2.897705 | 2.880208 |
| 100 | sa_sp_aligned | 3/3 | 32829.000000 +/- 114.118360 | 12107.666667 +/- 319.626866 | 0.368704 +/- 0.002437 | 723.183943 | 717.197917 |

Lower cost is better. Compare Area and HPWL separately as well as the weighted
cost. The report includes every completed trial rather than selecting a winning
seed. Shipped per-trial JSON preserves circuit, protocol, placements, independent
validation, per-net HPWL and timings.

## What does timing include

PPO trial timing excludes checkpoint loading, one warm-up forward pass and offline
training. It includes policy/environment interaction, lightweight step tracing and
independent validation. SA includes initialization, normalized temperature probes,
search, trace capture and independent validation. JSON disk writes are excluded.
Model-load time is recorded per case. Wall and process CPU time are both reported.
Other local validation processes ran during this collection; this is a workstation
measurement, not an isolated hardware benchmark. Budgets are intentionally explicit
and different; equal-budget search and amortized training-cost studies remain open.

## How can the evidence be inspected or reproduced

```powershell
python steps/serve.py
python -m src.experiments.compare --output steps/build/comparison-reproduction --cases 5 10 20 50 100 --seeds 0 1 2 --stages 180
python -m unittest discover -s tests/python -p test_comparison.py -v
```

Open `/steps/reading.html` chapter 05 or `/steps/comparison.html`. The two layout
views share the same canvas scale. PPO exposes each executed placement. SA retains
its first 24 proposals, each newly best state and final proposal, with actual
iteration numbers, before/candidate/after coordinates, delta, probability and random
draw. Gaps are explicit; this trace does not contain every search proposal. The
final best layout is separate from the last accepted state.

Published evidence lives in `steps/evidence/comparison-20261020/`. Reproduction
uses local checkpoints when available, otherwise the published checkpoints. Use
a new output directory to preserve the immutable existing plan.

## What remains for team review

Review the shared protocol and adapted SA choices before treating them as settled
interfaces. Formal course benchmarks, unseen-circuit generalization, multiple
training seeds, ECE Linux reproduction, Python baseline/scoring review and optional CUDA remain
separate tasks. See [the concrete handoff](team-handoff.md).
