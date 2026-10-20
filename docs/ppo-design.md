# How the first PPO implementation follows the project milestones

## Which materials determine the scope

Reviewed on October 19, 2026:

- [Course Choice #2](https://sites.usc.edu/ee680-lim/project-choices-spring-2026/): sequential grid placement, policy/value networks, rewards/advantages, PPO, Area/HPWL/runtime comparisons, learning curves and basic ablations. The viewer must show lower-left endpoint connections, including every pair within a hypernet, and placement progression.
- [Course policy](https://sites.usc.edu/ee680-lim/project/): private code, wildcard inputs, ECE-server reproduction, self-contained dependencies, PPTX report and oral defense. The page currently lists October 23 as the grace-period end, December 10 at 6:30 PM for submission, and December 14-15 for oral defense. Its submission sections mention both Brightspace and Canvas; confirm the actual portal when submitting.
- [Team Hub](https://docs.google.com/document/d/1w6uCu5viYM8DziHWaoV3GLgr2abjgnJMrIQvF5JjPKA/edit) and [Live Tasks](https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit): C12/C13 due October 23; C15 due October 30; C17 comparisons due November 13; C18 ablations due November 20; C19 viewer due November 27. These are team planning dates.
- The supplied `ee680.md` history: milestone 1 is a verifiable legal layout; implement core PPO explicitly in PyTorch; preserve the numbered learning sequence and source ownership.
- The two-page *EE680_Proposal_Lucas_Alarik.pdf*, Fall 2026, Proposal v12: October 17-30 targets the legal environment and end-to-end small-circuit PPO; the stability experiment requires at least three seeds and three predeclared reward settings. It also commits to independent scoring, preserving original SA-SP coordinates, replay at checkpoints, later wildcard tests and ECE reproduction. The downstream-label selection pilot is optional.
- The bootstrap archive: preserve private history and document real evidence. Its initialization-only scope and older proposed package layout have been superseded by the user's later implementation request and current directory structure.
- Professor Lim's 27-page *A Graph Placement Methodology for Fast Chip Design*: geometry, connectivity, current macro, legal masks, terminal reward, actor/critic and PPO; the full Google model also includes components the course explicitly permits students to omit.

The October 20 noon target brings C12/C13 and the first C15-style experiment forward. The user selected 5/10-macro training first, followed by larger inputs. Checklist updates and teammate interface approval remain separate actions; this document does not mark shared tasks closed.

## What is the state and action contract

The existing `Circuit` has ordered string IDs mapped to `(width, height)` and a list of hypernets. An observation contains that circuit, integer canvas dimensions, placements, current macro, remaining IDs and area weight. Every episode starts empty. Macros follow input order; actions do not rotate them.

Action `i = y * W + x` proposes an integer lower-left anchor. `PlacementEnv` checks the complete rectangle, permits touching edges and rejects positive-area overlap. The faster full mask marks the exact integer intervals excluded by the same inequalities; scalar `inspect_action` remains available for detailed teaching and parity tests.

Area is the tight occupied bounding-box area. HPWL sums each net's horizontal and vertical spans using lower-left endpoints. Successful terminal reward is `-alpha * Area/(W*H) - (1-alpha) * HPWL/max(1, M*(W+H))`. Intermediate reward is zero; an episode with no legal continuation ends at -2. Default alpha is 0.5; configurable alpha supports weight ablations. These Python contracts await joint C++ parity review. The received native SA-SP permits rotation and uses origin-anchored bounds. The separate aligned adapter disables rotation and applies the shared canvas and objective; see the [comparison report](ppo-sa-sp-comparison.md).

## How does the model see dimensions, positions and nets

`src/ppo/encoding.py` creates eight node features: normalized width, height, area, x, y, placed flag, current flag and incident-net count. Unplaced x/y are zero with `placed=0`, so they differ from a macro placed at the origin. A separate binary net-by-node incidence tensor preserves hypernets, without relying on ID spelling or a fixed macro count.

The encoder averages learned node embeddings into each net, then averages incident net messages into each node. It pools valid nodes and selects the current node embedding. Four global features describe progress, total macro area, board aspect and reward preference. Graph batching pads nodes/nets and masks padding during pooling.

For each anchor, six deterministic features describe x/y, prospective bounding-box area/width/height and incremental partial HPWL. They use only already placed endpoints and the current macro; unplaced coordinates are never invented. This engineering choice supplies geometric information directly to a small CPU model. Both candidate features and their learned scoring weights are part of the documented architecture.

The actor learns a context query and candidate embeddings and scores their dot products plus a learned feature projection. Its logits have no fixed heuristic bias. Illegal logits become negative infinity before `Categorical`; the actor samples during training and uses argmax during deterministic inference. The critic predicts one scalar from the shared context. An all-false mask ends the environment rather than constructing an invalid distribution.

For the first CPU version, occupied space is represented by exact placed rectangles and placement flags, with the resulting legal grid mask. It does not add a separate occupancy-image CNN. The maintained HTML provides the requested detailed learning view; the proposal's final side-by-side Matplotlib comparison is a later deliverable.

## How are samples used to update the model

`src/ppo/algorithm.py` collects complete on-policy episodes. Each transition retains its original encoded state/mask, action, old log probability, old value, reward and termination/boundary flags. GAE uses `delta = r + gamma * V(next) * (1-terminated) - V(current)` and stops recursion at episode boundaries. The complete-episode collector has no time-limit truncation; the tested GAE function separately supports time-limit bootstrap.

Returns are computed before advantage normalization. Updates use the minimum of unclipped and clipped probability-ratio objectives, critic squared error and an entropy bonus. Adam applies gradient clipping and checks finite loss/gradient norms. Approximate KL can stop an update early. Defaults are recorded in `PPOConfig` and every run's JSON, including gamma=1, lambda=0.95, clip=0.2, four epochs and minibatch size 32.

Checkpoints retain network/optimizer states, configuration, RNG and run metadata. `--resume` restores them. Evaluation uses a separate fixed RNG stream so viewing evaluation does not change subsequent training samples. Saved trajectories are independently rescored by `integration/evaluation/placement.py`; disagreement on complete layouts is an error.

For larger-input fine-tuning, `--initialize-from` reuses only model weights and
creates a fresh optimizer/RNG/update count, permitting a new canvas. The larger
runs explicitly override GAE lambda to 1. `src/experiments/study.py` loads each
saved checkpoint into a separate teaching experiment and records one inspectable
update without overwriting official models. It also repeats initial/final sampled
evaluation. The [larger-input report](ppo-scaled-training.md) records budgets,
checkpoint provenance and results.

## What must the first experiment record

Train on the two existing 5/10-macro inputs, retain initial evaluation, every update's training statistics, fixed-seed sampled evaluation, deterministic replay and checkpoint. Count failed episodes in completion rates; report conditional Area/HPWL only for complete legal results. Training wall time includes training-time evaluation and saves; per-layout inference time excludes model loading and disk output but includes trace construction when enabled. Fixed-instance training is the first experiment; wildcard evaluation and additional seeds are separate experiments.

Write new runtime evidence and models to ignored `steps/build/`. Selected actual records are published under `steps/evidence/` for fresh-clone replay. Keep reusable implementation in Lucas's `src/` and shared `integration/`; keep numbered explanations and entry points in `steps/`. The question-led reader embeds and preserves the 00/01 walkthrough and detailed 02-04 PPO visualizations.

## Which sources define the algorithm

- [PPO original paper](https://arxiv.org/abs/1707.06347): clipped surrogate objective and repeated on-policy minibatch updates.
- [GAE original paper](https://arxiv.org/abs/1506.02438): advantage estimation.
- [Invalid action masking study](https://arxiv.org/abs/2006.14171): masked policy distributions and gradients.
- [PyTorch 2.6 distributions](https://docs.pytorch.org/docs/2.6/distributions.html): `Categorical`, sampling, log probabilities and entropy.

The implementation is written for this repository; no reference solver or PPO library source was copied into these modules.
