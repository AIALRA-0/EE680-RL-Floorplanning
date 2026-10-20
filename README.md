<div align="center">

<h1>EE680-RL-Floorplanning</h1>

<p>Private EE680 Choice #2 project: reinforcement learning for macro floorplanning.</p>

<p><strong>Status: CPU PPO trained on 5/10/20/50/100 macros; replay, shared-protocol SA-SP comparison and question-led learning included.</strong></p>

<p><a href="https://docs.google.com/document/d/1w6uCu5viYM8DziHWaoV3GLgr2abjgnJMrIQvF5JjPKA/edit">Team Hub</a> · <a href="https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240001">Live Tasks</a> · <a href="https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240002">Live Checklist</a> · <a href="https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240003">Live Gantt</a></p>

<p><a href="https://github.com/AIALRA-0/EE680-RL-Floorplanning/issues">Issues</a> · <a href="https://github.com/AIALRA-0/EE680-RL-Floorplanning/pulls">Pull Requests</a> · <a href="https://github.com/AIALRA-0/EE680-RL-Floorplanning/projects">Projects</a></p>

</div>

## 1 Planned project scope

We plan to place rectangular macros legally without overlap and optimize Area
and Half-Perimeter Wirelength (HPWL), using Proximal Policy Optimization (PPO).
We will compare against Simulated Annealing over Sequence Pairs (SA-SP). CUDA
acceleration is an optional later experiment after CPU profiling.

The repository contains a standard-library circuit parser, a placement environment,
a PyTorch actor/critic and PPO training loop, a maintained local learning page,
and Alarik's supplied Python SA-SP script. The five/ten-macro PPO experiment has
saved checkpoints, learning curves, actual rollout records and independently
validated layouts; separate fine-tuning runs cover 20/50/100 macros. See the
[PPO design](docs/ppo-design.md), [first-run results](docs/ppo-first-run.md) and
[larger-input results](docs/ppo-scaled-training.md),
[shared-protocol PPO/SA-SP comparison](docs/ppo-sa-sp-comparison.md) and
[team handoff](docs/team-handoff.md). **Shared C++ scoring, optional CUDA,
ECE-server reproduction and formal unseen-circuit evaluation remain pending.**

## 2 Current repository layout

<div align="center">

Table 2.1. Module ownership and intended responsibilities

| Directory | Owner | Planned responsibility |
|---|---|---|
| `src/data/` | Lucas | Circuit parsing and data representations |
| `src/environment/` | Lucas | Reinforcement learning placement environment |
| `src/ppo/` | Lucas | PPO networks and training |
| `src/experiments/` | Lucas | Training experiments and analysis |
| `cpp/geometry/` | Alarik | Placement legality, Area, and HPWL calculations |
| `cpp/sa_sp/` | Alarik | SA-SP baseline integration; received script is Python |
| `cpp/cuda/` | Alarik | Optional CUDA acceleration |
| `integration/bindings/` | Both | Python/C++ interoperability |
| `integration/evaluation/` | Both | Unified benchmark evaluation and comparison |
| `integration/visualization/` | Both | Floorplan visualization and comparison |
| `benchmarks/` | Both | Shared inputs; current samples are Alarik's synthetic circuits |
| `tests/fixtures/` | Both | Manually verified test circuits |
| `tests/python/` | Lucas | Python unit tests |
| `tests/cpp/` | Alarik | C++ unit tests |
| `tests/integration/` | Both | Cross-module and end-to-end tests |
| `docs/` | Both | Interface specifications and technical documentation |
| `steps/` | Both | Numbered learning steps and the maintained HTML learning page; shared code stays in its ownership directories |

</div>

`README.md` is the project entry point; the existing `.gitignore` is preserved.
The parser is in `src/data/parse_blknet.py`, with tests in `tests/python/`.
Alarik's script and original instructions are in `cpp/sa_sp/`, his five synthetic
inputs are in `benchmarks/alarik-synthetic/`, and his 18 received images are in
`docs/sa-sp-results/alarik/`. See the [import and first-run notes](docs/sa-sp-first-run.md)
for provenance, commands, observed results, and remaining baseline checks.
Empty directories retain `.gitkeep`; CUDA remains optional.

Agree on shared interfaces in `docs/` before adding implementation files. Connect
the independently developed Python and C++ modules through
`integration/bindings/`, with shared evaluation, visualization, and integration
tests. The Team Hub's proposed architecture guides future work; add files only
when actual work requires them and the relevant interfaces have been reviewed.

### 2.1 Run the circuit parser

From the repository root, use Python to inspect the five-block sample and run
the parser tests. These commands require no third-party packages.

```powershell
# Read the actual sample into blocks and nets; this does not create a placement.
python src/data/parse_blknet.py benchmarks/alarik-synthetic/blknet_5.txt
# Check the five received inputs and invalid-input behavior.
python -m unittest discover -s tests/python -p test_parse_blknet.py -v
```

The sample prints:

```text
5 blocks, 5 nets
B0 = (25.0, 8.0)
Net 0 = ['B4', 'B0', 'B2', 'B1']
```

### 2.2 Follow the numbered steps

Start with [00: circuit input](steps/00-input/README.md), then
[01: deterministic placement](steps/01-placement/README.md), then
[02: policy](steps/02-policy/README.md), [03: rollout](steps/03-rollout/README.md)
and [04: PPO updates](steps/04-ppo-update/README.md), followed by
[05: PPO/SA-SP comparison](steps/05-comparison/README.md). The
[step guide](steps/README.md) connects each explanation to actual code and saved
records. Lucas keeps the environment and PPO
in `src/`; Alarik keeps baseline and C++ work in `cpp/`; shared preview code is in
`integration/visualization/`.

```powershell
python steps/00-input/run.py
python steps/01-placement/run.py
python steps/serve.py
python -m unittest discover -s tests/python -p 'test_*.py' -v
```

Open the address printed by `steps/serve.py` to use the English question-led reader
in `steps/reading.html`, or double-click `steps/start.cmd`. Its six chapters connect
sixteen source-backed questions, worked explanations, exercises and embedded
visualizations. See the [LessonWeave workflow adaptation](docs/learning-workflow.md).
The retained interactive workbench in `steps/index.html` presents input,
source code, per-candidate boundary and overlap checks, action-mask entries,
selection, actual placement, scoring, reward, and the next observation. Every
rejected candidate remains inspectable; finding a legal anchor and executing it
are separate steps. Local progress resumes after restarting the service, and
chapter notes remain in the browser. Runtime records and CLI data exports stay
in ignored `build/` directories. Selected actual models, records and figures ship
in `steps/evidence/` for viewing after a fresh clone; large JSON uses lossless gzip.
The small inputs use a 64-by-64 canvas. Lower-left endpoints, no-rotation actions,
integer anchors and terminal reward references form the documented comparison
protocol, awaiting teammate review.
Chapters 02-04 show actual node/net tensors, masked action probabilities, value
estimates, training samples, losses, parameter changes and saved-layout replay.
The 01 demonstration retains its separate deterministic state and browser notes.

### 2.3 Train and load PPO

The parser, environment and server use the standard library. PPO requires
PyTorch 2.6.0; exporting plots uses Matplotlib 3.10.9 and the received SA-SP uses
NumPy 2.4.4. These are
declared in `requirements.txt`. The first verified run used Python
3.12.7 and PyTorch 2.6.0+cpu on Windows. From the repository root:

```powershell
# Train a shared policy on both received small inputs and retain all run evidence.
python -m src.experiments.train --input benchmarks/alarik-synthetic/blknet_5.txt benchmarks/alarik-synthetic/blknet_10.txt --output steps/build/ppo-5-10-seed0 --updates 300 --episodes-per-update 16 --eval-every 25 --eval-episodes 32 --seed 0
# Load the checkpoint in a separate process and independently validate its layout.
python -m src.experiments.evaluate --input benchmarks/alarik-synthetic/blknet_10.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/build/ppo-5-10-seed0/loaded-replay-10.json
# Check the parser, environment, masked policy, PPO update, checkpoint and server.
python -m unittest discover -s tests/python -p 'test_*.py' -v
```

New runtime JSON and models stay in ignored `build/` directories; curated records
are retained in `steps/evidence/`. For inference without retraining, use
`steps/evidence/ppo-5-10-seed0/checkpoint.pt` with the evaluation command above and
write the new output to a build directory. The
[training guide](steps/04-ppo-update/README.md) explains dependencies, continuation,
reward-weight controls and the maintained viewer. Larger circuits need explicit
canvas dimensions with sufficient space; the 64-by-64 small experiment is not a
universal canvas setting. The first comparison uses the five supplied synthetic
inputs; formal wildcard experiments remain pending.

## 3 Collaboration and first steps

The Team Hub is our authoritative shared guide and resource index. Google files
require team access. The Projects listing is a navigation entry, not a verified
link to a specific board.

- **Lucas:** Python/PyTorch, PPO environment, reward design, training, and analysis.
- **Alarik:** C++ geometry, SA-SP integration, CPU profiling, and optional CUDA.
- **Both:** shared interfaces, scoring correctness, benchmark validation,
  visualization, and final submission.

Read the Team Hub, claim a task, and agree on interfaces before writing code.
Both teammates may edit the Hub, add tasks, and adjust draft owners and dates.
Use **Live Tasks** to update progress, Start, DDL, notes, and evidence; Live
Checklist and Live Gantt derive from that sheet. Google Docs checkboxes and
GitHub Issues do not automatically update the sheet. Official course deadlines
remain controlled by the course.

Work on a dedicated branch and open a PR for teammate review before merging.
Use Discord for quick coordination and record lasting decisions in the Hub or
the corresponding issue. Both teammates can contribute to PPO after the shared
interfaces and basic environment are agreed. Build, training, and integration
commands will be documented as those components are implemented and verified.

## 4 Validation and private sharing

Current checks cover the parser, environment, recorded placement traces, detailed
search steps, observation encoding, legal sampling, hand-calculated GAE/PPO losses,
finite parameter updates, checkpoint continuation, independent Python scoring,
learning-page interactions, course references, compressed evidence, SA-SP export,
common objective and acceptance rules, Git diff and README links. The original
five-block SA-SP smoke run and adapted shared-protocol comparison retain separate
provenance. C++ parity, optional CUDA, formal benchmarks and ECE Linux checks
remain pending. Actual commands and results are recorded in the delivery report.

The README follows [github-readme-standardizer v1.0.0](https://github.com/AIALRA-0/github-readme-standardizer/releases/tag/v1.0.0)
with the explicit **English-only** requirement. Its unmodified strict auditor
reports `MISSING_README` for the intentionally absent `README.en.md` and
`PRIMARY_LANGUAGE_REVIEW` for this English README. These are documented language
exceptions. The repository scan flags `IMAGE_METADATA` on nine received
floorplan PNGs and eleven retained experiment plots. Metadata inspection found
Matplotlib software/version and DPI fields; PNG color metadata also occurs in
the received terminal images. The received images remain unchanged. The unmodified
strict audit still reports FAIL. See the first-run notes for the review details.

Keep course code private, including after the semester, and follow the
[official project policy](https://sites.usc.edu/ee680-lim/project/).
The [official Choice #2 requirements](https://sites.usc.edu/ee680-lim/project-choices-spring-2026/)
define the eventual algorithm and viewer deliverables. Cite sources and obtain
permission before reusing code. Alarik's received files remain private, with
upstream provenance and redistribution terms still to be documented. Included
checkpoints were trained locally on the supplied synthetic inputs for this private
team project. No open-source license is added.
