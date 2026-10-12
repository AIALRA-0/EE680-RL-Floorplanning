<div align="center">

<h1>EE680-RL-Floorplanning</h1>

<p>Private EE680 Choice #2 project: reinforcement learning for macro floorplanning.</p>

<p><strong>Status: minimal repository scaffold; algorithm implementation has not started.</strong></p>

<p><a href="https://docs.google.com/document/d/1w6uCu5viYM8DziHWaoV3GLgr2abjgnJMrIQvF5JjPKA/edit">Team Hub</a> · <a href="https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240001">Live Tasks</a> · <a href="https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240002">Live Checklist</a> · <a href="https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240003">Live Gantt</a></p>

<p><a href="https://github.com/AIALRA-0/EE680-RL-Floorplanning/issues">Issues</a> · <a href="https://github.com/AIALRA-0/EE680-RL-Floorplanning/pulls">Pull Requests</a> · <a href="https://github.com/AIALRA-0/EE680-RL-Floorplanning/projects">Projects</a></p>

</div>

## 1 Planned project scope

We plan to place rectangular macros legally without overlap and optimize Area
and Half-Perimeter Wirelength (HPWL), using Proximal Policy Optimization (PPO).
We will compare against Simulated Annealing over Sequence Pairs (SA-SP). CUDA
acceleration is an optional later experiment after CPU profiling.

PPO, SA-SP, CUDA, geometry/scoring, and benchmark integration are **not implemented
or tested**. This scaffold contains no algorithm source, package configuration,
dependencies, runnable solver, or algorithm tests.

## 2 Current repository layout

<div align="center">

Table 2.1. Directory scaffold; each listed leaf contains only a `.gitkeep`

| Directory | Owner | Planned responsibility |
|---|---|---|
| `src/data/` | Lucas | Circuit parsing and data representations |
| `src/environment/` | Lucas | Reinforcement learning placement environment |
| `src/ppo/` | Lucas | PPO networks and training |
| `src/experiments/` | Lucas | Training experiments and analysis |
| `cpp/geometry/` | Alarik | Placement legality, Area, and HPWL calculations |
| `cpp/sa_sp/` | Alarik | SA-SP baseline integration |
| `cpp/cuda/` | Alarik | Optional CUDA acceleration |
| `integration/bindings/` | Both | Python/C++ interoperability |
| `integration/evaluation/` | Both | Unified benchmark evaluation and comparison |
| `integration/visualization/` | Both | Floorplan visualization and comparison |
| `benchmarks/` | Both | Shared benchmark inputs, subject to permission checks |
| `tests/fixtures/` | Both | Manually verified test circuits |
| `tests/python/` | Lucas | Python unit tests |
| `tests/cpp/` | Alarik | C++ unit tests |
| `tests/integration/` | Both | Cross-module and end-to-end tests |
| `docs/` | Both | Interface specifications and technical documentation |

</div>

`README.md` is the project entry point; the existing `.gitignore` is preserved.
This hierarchy establishes ownership boundaries; the directories contain no
implementations, tests, datasets, or interface specifications yet. Parent
directories are tracked through their children. **Algorithm implementations have
not started**, and CUDA remains optional.

Agree on shared interfaces in `docs/` before adding implementation files. Connect
the independently developed Python and C++ modules through
`integration/bindings/`, with shared evaluation, visualization, and integration
tests. The Team Hub's proposed architecture guides future work; add files only
when actual work requires them and the relevant interfaces have been reviewed.

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
the corresponding issue. Environment setup, build/run commands, and executable
tests will be documented when those components exist.

## 4 Validation and private sharing

Initialization checks cover the actual Git diff, directory contents, README
links, rendering, and privacy. There are no runnable algorithm, benchmark, C++,
CUDA, or ECE Linux checks at this stage.

The README follows [github-readme-standardizer v1.0.0](https://github.com/AIALRA-0/github-readme-standardizer/releases/tag/v1.0.0)
with the explicit **English-only** requirement. Its unmodified strict auditor
reports `MISSING_README` for the intentionally absent `README.en.md` and
`PRIMARY_LANGUAGE_REVIEW` for this English README. These are documented language
exceptions, not a strict audit PASS; other findings still require correction.

Keep course code private, including after the semester, and follow the
[official project policy](https://sites.usc.edu/ee680-lim/project/).
The [official Choice #2 requirements](https://sites.usc.edu/ee680-lim/project-choices-spring-2026/)
define the eventual algorithm and viewer deliverables. Cite sources and obtain
permission before reusing code. No third-party source, benchmark archives, model
checkpoints, or open-source license are included in this scaffold.
