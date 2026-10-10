<div align="center">

<h1>EE680-RL-Floorplanning</h1>

<p>Private EE680 Choice #2 project: reinforcement learning for macro floorplanning.</p>

<p><strong>Status: minimal repository scaffold; algorithm implementation has not started.</strong></p>

</div>

[Team Hub](https://docs.google.com/document/d/1w6uCu5viYM8DziHWaoV3GLgr2abjgnJMrIQvF5JjPKA/edit) · [Live Tasks](https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240001) · [Live Checklist](https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240002) · [Live Gantt](https://docs.google.com/spreadsheets/d/1kIy4pi5xVvRvsZ2svOghlOr8vObGK5ZaGIsXHfsiezA/edit#gid=817240003)

[Issues](https://github.com/AIALRA-0/EE680-RL-Floorplanning/issues) · [Pull requests](https://github.com/AIALRA-0/EE680-RL-Floorplanning/pulls) · [Projects listing](https://github.com/AIALRA-0/EE680-RL-Floorplanning/projects)

The Team Hub is our authoritative shared guide and resource index. Google files
require team access. The Projects listing is a navigation entry, not a verified
link to a specific board.

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

Table 2.1. Initial directories; each tracks only a `.gitkeep`

| Directory | Intended future responsibility |
|---|---|
| `src/` | Python data, environment, PPO, scoring, and experiment code |
| `cpp/` | C++ geometry/scoring and optional accelerated implementation |
| `tests/` | Independently verified correctness tests and synthetic fixtures |
| `docs/` | Agreed architecture, benchmark notes, and reproducibility evidence |

</div>

`README.md` is the project entry point; the existing `.gitignore` is preserved.
The Team Hub's detailed file tree is a **future architecture**, not a list of
files that must exist during initialization. Add files only when actual work
requires them and the relevant interfaces have been reviewed.

## 3 Collaboration and first steps

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
