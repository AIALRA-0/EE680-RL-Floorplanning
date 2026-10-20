# How do the Python solvers fit together

Updated October 20, 2026. Both PPO and SA-SP currently run in Python. A C++
rewrite is not a project dependency. Optional acceleration needs CPU profiling
and a measured benefit before it becomes a task.

## Which files should each teammate change

| Location | Lead owner | Responsibility |
|---|---|---|
| `src/data/parse_blknet.py` | Lucas; Alarik reviews new inputs | Parse ordered macro dimensions and hypernets |
| `src/environment/`, `src/ppo/` | Lucas | Legal actions, observations, actor/critic, rollouts and PPO updates |
| `src/sa_sp/seqpair_usc.py`, `HowToRun.txt` | Alarik | Preserve the received Python solver and instructions unchanged |
| `src/sa_sp/native.py` | Alarik | Load the received solver, export actual coordinates and reproduce native metrics |
| `src/sa_sp/solver.py` | Alarik | Maintain the separately adapted no-rotation, bounded-canvas SA-SP search |
| `src/experiments/compare.py` | Both | Orchestrate frozen PPO and SA-SP trials using the shared protocol |
| `integration/evaluation/` | Alarik; Lucas reviews | Independently validate completeness, legality, Area and per-net HPWL |
| `integration/visualization/`, `steps/` | Both | Render saved coordinates, actual decisions, scores and learning explanations |
| `benchmarks/`, `tests/fixtures/` | Alarik; both approve | Document input rights, generate held-out circuits and hand-check expected scores |
| `tests/python/`, `tests/integration/` | Module owner | Test real solver behavior, scoring and cross-module exports |
| `docs/` | Both | Keep interfaces, experiment conditions, milestones and actual results current |

The former `cpp/sa_sp/adapter.py` is now `src/sa_sp/native.py`; the former
`cpp/sa_sp/aligned.py` is now `src/sa_sp/solver.py`. The received filenames remain
unchanged for provenance. Imports and live teaching source links use the new
paths. The original release report, file manifest and recorded experiment JSON
retain their historical paths; those strings describe the run that produced them.
The unused `cpp/geometry/`, `cpp/cuda/`, `tests/cpp/` and
`integration/bindings/` markers were removed. No functional C++ code existed there.

## How does a layout reach the evaluator and viewer

1. The parser returns `blocks` keyed by macro ID and a list of hypernets.
2. PPO uses the placement environment; SA-SP decodes its two permutations.
   Neither algorithm imports the other algorithm's search or policy.
3. Each solver exports named lower-left coordinates. The native SA-SP wrapper
   also retains actual orientation; the aligned comparison rejects rotation.
4. `integration/evaluation/protocol.py` scores both exports independently.
   The aligned contract uses integer anchors, a hard board, legal touching,
   tight occupied bounding-box Area and lower-left-endpoint HPWL.
5. Experiment orchestration records seeds, settings, failures and timing. The
   viewer reads these records rather than calculating a different objective.

Alarik owns baseline correctness and experimental evidence, not only running the
original script. Lucas owns PPO training and its generalization study. Shared
scoring changes require review by both because they change every reported result.
The [milestones](milestones.md) define Alarik's next concrete deliveries.

## Which content must remain separate

The native solver permits rotation and has its own cost normalization. Its
880/84 five-macro smoke result is preserved separately from aligned comparisons.
Changing its rules would destroy the reference experiment, so baseline
improvements belong in `solver.py` and must retain an unchanged comparison run.

Keep new runtime records in ignored `steps/build/` directories. Publish selected
English evidence only after checking its inputs, provenance and scores. The local
Chinese learning edition stays in `steps/build/local-zh/`. No new runtime
dependency, benchmark download or training result is introduced by this migration.
