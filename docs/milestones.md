# What must each teammate deliver next

Updated October 20, 2026. These are team execution targets in America/Los_Angeles,
not changes to course deadlines. Both algorithms use Python. The original
C01-C25 IDs remain stable in the Team Hub and Live Tasks so existing notes and
links still identify the same work. C++ compilation and Python/C++ parity are
replaced by Python baseline verification; CUDA stays optional after profiling.

## What already exists

PR #4 delivered the first CPU PPO loop, original Python SA-SP import and aligned
comparison: 35 complete, independently legal trials across five supplied circuits.
Those circuits are synthetic. All PPO checkpoints use training seed 0; three
sampled inference seeds do not count as three independent training runs.
The 61-test release result and original native Area 880 / HPWL 84 are historical
evidence, not results of the future tasks below. Alarik's review, input provenance,
held-out evaluation and ECE reproduction remain open.

## What does Alarik own

Deliver four focused PRs. Start with A1; later work may proceed when its listed
dependencies pass. Every PR includes actual commands, observed results and
links to its outputs. Review is required before merge.

| Delivery | Target date | Checklist IDs | Dependency | Acceptance |
|---|---|---|---|---|
| A1: verify the Python baseline and scoring | October 23 | C04, C06-C08, C10-C11 | Current main and agreed protocol | Native reproduction, hand-verified fixtures and decoder/export tests pass |
| A2: deliver a configurable batch runner | October 26 internal checkpoint; C16 remains October 30 | C16 | A1 | One command completes a declared circuit/seed list, records failures and reproduces exported scores |
| A3: run baseline and search-budget experiments | October 30 | C16; baseline portion of C17 | A2 | All 45 declared runs are accounted for, with raw records, summary and plots |
| A4: deliver held-out evaluation and a CPU/ECE report | November 6 internal checkpoint; C17 remains November 13 | C14, C17, C22-C23 | A1-A2; profiling can start earlier | Frozen held-out manifest, independently checked results, reproducible CPU profile and ECE transcript or exact access blocker |

### What must A1 contain

- Confirm the script's original source, author and reuse terms in
  `docs/sa-sp-first-run.md`. Keep received code, instructions and images intact.
- Review `native.py`, `solver.py` and the common protocol with Lucas. Record the
  no-rotation, hard-board, integer-anchor, tight-Area and lower-left-HPWL decisions.
- Add hand-calculated fixture files under `tests/fixtures/` and tests under
  `tests/python/`. Cover touching edges, overlap, each board boundary,
  missing/unknown macros, a three-terminal net, translated layouts, rotation
  and fractional anchors. Reuse the existing evaluator; avoid a second
  production scoring implementation.
- Check the sequence-pair decoder on all pairs of permutations for two and
  three macros. Verify completeness, non-overlap and exported dimensions;
  compare scores against hand-derived expectations where applicable.
- Reproduce the unchanged five-macro seed-0 run: Area 880, HPWL 84, 36000 moves.
  Retain a layout JSON and actual test output. Changes to aligned search must
  not silently change this native reference.

### What must A2 contain

Implement `src/sa_sp/run.py` as the baseline-only CLI, reusing `solver.py` and
the shared evaluator. This filename is an assigned deliverable, not a file
already implemented by this architecture update.

Accept explicit input paths, canvas dimensions, seeds, Area weight, temperature
stages, cooling factor, proposals per stage and output directory. New circuits
must run without editing source code or using their filenames as configuration.
Reject invalid arguments. Preserve existing output directories and support
resuming unfinished trials without silently mixing settings.

Write one named-coordinate JSON per trial plus a CSV/JSON summary. Include input
SHA-256, protocol/settings, seed, source commit, Python/NumPy versions, machine
description, wall/CPU time, proposal/acceptance counts, best-cost history and
validation. Record invalid output, exceptions and interrupted trials explicitly;
never drop them from the denominator. Keep canonical exports compatible with
the existing comparison viewer and independently rescore every saved layout.

### What must A3 contain

Before running, freeze the manifest and settings. Use the five supplied sizes
5/10/20/50/100 with their existing boards 64/64/96/176/256 square units.
Use seeds 0-4, Area weight 0.5, cooling 0.95 and max(20*N, 200) proposals per stage.

- Run all five circuits at 180 stages: 25 runs.
- On 20 and 50 macros, also run 60 and 120 stages with the same five seeds:
  20 additional runs. Reuse their 180-stage rows from the first set.
- Report all 45 outcomes, legal completion rate, individual scores, mean/sample
  standard deviation for complete legal Area/HPWL/cost, runtime and acceptance.
- Plot cost against proposal count and wall time, plus representative same-scale
  layouts and convergence histories. State timer boundaries and hardware.
- Write `docs/sa-sp-experiments.md`, with the manifest, commands, raw-output
  locations and interpretation. Report regressions. Compare against the frozen
  published PPO evidence only under matching protocol conditions and label
  offline PPO training separately. Do not claim equal compute budgets.

### What must A4 contain

- Prepare ten reproducible held-out synthetic circuits, two at each existing
  size, with a generator seed, input hash and a known feasible construction.
  Freeze their manifest before either teammate tunes on them. Existing five
  circuits remain development inputs. If formal inputs are available, verify
  permission and format before import; synthetic tests do not close formal
  benchmark acceptance.
- Run the agreed baseline on every held-out input and independently validate
  all exports. Hand Lucas the same manifest and frozen baseline settings for
  PPO evaluation. Keep tuning results separate from the held-out test set.
- Profile decoding, HPWL and validation on 20/50/100 macros. Provide cumulative
  CPU profiles and counts. Implement one justified Python/NumPy improvement in
  the adapted solver if profiling identifies a worthwhile hotspot; otherwise
  document the measured result. For an optimization, compare matched seeds and
  budgets before/after, validate coordinates/scores and report timing including
  regressions. Preserve the unmodified native script.
- By October 23, report whether ECE access works. Reproduce baseline tests and
  a saved-layout run on ECE Linux, recording versions and commands. If access
  fails, give the actual error and required owner action; keep local profiling
  and experiments moving. Do not mark C23 complete until both solvers reproduce.
- Deliver `docs/sa-sp-performance.md` and held-out results for the joint C17
  report. C++/CUDA work needs a separate decision supported by these profiles.

## What does Lucas own in parallel

Lucas reviews the protocol and fixtures for A1, maintains PPO/environment code,
and completes independent-training-seed replication and reward ablations (C18).
He evaluates frozen PPO on Alarik's held-out manifest without retraining on that
test set, records training separately from inference, reproduces PPO on ECE and
integrates the joint plots, viewer, report and demo with Alarik (C17, C19, C23-C24).

## How do the original milestone IDs change

| IDs | Revised meaning | Completion condition |
|---|---|---|
| C04 | Verify source/input provenance | Exact source and usable permissions recorded |
| C06-C07 | Agree Python contract and hand fixtures | Both reviewers reproduce expected values |
| C08 | Verify and maintain independent Python scoring | Added geometry/HPWL fixtures pass; no C++ build required |
| C09 | Python deterministic environment | Existing implementation; review its saved evidence |
| C10 | Reproduce the received Python SA-SP | Native coordinates and 880/84 reproduction verified |
| C11 | Verify Python solver/export/evaluator consistency | Shared fixtures and decoder/export checks pass |
| C12-C13, C15 | PPO environment, learning and small-input training | Existing synthetic evidence; formal-input acceptance still separate |
| C14 | Audit Python/NumPy/PyTorch on ECE and profile baseline CPU | Actual environment and profile evidence |
| C16 | Maintain Python SA-SP runner and seed/budget study | A2-A3 accepted |
| C17 | Joint development and held-out comparison | Same contract, raw outputs, failures and timing; formal inputs when permitted |
| C18-C19 | PPO ablations and joint visualization | Independent training seeds and required visuals accepted |
| C20 | Optional acceleration after profiling | Choose measured Python/NumPy optimization first; consider C++/CUDA only if justified |
| C22-C23 | Unseen inputs and ECE reproduction | Both solvers run from a clean checkout without per-circuit source edits |

C01-C03, C05, C21 and C24-C25 retain their existing scope. Team dates October 23,
October 30, November 13/20/27 and December 3/9/10 remain as recorded unless the
team changes them explicitly. No future task is marked complete by this document.
