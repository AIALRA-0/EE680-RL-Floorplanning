# What should each teammate do next

PR #4 is merged. It delivered CPU PPO training/replay and the first Python SA-SP
comparison. Current paths and ownership are in [architecture](architecture.md);
the deliverables and acceptance criteria are in [milestones](milestones.md).

## What should Alarik deliver

1. **By October 23: baseline correctness PR.** Review the shared scoring rules,
   reproduce native Area 880 / HPWL 84, add hand-checked scoring fixtures and
   sequence-pair decoder/export tests. Report ECE access and source permissions.
2. **By October 26: Python runner PR.** Deliver a configurable baseline-only CLI,
   named-coordinate JSON, CSV summaries, recorded failures and resumable batches.
3. **By October 30: baseline experiments PR.** Complete the 45-run five-seed and
   search-budget study, with raw results, convergence plots and interpretation.
4. **By November 6: evaluation/performance PR.** Deliver ten frozen held-out
   circuits, baseline results, CPU hotspot profiles and ECE reproduction evidence.
   Feed these into the joint November 13 comparison milestone.

Work in `src/sa_sp/`, `integration/evaluation/`, `tests/`, `benchmarks/` and the
relevant result documents. Keep the received `seqpair_usc.py` and `HowToRun.txt`
unchanged. Reuse existing export/scoring code. A C++ rewrite is not required;
CUDA remains optional after profiling. Open focused PRs for Lucas to review.

## What should Lucas deliver

Review Alarik's protocol and fixture PR, maintain PPO, run independent training
seeds and reward ablations, evaluate frozen checkpoints on the held-out manifest,
and reproduce PPO on ECE. Maintain the learning page and integrate the joint
report, plots and final demonstration. Both teammates review scoring changes.

## Which existing items are eligible for closure after review

The parser, deterministic environment, masked actor/critic, PPO updates,
checkpoint save/load/resume, initial synthetic training and first 35-trial
comparison have recorded evidence. Their implementation subtasks can close when
reviewers accept that evidence. Formal benchmark, multiple-training-seed,
held-out and ECE deliverables remain open. A checkbox in the Team Hub does not
automatically update the sheet; use the matching C-ID in Live Tasks.

No Discord message or GitHub assignee change is sent by this handoff. The owner
can paste the short bilingual task instructions from the project conversation.
