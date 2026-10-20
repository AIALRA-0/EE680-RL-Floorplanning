# What should each teammate do after reviewing this delivery

This delivery completes the first CPU PPO training/evaluation/replay loop and a
shared-protocol comparison against an adapted version of the received SA-SP.
The owner reviews and merges the PR; no automatic merge is performed.

## What can be marked complete after review

- Import the supplied SA-SP script, original instructions, five synthetic circuits
  and eighteen result images into the agreed repository directories.
- Parse the input format with validation and meaningful tests.
- Implement the integer-anchor, no-rotation environment and exact legal mask.
- Connect geometry/net encoding, masked actor, critic, rollout, GAE and PPO updates.
- Save/load and resume checkpoints; independently validate exported layouts and
  inspect actual input/action/reward/parameter records.
- Complete the 5/10-macro CPU run, three-seed reward-weight ablation and
  20/50/100-macro fine-tuning, retaining the failed 192-board probe.
- Export native SA-SP coordinates, apply the shared protocol, run the first
  five-size comparison and retain per-trial validation/timing.
- Publish the English question-led course with embedded visualizations and retain
  the Chinese reading edition locally.

These repository items are eligible for closure after teammate review. This PR
does not edit the external Live Tasks, Live Checklist or Live Gantt.

## What should Akira review and implement next

1. Read `docs/ppo-sa-sp-comparison.md` and chapter 05. Confirm no rotation,
   lower-left integer anchors, hard board dimensions, tight-bounding-box Area,
   lower-left HPWL and the normalized objective as explicit shared decisions.
2. Review `cpp/sa_sp/adapter.py` against unchanged `seqpair_usc.py`: coordinate and
   dimension export, and the independently reproduced native 880/84 five-macro
   result. Original files retain Alarik's attribution.
3. Review `cpp/sa_sp/aligned.py`: shelf initialization, swaps, normalized
   temperature probes, board rejection and Metropolis acceptance. Inspect an
   accepted proposal and a rejected proposal in the viewer. This is an adapted
   Python baseline; the C++ solver has not been implemented.
4. Implement C++ geometry/scoring under `cpp/geometry/`, using exported coordinates
   and per-net HPWL as parity fixtures. Match Python legality, tight Area and
   endpoints first, including touching edges and invalid layouts.
5. Keep later SA-SP C++ work in `cpp/sa_sp/`; connect through shared evaluation
   after parity tests pass. Profile CPU before considering optional CUDA.

No teammate message or guessed GitHub assignment is sent. These concrete review
instructions are included in the PR for the team to act on.

## What should Lucas do next

1. Read the course in order, including chapter-01 checks before PPO. Explain one
   actual transition, GAE recursion and Adam step; retain questions in reader notes.
2. Reproduce documented tests and shipped-checkpoint inference on ECE Linux.
   Record actual runtime and dependency differences.
3. Agree the shared protocol with Akira before changing rewards or endpoints.
4. Add held-out circuits and multiple training seeds for formal generalization
   studies. Retain failures in completion statistics and separate offline training
   from per-layout inference costs.

## Which course deliverables still need work

Formal benchmark permission/import, wildcard/unseen-circuit evaluation, ECE-server
reproduction, C++ parity/bindings, multiple-training-seed replication and final
milestone/report/demo preparation remain open. CUDA is optional. No external
dataset is downloaded and those later items are not marked complete.
