# How to study the complete placement-to-PPO loop

Start the server from the repository root:

```powershell
# Serve the structured course and preserved interactive labs.
python steps/serve.py
```

Open `/steps/reading.html` for the question-led course. Each question connects a premise, an explanation plan, source-grounded reasoning, an answer, a transferable principle, media and exercises. The interactive lab remains available at `/steps/index.html` and is embedded in the course.

| Step | Question | Implementation |
|---|---|---|
| [00-input](00-input/README.md) | How do dimensions and nets become circuit data? | `src/data/parse_blknet.py` |
| [01-placement](01-placement/README.md) | How are candidates checked, selected, executed and scored? | `src/environment/placement_env.py` |
| [02-policy](02-policy/README.md) | How do geometry and connectivity become a masked policy? | `src/ppo/encoding.py`, `model.py` |
| [03-rollout](03-rollout/README.md) | How do actual interactions become fixed training targets? | `src/ppo/algorithm.py` |
| [04-ppo-update](04-ppo-update/README.md) | How do GAE, PPO losses and Adam change parameters? | `src/ppo/algorithm.py`, `src/experiments/` |
| 05-comparison | How can frozen PPO and SA-SP share one evaluation contract? | `cpp/sa_sp/`, `integration/evaluation/` |

The lab retains detailed candidate checks, masks, network inputs, probabilities, before/after layouts, checkpoint selection, training curves and failure traces. Browser notes and the actual chapter-01 teaching session persist independently. Published English course material and curated evidence are tracked; local Chinese reading material and personal notes remain local.

Lucas owns Python/PPO, Alarik owns the baseline and planned C++ work, and shared integration connects their results. Read [the learning workflow](../docs/learning-workflow.md), [PPO design](../docs/ppo-design.md), [small-circuit results](../docs/ppo-first-run.md) and [scaled results](../docs/ppo-scaled-training.md).
