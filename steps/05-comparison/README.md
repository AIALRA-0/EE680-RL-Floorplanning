# How do PPO and SA-SP use the same scoring rules

Read chapter 05 in `/steps/reading.html` after starting `python steps/serve.py`.
Its embedded `/steps/comparison.html` reads actual independent validation, trial
coordinates and measured timings. It does not launch training or search.

The original `cpp/sa_sp/seqpair_usc.py` stays unchanged. `adapter.py` exports its
returned coordinates; `aligned.py` reuses its decoder with no rotation, a hard
canvas and the same normalized objective as PPO. The original smoke run and the
adapted comparison are labeled separately.

```powershell
python -m src.experiments.compare --output steps/build/comparison-reproduction --cases 5 10 20 50 100 --seeds 0 1 2 --stages 180
python -m unittest discover -s tests/python -p test_comparison.py -v
```

The command uses existing local checkpoints when present, otherwise the published
`steps/evidence/` checkpoints. Choose a new output directory; a plan already there
is protected from accidental overwrite. Each circuit has one argmax and three
sampled PPO inference runs from a frozen training-seed-0 model, and three SA seeds.

The proposal viewer retains SA's first 24 proposals, each new best and its final
proposal. It labels original iteration numbers, acceptance probabilities, random
draws, candidate feasibility and before/after costs. The final best layout can
differ from the last accepted state. Replay sliders preserve this distinction.

Read [conditions and results](../../docs/ppo-sa-sp-comparison.md),
[the shared protocol](../../integration/evaluation/protocol.py) and
[the teammate handoff](../../docs/team-handoff.md) before changing the comparison.
