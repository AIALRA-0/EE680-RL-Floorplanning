# How the actor and critic read a placement state

The maintained page's **02** chapter reads actual saved evaluation traces, including
node features, net incidence, action probabilities, value estimates and placement
progression. Read each explanation with its actual source excerpt, then move the
replay slider to inspect another decision.

Reusable encoding and network code lives in `src/ppo/encoding.py` and `model.py`.
The [design](../../docs/ppo-design.md) specifies the feature order, message passing,
masking and reward conventions.

From the repository root, after training:

```powershell
# Load the saved network; export every evaluated action and its inputs.
python steps/02-policy/run.py --input benchmarks/alarik-synthetic/blknet_5.txt --checkpoint steps/build/ppo-5-10-seed0/checkpoint.pt --output steps/02-policy/build/replay.json
```

The actor uses argmax for this command; add `--sample --seed 10000` to sample.
The process exits with status 2 if the complete layout fails independent legality
checks. Inference does not update model weights.

The chapter now has 20 explanations, from the eight input columns through net
message passing, candidate features, actor/critic and masking. Each explanation
uses numbers from the selected replay and links its actual source definition.
Use the experiment and checkpoint selectors to compare the original 5/10-macro
model with the 20/50/100-macro fine-tuning runs. The local zoom and exact legal
anchor overlay also work on the larger grids.
