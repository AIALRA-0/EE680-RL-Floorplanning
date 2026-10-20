# How does an individual placement produce feedback

Chapter 01 uses a fixed policy: visit macros in input order and choose the first legal anchor in a row scan. It validates the environment and explains state, checking, selection, execution and feedback. PPO episodes separately start empty.

The maintained lab unfolds each actual check into short explanations. `i` identifies a candidate anchor; `x=i%W` gives its column and `y=i//W` gives its row. Boundary checks inspect left, bottom, right and top edges separately. Four strict overlap comparisons require positive-length intersection along both axes, allowing touching edges. A candidate becomes legal only after every required test passes.

Checking or selecting a candidate does not mutate placements. `env.step(action)` revalidates and writes coordinates, checks termination, then returns observation, reward, terminated, truncated and info. The teaching trace shows the same actual call before and after its write. Reading another explanation does not reexecute it.

```powershell
# Start the persistent lab and structured course.
python steps/serve.py
# Export the actual deterministic placement trajectory.
python steps/00-input/run.py
python steps/01-placement/run.py
```

B0 at (0,0) ends at x=25. B1 candidates (0,0) through (24,0) overlap B0; at (25,0), 25<25 is false and the rectangles merely touch. The recorded search visits each rejected anchor rather than jumping to a known solution.

Actions are integer lower-left coordinates without rotation. The default 64-by-64 demonstration canvas is explicit, rather than inferred from input files. Area is the tight occupied bounding box; HPWL sums lower-left endpoint spans for each net. Unplaced endpoints are excluded from partial metrics.

Intermediate legal reward is zero. Completion returns `-alpha*Area/(W*H)-(1-alpha)*HPWL/max(1,M*(W+H))`; impossible continuation returns -2 and terminates. The lab records metrics even when reward is zero. Chapter 02 introduces learned selection; chapter 01 keeps its fixed policy and separate saved progress.
