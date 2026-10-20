# Alarik's supplied SA-SP material

This document records the initial import and native smoke run. Subsequent work
added coordinate export and a separate shared-protocol adapter without editing
the received files. See the [current comparison](ppo-sa-sp-comparison.md) and
[team handoff](team-handoff.md) for the completed export and next review tasks.

## Where did these files come from

Alarik supplied `SASPADRun.zip` to the team. All 25 files were copied unchanged
into this private working tree. The archive contains a Python script, its
original run instructions, five synthetic circuit inputs, and 18 result images.
It does not contain a C++ implementation.

| Received files | Repository location |
|---|---|
| `ADRun/seqpair_usc.py` and `ADRun/HowToRun.txt` | `src/sa_sp/` |
| `ADRun/blknet_5.txt`, `blknet_10.txt`, `blknet_20.txt`, `blknet_50.txt`, `blknet_100.txt` | `benchmarks/alarik-synthetic/` |
| All 18 `ADRun/Blk*Floor.png` and `ADRun/Blk*Results.png` images | `docs/sa-sp-results/alarik/` |

`src/sa_sp/` is the current Python baseline location (moved from `cpp/sa_sp/`).
The received script keeps
its original filename and Python implementation. Alarik's existing images are
received artifacts, separate from the new local smoke run described below.
The input headers identify the circuits as auto-generated designs. They are
synthetic samples, not verified official course benchmarks.

All 25 copied files were compared byte-for-byte with their archive entries, and
the staged Git blobs also preserve those bytes. The original `HowToRun.txt`
uses CRLF line endings: the default `git diff --cached --check` flags those
three received lines, while `git -c core.whitespace=cr-at-eol diff --cached --check`
passes. No global Git setting or received file was changed to suppress this.
All 18 PNGs passed decoding checks. The v1.0.0 README auditor flags nine floorplan
PNGs because they contain a text metadata container. Inspection found only
`Software: Matplotlib version3.11.2, https://matplotlib.org/` and DPI information
in those files. The other nine PNGs contain resolution/color information.
The originals are retained, so these nine `IMAGE_METADATA` findings remain in
the unmodified strict audit alongside the documented English-only exceptions.
The received terminal screenshots also retain their original shell prompts.

The archive provides no upstream URL or license file. Alarik still needs to
document the exact source/version and redistribution terms before these files
are distributed beyond the private team. No open-source license was added.

## How can the received script be run

Run from the repository root with a Python environment containing NumPy and
Matplotlib. The original `HowToRun.txt` assumes that input files are beside the
script; use the repository paths below after this import.

```powershell
# Run the received script on the five-block input and keep its new image outside Git.
python src/sa_sp/seqpair_usc.py --input benchmarks/alarik-synthetic/blknet_5.txt --alpha 0.5 --beta 0.5 --seed 0 --output "$env:TEMP/ee680-sa-sp-smoke5.png"
```

On October 15, 2026, the unchanged script was run locally with Python 3.12.7,
NumPy 2.4.4, and Matplotlib 3.10.9. The command used the same input, weights, and
seed as above; its image was saved in an external review directory.

| Quantity reported by the script | Observed value |
|---|---:|
| Blocks / nets | 5 / 5 |
| Initial / final Area | 1908 / 880 |
| Initial / final HPWL | 253 / 84 |
| Moves / accepted moves | 36000 / 34524 |
| Acceptance rate | 95.90% |
| Algorithm runtime | 4.138 seconds |

The process exited with code 0, generated its PNG, and reported no overlaps in
the initial or final placement. These are the supplied script's results;
independent geometry/metric validation remains part of C07/C08/C11. The runtime
belongs to this local run, not a CPU/GPU comparison or a Linux compatibility test.

## What still needs agreement or verification

- The script calculates HPWL from macro lower-left corners and allows rotation
  by swapping width and height. Both developers must agree on the same
  conventions before comparing SA-SP with PPO.
- Its bounding box is anchored at the origin. The shared evaluator must agree
  on how translated placements are scored.
- Initial annealing temperature comes from raw Area changes, while acceptance
  uses a normalized Area/HPWL cost. Alarik should review that scale difference;
  the received source has not been changed here.
- The script currently filters unknown net references when calculating HPWL.
  Our new parser rejects them; the eventual shared evaluator must define the
  accepted input contract explicitly.
- Export final coordinates and orientations in the agreed `Placement` format,
  and record source provenance, official benchmark inputs, seeds, and raw logs.
  These are needed to complete C04/C10 and the later comparison.

## What does C09.1 add

`src/data/parse_blknet.py` reads `BLOCKS` followed by `NETS`. It returns a dictionary
with `blocks` (macro ID to original width/height) and `nets` (one list per net).
It preserves input order and does not generate coordinates or orientations.

Blank lines and full-line comments are ignored; section headers are
case-insensitive. Duplicate IDs, nonpositive/nonfinite sizes, malformed block
rows, invalid section order, and unknown net references raise `ValueError`.
Every circuit must contain at least one block and a `NETS` section; an empty
`NETS` section is allowed for disconnected circuits.

These files specify no canvas dimensions, coordinate units, pin offsets, net
weights, or rotation constraints. The parser does not invent those fields. C06
still needs to settle the shared format for the integrated implementation. The
subsequent [numbered learning steps](../steps/README.md) use an explicitly
documented teaching canvas and reward convention while that agreement is pending.

The parser uses only Python's standard library. Its 11 unit tests passed locally,
including all five received inputs and the error cases above.
