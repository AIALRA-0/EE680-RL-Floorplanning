# What is included in this repository delivery

Delivery date: 2026-10-20. Branch: `feat/ppo-training-loop`, based on existing
`main` commit `ba270fb11a5aad30ea210d22e9d3ea6722ac2e2a`. The repository remains
private. Existing commits, the original `.gitignore` and the received SA-SP files
are preserved. The PR is for team review; the owner performs any merge.

## Which work is ready to review

- Actual parser, placement environment, legal mask, PyTorch encoder/actor/critic,
  rollout, GAE, clipped PPO updates and checkpoint save/load/resume.
- Completed 5/10-macro training, reward-weight ablation and 20/50/100-macro
  fine-tuning, with fixed evaluation records and the failed 192-board probe.
- Original SA-SP import, coordinate export and an explicitly adapted baseline
  using the common hard canvas, no rotation and independent normalized scoring.
- An English question-led course adapted from the inspected LessonWeave workflow:
  six chapters, sixteen questions, twenty-four multiple-choice questions and six
  short-answer questions, retaining the detailed interactive workbench.
- Actual checkpoints, compressed replays, teaching updates, plots and source
  provenance under `steps/evidence/`, with fresh-clone fallback in the server.
- Concrete next steps for Akira and Lucas in [the handoff](team-handoff.md).

The exact added/modified/deleted paths are listed in [the file manifest](delivery-files.txt).
The final comparison conditions, raw-output locations and measured results are in
[the comparison report](ppo-sa-sp-comparison.md). The Chinese reader and personal
notes remain local under ignored `steps/build/local-zh/` and browser storage.

## How is the final directory structure organized

```text
.
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- src/
|   |-- data/
|   |-- environment/
|   |-- ppo/
|   `-- experiments/
|-- cpp/
|   |-- geometry/                  (empty; tracked for C++ work)
|   |-- sa_sp/                     (received Python solver and adapters)
|   `-- cuda/                      (empty; optional)
|-- integration/
|   |-- bindings/                  (empty)
|   |-- evaluation/
|   `-- visualization/
|-- benchmarks/alarik-synthetic/
|-- tests/
|   |-- fixtures/                  (empty)
|   |-- python/
|   |-- cpp/                       (empty)
|   `-- integration/
|-- docs/sa-sp-results/alarik/
`-- steps/
    |-- 00-input/
    |-- 01-placement/
    |-- 02-policy/
    |-- 03-rollout/
    |-- 04-ppo-update/
    |-- 05-comparison/
    |-- course/
    `-- evidence/
```

Empty directories retain their necessary `.gitkeep`; redundant parent markers
are removed when real files begin tracking the directory. Source ownership stays
with `src/`, `cpp/` and shared `integration/`; numbered steps contain learning
material and entry points.

## Which checks were actually run

Commands below run from the repository root unless otherwise stated. The verified
environment uses Python 3.12.7, PyTorch 2.6.0+cpu, NumPy 2.4.4 and Matplotlib
3.10.9. Optional browser validation uses Playwright 1.54.0 with Chromium.

| Command or actual check | Result |
|---|---|
| `python -m unittest discover -s tests/python -p 'test_*.py' -v` | 61 passed in 12.503 seconds in the working tree |
| Same unit command in a snapshot exported from the staged Git tree | 61 passed in 11.966 seconds |
| `python steps/check_course.py` | Six chapters, sixteen questions, source spans, dependencies, media and exercises passed |
| `python tests/integration/check_learning_pages.py --output steps/build/page-validation --include-local-zh` | Six light/dark viewport combinations passed; all sixteen questions checked at desktop size, representative 00/01/05 questions at narrower sizes; no page-script errors or document overflow |
| Separate PPO browser sweep | All 50 explanations passed in each of six views; actual checkpoint/sample switching and numerical panels passed |
| Chapter-01 source contract in browser validation | 40 representative states, 195 frames and 451 code references passed |
| `python -m compileall -q src integration cpp/sa_sp steps tests` | Passed |
| `node --check` on the maintained reader, comparison and lesson scripts | Passed |
| README local Markdown rendering and link check | Six views passed; English-only and all existing external URLs preserved |
| Archive entry and staged-blob byte comparisons | All 25 received files match `SASPADRun.zip` exactly |
| Published-checkpoint inference in the staged-tree snapshot | Five-macro Area 1221, HPWL 135; independently complete and legal |
| Server and published-record loading with Python `-S` | Passed using only the standard library |
| `git diff --check` | Passed |
| `git diff --cached --check` | Flags the three original CRLF lines in received `HowToRun.txt`; that file remains byte-identical |
| `git -c core.whitespace=cr-at-eol diff --cached --check` | Passed with received CRLF preserved |

Viewport combinations are 1440, 736 and 320 pixels, each in light and dark mode.
Reader notes survive reload; reading preserves the chapter-01 environment. The
8868 service was restarted and its prior session snapshot remained identical.

The five-size comparison completed all 35 trials: every output was complete and
independently legal. Fifteen adapted SA runs executed 2,052,000 proposals. SA
achieved lower normalized cost at each tested size; frozen PPO inference was
faster, with offline training reported separately. The full per-seed table is in
the comparison report.

`python tests/integration/check_saved_evidence.py --output steps/build/evidence-validation.json`
passed: 200 original-byte hashes, five runs, eighteen isolated teaching updates,
2077 transition calculations and independent rescoring of all 35 comparison trials.
The same check passes in the staged-tree snapshot. The final browser sweep also
opens every one of the 35 published trials in addition to the six viewport checks.

## What does the unmodified README audit report

The pinned `github-readme-standardizer` is exactly tag `v1.0.0`, commit
`9348b81dce08f9ffe555cb4f5636e9dfbcb767eb`. Its command is:

```powershell
python <pinned-v1.0.0>/scripts/audit_readme.py . --scan-repository --strict-warnings
```

The unmodified strict audit reports **FAIL**: one `MISSING_README` error for the
intentionally absent `README.en.md`, one `PRIMARY_LANGUAGE_REVIEW` warning, and
twenty `IMAGE_METADATA` errors. The explicit English-only requirement takes
precedence over its bilingual default. Nine received floorplans and eleven
retained experiment plots have Matplotlib Software/DPI metadata. Received
terminal screenshots have resolution/color metadata and retain their original
shell prompts. All twenty-nine images decode and have no EXIF entries; reviewed
metadata keys are Software, dpi, gamma and srgb. Original images stay unchanged.

This is a documented manual review, not a strict-audit PASS. Local rendered checks
also do not claim inspection of the live GitHub site's light/dark UI.

## Which publication risks and limits were reviewed

The outgoing branch contains one project's implementation, supplied synthetic
inputs, locally trained model states and experiment artifacts. Expanded inspection
loaded forty-six models with `weights_only=True`, checked finite floating tensors,
read compressed JSON and inspected PNG metadata. Text, decompressed records and
model metadata contained no Chinese text or actual host-absolute paths. Existing
Windows-relative paths in training metadata remain as recorded provenance.

Gitleaks 8.30.1 scanned staged text and the expanded records/model metadata with
zero findings. Google planning links remain the intended team links. No deployment
credentials, external dataset download, new license or copied LessonWeave runtime
is introduced. Received SA-SP provenance/redistribution terms still require the
author's documentation before sharing beyond this private team.

F: temporarily filled during screenshot output. Original generated JSON replays
were preserved in an E: archive; lossless copies continue to serve from the build
paths. Source, models, notes and original received files were not deleted. Large
future training runs need additional disk capacity.

Formal course benchmarks, wildcard/unseen-circuit evaluation, multiple training
seeds, ECE Linux reproduction, C++ scoring/bindings and optional CUDA remain open.
Review items eligible for checklist closure are enumerated in the handoff; no
external task/checklist or repository protection setting is changed by this PR.
