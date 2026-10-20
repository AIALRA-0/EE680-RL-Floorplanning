# How does the question-led learning workflow work

Start `python steps/serve.py` and open `/steps/reading.html`. The English reader
embeds the existing interactive workbench at `/steps/index.html`, retaining its
visualizations, individual checks, actual numerical records and browser notes.

## What did we adapt from LessonWeave

The reference is [AIALRA-LessonWeave](https://github.com/AIALRA-0/AIALRA-LessonWeave),
local commit `53720b8271f5c08a55306df1436e638ae74eb5a1`, including local uncommitted
workflow edits inspected on 2026-10-20. Its `lean-c-1.0` workflow separates source
material, question dependencies, chapter order, question explanations, media,
exercises, structural checks and reader assembly. Relevant reference files are
`docs/current-flow.md`, `framework/05_writing/lean-flow.md`,
`framework/05_writing/lean_contract.py` and `framework/05_writing/light-checks.md`.

This is a manually authored static adaptation. It does not run LessonWeave's
generation pipeline or claim compatibility with its full C-unit schema. The reader
HTML, CSS and JavaScript are independently implemented here; no LessonWeave reader
code or runtime is copied into this project.

## Which artifacts make the progression reviewable

1. `steps/course/materials.json` identifies actual source functions and line ranges.
2. `questions.json` connects each question to prerequisites, three explanation
   steps, source references, a conclusion and a concrete misunderstanding to avoid.
3. `chapters.json` assigns each question exactly once and embeds the relevant lab
   with a reading guide, labeled components and takeaway. Chapter bridges reuse
   the preceding question's actual answer.
4. `book.json` assembles those records, the glossary and exercises. Each chapter
   has four multiple-choice questions and one short-answer question. This is the
   EE680 adaptation of the reference's per-section exercise pattern.
5. `python steps/check_course.py` checks source spans, unique IDs, prerequisite
   order, assignment, media, plan references and answer indices. These are
   deterministic structural checks, not an independent semantic grade.

The optional browser check is
`python tests/integration/check_learning_pages.py --output steps/build/page-validation`.
It uses the validation environment's Playwright/Chromium installation, separate
from runtime requirements. Add `--include-local-zh` only when that ignored local
edition exists. It checks six light/dark viewport combinations, question and code
navigation, actual comparison traces, exercises, notes and preservation of the
chapter-01 state. Screenshots remain in the selected ignored output directory.

Six chapters contain sixteen questions. Chapter 01 retains 195 detailed frames
covering the five-macro deterministic search. Chapters 02-04 retain 20, 14 and 16
explanations, with actual tensors, source, rollout samples, GAE, losses, gradients
and parameter changes. Chapter 05 embeds same-scale PPO/SA-SP layouts, retained
proposal events and measured result tables.

For maintenance, edit the reviewed source records in `materials.json`,
`questions.json` and `chapters.json`, then run `python steps/assemble_course.py`
and `python steps/check_course.py`. Assembly carries the actual preceding answer
into the next premise and chapter bridge. The check also requires the three
source artifacts to match the assembled book. Review explanations against any
changed implementation before updating their source spans.

## How are language and personal notes handled

Tracked course files and project documentation are English. A Chinese local edition
uses the same question IDs and source references under ignored
`steps/build/local-zh/`. Its entry is `/steps/build/local-zh/reading.html`.

Reader position, question notes and short answers stay in browser local storage,
separately for each language. Existing workbench notes and chapter-01 execution
state retain their own storage. Reading another chapter does not reset that state.

## Which experiment records ship with the course

`steps/evidence/` contains selected actual checkpoints, saved replays, teaching
updates, evaluation/ablation records and figures. Large JSON files use lossless
gzip. `manifest.json` records source paths, byte lengths and SHA-256 values for
checking decompressed bytes. Models came from this project's CPU training on the
supplied synthetic circuits. No external model or benchmark download was added.

The server prefers local `steps/build/` pointers and falls back to tracked evidence
after a fresh clone. It reads compressed records through the same API without
importing PyTorch. New experiments write to ignored build directories.

Comparison JSON includes circuit, protocol, coordinates, trial seed, independent
validation and timing. Native SA-SP smoke output stays distinct from adapted
comparison trials. Read [comparison conditions](ppo-sa-sp-comparison.md) and
[the team handoff](team-handoff.md) before extending either solver.

Run `python tests/integration/check_saved_evidence.py` to check archived byte
hashes, all eighteen teaching updates and their 2077 transitions, plus independent
rescoring of the published comparison trials. It uses the standard library and
the project's own evaluator; its output stays under ignored `steps/build/`.
