# Results archive

Raw outputs of every experiment run (one JSON per run, written by
`run_experiment.py`) and the matching logs in `logs/`. File names encode
the configuration: attacker model and strategy, victim, judge,
`q<questions>_p<paraphrases per round>_r<rounds>`, search mode, timestamp.

Every "success" in these files is an automatic verdict. All successes in
the runs below were reviewed case by case by the AI assistant (Claude)
used in this project; the "genuine" counts are from that review, which was
not independently verified. Earlier runs predate several fixes (judge substring bug, exact-match
evaluator, answer-preservation check, question curation), so their raw
success counts are not comparable to later ones.

## Main results (curated questions, current pipeline)

| File (timestamp) | Run | Raw -> genuine | Log |
|---|---|---|---|
| `...minimal_edit__victim_qwen3_4b__...q22_p5_r8...20260927_053549.json` | **Full run**, 22 questions, reasoning victim `qwen3:4b` | 69 -> 14 (Grinch's dog, Cybermen, Marvel vs Capcom) | `run_full22_final.log` |
| `...minimal_edit__victim_qwen3_4b_instruct__...q16_p5_r8...20260927_142038.json` | **Non-reasoning victim** `qwen3:4b-instruct`, the 16 questions it answers | 64 -> 17 (Louisiana Purchase, Phantom composer); 4 more on the WWI-entry question gave only '1917' and count as borderline | `run_instruct16.log` |
| `...minimal_edit__...q5_p5_r8...20260926_131046.json` | Minimal-edit test, 5 questions | 6 -> 2 | `run_minimal.log` |

## Main study (96 questions, Colab A100)

Victim `qwen3:4b-instruct`, questions `data/main_study_questions.json`.
Ollama 0.34.4 for part 2 and B1; part 1 ran on an unpinned install whose
version was not recorded.
These runs also have the 20-answer confidence sample and re-test
("confirmed"). Labels for every success, with a reason, are in
`labels/` (Excel; column "My label").

| File (timestamp) | Run | Raw / confirmed -> genuine (genuine and confirmed) | Log |
|---|---|---|---|
| `...q96_p5_r8_evolutionary__main_study_combined.json` | **Full method**, combined from the two parts below | 454 / 424 -> 146 (135), on 22 of 95 attacked questions | see parts |
| `...q96_p5_r8_evolutionary__20260930_024234.json` | part 1: stopped after 68 questions (Colab disconnected) | | `logs/main_study_part1.log` |
| `...q28_p5_r8_evolutionary__20260930_114610.json` | part 2: the remaining 28 questions, same settings | | `logs/main_study_part2.log` |
| `...semantic_preserve...q96_p40_r1...20261001_000152.json` | **B1**, plain paraphrasing, 40 at once | 219 / 209 -> 68 (65), on 22 questions (13 shared with the full method) | `logs/main_study_B1_plain.log` |

## Baselines on 16 questions (Colab A100, 2026-09-28)

Victim `qwen3:4b-instruct`, questions
`data/accepted_questions_qwen3_4b_instruct.json` (15 attacked: the
WWI-entry question was judged wrong at baseline on the A100). Same budget
of 40 paraphrases per question.

| File (timestamp) | Run | Raw -> genuine | Log |
|---|---|---|---|
| `...semantic_preserve...q16_p40_r1...20260928_184240.json` | B1: plain paraphrasing, one shot | 22 -> 4 | `logs/A100_16q_B1_plain_oneshot.log` |
| `...minimal_edit...q16_p40_r1...20260928_185644.json` | B2: attacker prompt, one shot | 30 -> 4 | `logs/A100_16q_B2_minimal_oneshot.log` |
| `...minimal_edit...q16_p5_r8_reflective__20260928_191017.json` | B3: 8 rounds with feedback | 36 -> 10 | `logs/A100_16q_B3_reflective.log` |
| `...minimal_edit...q16_p5_r8_evolutionary__20260928_193042.json` | B4: full method, repeated | 35 -> 7 | `logs/A100_16q_B4_evolutionary.log` |
| `...minimal_edit...q16_p5_r8_evolutionary__20260929_014434.json` | Test of the pipeline-fixes branch (B4 settings) | 62, not reviewed | `logs/pipeline_fixes_branchtest.log` |

## Question curation (main study)

`curation/candidates_150.json` and `.log`: the 150 NQ-Open candidates the
non-reasoning victim answers correctly (screened on an A100).
`curation/candidate_review_decisions.xlsx`: suggested keep / reject /
unsure for each, and the team's decisions and edits.

## Earlier and incomplete runs

| File (timestamp) | Run | Notes |
|---|---|---|
| `...misleading_entity__...q10_p5_r8...20260926_050834.json` | 10 x 8, full rewrites, improved checks | 58 -> 4 genuine; mostly drift. Log `run10x8_v3.log` |
| `...misleading_entity__...q10_p5_r8...20260926_034913.json` | 10 x 8, stopped by hand after question 1 | partial. Log `run10x8_v2.log` |
| `...q22...20260926_160658.json` | Full run on c-001 | GPU failure: only 2 questions valid. Log `run_full22.log` |
| `...q22...20260927_010813.json` | Full run on c-003 | disk quota exceeded: 3 questions saved. Log `run_full22_c003.log` |
| `PARTIAL_GPU-FAILURE__B1_plain_oneshot__...` | Baseline B1, plain paraphrasing | GPU failure; only question 1 and the "too different" counts are usable (see its commit message) |
| `...judge_qwen3_4b...2025*` and `...20260923_*`, `...20260925_*` | Development runs | before the judge, checks and question set were fixed; not comparable |

The first 10 x 8 run (Sep 26, crashed on a GPU failure) never wrote a
results file; its only record is `logs/run10x8.log`.
