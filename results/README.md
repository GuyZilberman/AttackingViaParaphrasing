# Results archive

Raw outputs of every experiment run (one JSON per run, written by
`run_experiment.py`) and the matching logs in `logs/`. File names encode
the configuration: attacker model and strategy, victim, judge,
`q<questions>_p<paraphrases per round>_r<rounds>`, search mode, timestamp.

Every "success" in these files is an automatic verdict. All successes in
the runs below were reviewed by hand; the "genuine" counts are from that
review. Earlier runs predate several fixes (judge substring bug, exact-match
evaluator, answer-preservation check, question curation), so their raw
success counts are not comparable to later ones.

## Main results (curated questions, current pipeline)

| File (timestamp) | Run | Raw -> genuine | Log |
|---|---|---|---|
| `...minimal_edit__victim_qwen3_4b__...q22_p5_r8...20260927_053549.json` | **Full run**, 22 questions, reasoning victim `qwen3:4b` | 69 -> 14 (Grinch's dog, Cybermen, Marvel vs Capcom) | `run_full22_final.log` |
| `...minimal_edit__victim_qwen3_4b_instruct__...q16_p5_r8...20260927_142038.json` | **Non-reasoning victim** `qwen3:4b-instruct`, the 16 questions it answers | 64 -> 21 (Louisiana Purchase, Phantom composer, US joining WWI) | `run_instruct16.log` |
| `...minimal_edit__...q5_p5_r8...20260926_131046.json` | Minimal-edit test, 5 questions | 6 -> 2 | `run_minimal.log` |

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
