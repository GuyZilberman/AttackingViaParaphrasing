# Colab runs that tested four ideas for the attack

These scripts produced the numbers in the findings doc "Paraphrase attack: testing four ideas"
(2026-09-26 to 2026-09-29; a private Claude Docs page, ask Tomer for access). They test the pipeline
as it was at commit `a1e21e5` (the tip of `minimal-edit-paraphrasing` then), which `setup.sh` checks out;
they do not import the code on this branch.

| Idea | Verdict | On this branch |
| --- | --- | --- |
| Re-test flagged successes with fresh samples | Essential | `--retest-samples` (default 20) |
| Report results by the victim's confidence on the original | Essential | `--confidence-samples` (default 20) |
| Reference-model gate | Useful as one check among several | `--reference-model`, `--reference-samples`, `--reference-min-correct` |
| Evolve the attacker's instruction | No gain with either victim | not added |

## Main results

- Re-test: 14 of 37 flagged successes (reasoning victim `qwen3:4b`) held up at 30 fresh samples; by hand,
  13 of those 14 were drift, judge artifacts or an answer leak.
- Confidence: greedy flips on branch-valid paraphrases were 4-5% when the victim got the original right
  20/20 times, about 21% at 1-4/20 wrong and about 50% at 5+/20 wrong. The adversarial prompt matched
  plain paraphrasing in every group.
- Reference gate (`gemma3:12b`, at least 4 of 5 answers right): on the 86 hand-labelled paraphrases with
  well-posed originals it rejected 0% of valid ones and accepted 43% of drift (branch judges: 29% and 24%).
- Instruction evolution: no evolved instruction beat the seed prompt. Against the non-reasoning victim
  `qwen3:4b-instruct-2507` (follow-up F: 37 training and 74 held-out questions), the seed prompt did not
  beat plain paraphrasing either (+0.02 wrong-rate gain, 95% CI -0.02 to +0.06). That victim does fail
  for real, though: 29 of 55 re-test-confirmed flips were faithful rewrites answered wrong.

## Files

- `setup.sh`: installs Ollama 0.34.4, clones the repo at `a1e21e5` into `/content/avp`, pulls the models.
- `rebuild_sample.py`: rebuilds the 480 sampled questions from Hugging Face NQ-Open by ID.
- `exp.py`: the four-idea pipeline (stages checkpointed to `/content/results/state.json`).
- `extras.py`: Colab settings, extra stages and reports, and follow-up F (`start_followup()`,
  `report_followup()`). Its docstring has the usage.

The result files (`state.json`, `state_v2.json`) are kept outside git.
