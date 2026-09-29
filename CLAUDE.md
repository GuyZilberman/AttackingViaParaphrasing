# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research harness for **adversarial paraphrasing attacks on LLM question answering**. An attacker LLM rewrites a
factual question so it keeps exactly the same meaning and answer, hoping the victim LLM then answers it wrongly. The
search is iterative: each round generates candidates, drops the ones that change the question, asks the victim the
rest, and (evolutionary search) mutates the candidates that fooled the victim most often. Every model call goes to a
local **Ollama** server over HTTP (`ollama_client.py`); there is no torch code in the pipeline (only in `data_fabric/`).

## Running

Imports are flat (`from config import ...`), so run everything **from the repository root**. The Ollama server must be
running at `http://127.0.0.1:11434` (`--ollama-base-url` to change) with the models pulled. `start_ollama.sh`
hardcodes a Linux VM layout; elsewhere run `ollama serve`.

Default models: attacker and equivalence judge `llama3.1:8b`; victim `qwen3:4b`; answer judge, answer-preservation
judge and reference model `gemma3:12b`. `qwen3:4b` is a reasoning build that reasons even with `think=False` (about 490
tokens per answer); `qwen3:4b-instruct-2507-q4_K_M` answers directly and is far more sensitive to paraphrasing.

```bash
python run_experiment.py --n-questions 3 --n-paraphrases 5 --max-rounds 2
python run_experiment.py --dataset-path data/accepted_questions.json --n-questions 10
python run_experiment.py --attacker-strategy semantic_preserve --n-questions 3   # the control
```

`python run_experiment.py --help` lists every flag. Each flag maps to a field of `ExperimentConfig` in `config.py`,
and `parse_args()` builds the config field by field: a new option goes in the dataclass, the argparse definitions and
that constructor call. `ATTACKER_STRATEGIES` is derived from `STRATEGIES` in `attackers/llm_paraphraser.py`
(`misleading_entity` = the adversarial prompt, `minimal_edit` = the default, `semantic_preserve` = the control).

Dependencies are not pinned (no requirements file): `requests`; `datasets` for `utils/download_nq_open.py` and
`data_fabric/download_questions.py`; `torch` + `transformers` only for `data_fabric/filter_questions.py`; `json5` optional.

### Tests

- `python -m pytest tests/test_pipeline.py`: offline tests with fake models for the check order, the leak check,
  confidence groups, the re-test and the reference gate. Run after changing `experiment.py`.
- `python tests/eval_judges.py --answer-judge gemma3:12b` / `--preservation-judge gemma3:12b`: scores judges against
  the hand labels in `tests/data/` (live Ollama). Note that the preservation prompt was tuned on those same labels.
- `python utils/question_equivalence_judge.py`: self-test of the equivalence judge on labelled pairs (live Ollama).

## Architecture

`experiment.run_experiment(cfg)` loops over questions (`dataset.load_questions`), writing
`results/<run_id>__<timestamp>.json` after each one (`complete` stays false until the run ends). The result schema is
the `experiment.py` module docstring. Per question (`_run_question` → `_iterative_attack`):

1. The victim answers the original greedily; if every evaluator marks it wrong, the question is not attacked.
2. Reference check (`--reference-model`): if the reference model gives the known answer to the original fewer than
   `reference_min_correct` of `reference_samples` times, the question is skipped (`skipped: "reference_unknown"`).
3. Baseline: `fitness_samples` victim answers on the original give `original_wrong_rate` (greedy counted as right),
   and `confidence_samples` fresh answers give `original_confidence_wrong_rate` and the `confidence_group`
   (certain = none wrong, mostly = under 25%, unsure).
4. Rounds: the attacker (`LLMParaphraser.generate_paraphrases`, shown the known answers and the history of earlier
   attempts) proposes candidates. Each passes, in order: leaked attacker notes → word overlap with the original
   (`min_original_overlap`) → answer leak → equivalence judge (`utils/question_equivalence_judge.py`) →
   answer-preservation judge (`utils/answer_preservation_judge.py`) → reference gate. Free checks come first.
   Rejections go into the attacker's history with a reason, except the reference gate, whose answers the attacker
   never sees.
5. Survivors are asked greedily plus `fitness_samples` times; `fitness` = share wrong. A success is a wrong greedy
   answer (not a non-answer) on a question the victim answered right.
6. Re-test: each success gets `retest_samples` fresh answers, compared with the confidence samples by a one-sided
   Fisher exact test; `confirmed_success` if p < `retest_alpha`. Report confirmed successes, by confidence group.

Answers are always graded against the ORIGINAL question and gold answers by `evaluators/llm_judge.py`
(`LLMJudgeEvaluator`, the only evaluator), cached per normalised answer within a question.

Question data: `data/` holds curated sets (`accepted_questions*.json`, rules in `data/CURATION.md`) and NQ-Open
samples built by `utils/download_nq_open.py` (victim-screened, context-checked). `data_fabric/` downloads all of
NQ-Open and filters it with a local transformers model (its output under `data_fabric/data/` is gitignored).
`experiments/colab_ideas/` holds the Colab scripts that tested the re-test, confidence, reference-gate and
instruction-evolution ideas (against commit `a1e21e5`, not this code).

## Gotchas

- **The answer judge sets the granularity.** It marks "1952" wrong against "6 February 1952" and "1968–1998" wrong
  against "1968" / "1998"; a search learns to ask for a different granularity. Check confirmed successes by hand.
- **Most flips on unsure questions are the victim's own uncertainty**, not the paraphrase: compare successes by
  confidence group, not overall.
- **One failed model call does not end the run**: a candidate whose calls fail is recorded with `status: "error"`;
  a question fails once at least 5 of its candidates, and at least half of those tried, have errored, and
  `--max-failures` consecutive failed questions stop the run.
- **Model output goes to stdout and logging**: rejections are printed (`[QUESTION EQUIVALENCE FAILED]`,
  `[REFERENCE CHECK FAILED]`, ...); everything else is `logging` at INFO.
