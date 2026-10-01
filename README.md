# Attacking LLMs via subtle paraphrasing

Course project 5.5 (Trustworthy Machine Learning, Spring 2026). Can a small,
meaning-preserving rewording of a factual question make an LLM answer it
wrongly, and reliably so?

An **attacker** LLM rewrites a question in small steps. Each paraphrase must
pass validity checks (same question, same correct answer, mostly the same
words) before it reaches the **victim** LLM. An LLM **judge** decides whether
the victim's answer is wrong. The search runs for several rounds, feeding back
what worked (evolutionary search, after GEPA). Every success is then re-tested
on fresh answers, and successes are reviewed by hand.

All models run locally through [Ollama](https://ollama.com):

| Role | Model |
|---|---|
| Victim | `qwen3:4b-instruct` (main study); `qwen3:4b` (reasoning variant, comparison) |
| Attacker, question-equivalence check | `llama3.1:8b` |
| Answer judge, answer-preservation check | `gemma3:12b` |

## Setup

Requirements: Python 3.10+, an NVIDIA GPU with at least 24 GB of memory for
all three models at once (we used an A100 40 GB on Google Colab; a 16 GB T4 is
too small), and about 16 GB of disk for the models.

```bash
# 1. Ollama, pinned to the version used for the results (part 1 of the
#    main study ran on an unpinned install whose version was not recorded)
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.34.4 sh
OLLAMA_KEEP_ALIVE=24h OLLAMA_MAX_LOADED_MODELS=3 ollama serve &

# 2. Models (~16 GB)
ollama pull qwen3:4b-instruct
ollama pull llama3.1:8b
ollama pull gemma3:12b
ollama pull qwen3:4b          # only for the reasoning-victim runs

# 3. Code
git clone https://github.com/GuyZilberman/AttackingViaParaphrasing.git
cd AttackingViaParaphrasing
pip install -r requirements.txt
```

The code talks to Ollama at `http://127.0.0.1:11434` (change it with
`--ollama-base-url`). `start_ollama.sh` starts the server on the TAU servers we
used first; elsewhere, start it as above.

**Google Colab:** choose an A100 runtime, mount Google Drive, run the three
setup steps in a cell, and pass `--results-dir` a folder on Drive. Colab wipes
the runtime's disk when it disconnects; results are saved after every
question, so a disconnect loses at most the question in progress.

## Running an experiment

```bash
python3 run_experiment.py --help        # all options and defaults
```

The main study (full method on the 96 curated questions, about 2–2.5 h on an
A100):

```bash
python3 run_experiment.py \
    --dataset-path data/main_study_questions.json --n-questions 96 \
    --victim-model qwen3:4b-instruct \
    --attacker-strategy minimal_edit --n-paraphrases 5 --max-rounds 8 \
    --search evolutionary --results-dir results
```

The other configurations differ only in these flags:

| Run | Flags |
|---|---|
| B1: plain paraphrasing, no attack intent | `--attacker-strategy semantic_preserve --n-paraphrases 40 --max-rounds 1` |
| B2: attacker prompt, one shot | `--attacker-strategy minimal_edit --n-paraphrases 40 --max-rounds 1` |
| B3: rounds with feedback only | `--attacker-strategy minimal_edit --n-paraphrases 5 --max-rounds 8 --search reflective` |
| B4 / main: evolutionary search | `--attacker-strategy minimal_edit --n-paraphrases 5 --max-rounds 8 --search evolutionary` |
| Reasoning victim | `--victim-model qwen3:4b` with `--dataset-path data/accepted_questions.json` |

All four give the attacker the same budget of 40 paraphrases per question.
The baselines B1–B4 in the report were run on the 16 questions in
`data/accepted_questions_qwen3_4b_instruct.json`.

The run prints a summary at the end. Unless set otherwise, it also:

- samples the victim 20 times on each original question to place it in a
  confidence group (`--confidence-samples`);
- re-tests every success on 20 fresh answers and marks it **confirmed** if it is
  wrong significantly more often than the original (one-sided Fisher exact
  test, `--retest-samples`, `--retest-alpha`);
- stops after 2 questions in a row fail, for example when the Ollama server or
  GPU stops working (`--max-failures`).

## Output

Each run writes one JSON file to `--results-dir`, named after its
configuration, e.g.
`attacker_llama3_1_8b_minimal_edit__victim_qwen3_4b_instruct__judge_gemma3_12b__q96_p5_r8_evolutionary__<timestamp>.json`.
It is rewritten after every question and contains the settings, every
paraphrase with the reason it was rejected or the victim's answers, every
success with its re-test, and a summary. The format is documented at the top
of `experiment.py`.

A success is an automatic verdict. The numbers in the report count only
successes reviewed by hand as **genuine**: the same question in other words,
answered wrongly. The result files and logs of all runs are on the
`results-archive` branch (`results/README.md` there indexes them).

## Reproducing the report's numbers

```bash
git restore --source=origin/results-archive --worktree -- results/   # result files and labels
python3 analysis/report_tables.py
```

This prints every table the report uses (main study vs. B1, confidence
groups, the B1–B4 baselines, reasoning vs. non-reasoning victim) from the
result files and the success labels in `results/labels/`. The labels come
from a case-by-case review by the AI assistant used in the project and were
not independently verified. `results/` is git-ignored on `main`, so the
restored files don't show up as changes.

## Questions

Questions come from NQ-Open (`google-research-datasets/nq_open`) and are
curated by hand: one defensible answer, no unstated context, a fitting gold
answer, and a victim that answers correctly. The rules and the process are in
`data/CURATION.md`.

| File | Contents |
|---|---|
| `data/main_study_questions.json` | the 96 questions of the main study |
| `data/accepted_questions_qwen3_4b_instruct.json` | the 16 questions of the baseline and victim comparisons |
| `data/accepted_questions.json` | the original 22, screened against the reasoning victim `qwen3:4b` |
| `data/new_questions_qwen3_4b_instruct.json` | the 80 questions added for the main study |

New candidates: `python3 utils/download_nq_open.py --n 150 --victim-model qwen3:4b-instruct --judge-model gemma3:12b --output candidates.json`.

## Tests

```bash
python3 -m pytest tests/test_pipeline.py      # offline, fake models, no GPU
python3 tests/eval_judges.py --answer-judge gemma3:12b --preservation-judge gemma3:12b
                                              # judge accuracy on hand-labelled data (needs Ollama)
python3 tests/test_model_sanity.py            # victim answers to sample questions (needs Ollama)
```

## Repository layout

| Path | What it is |
|---|---|
| `run_experiment.py` | command-line entry point |
| `config.py` | all settings and their defaults |
| `experiment.py` | the search, the checks, the re-test, and the result file |
| `attackers/llm_paraphraser.py` | attacker prompts (`minimal_edit`, `semantic_preserve`, `misleading_entity`) |
| `victim.py` | the victim model |
| `evaluators/llm_judge.py` | the answer judge |
| `utils/question_equivalence_judge.py` | check: does the paraphrase ask the same question? |
| `utils/answer_preservation_judge.py` | checks: same correct answer, enough original words kept, answer not named |
| `utils/download_nq_open.py` | download and screen NQ-Open candidates |
| `ollama_client.py` | Ollama HTTP client |
| `data/` | question sets and curation rules |
| `analysis/report_tables.py` | recomputes the report's tables from the archived results and labels |
| `tests/` | offline tests, judge evaluation, labelled data |
| `data_fabric/` | a teammate's earlier data-gathering scripts (Hugging Face models, not used by the pipeline) |
| `experiments/colab_ideas/` | scripts behind the "four ideas" experiments (re-test, confidence groups, reference gate, instruction evolution); see its README |
