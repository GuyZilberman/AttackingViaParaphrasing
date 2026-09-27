# Curating attack questions

How the question sets in this directory are built, and the rules a new
question must pass. A question only makes sense as an attack target if a
paraphrase that "flips" the victim's answer shows a real framing effect,
not an ambiguous question, a wrong gold answer, or a victim that never
knew the answer.

## Rules for a question

A question is kept only if **all** of these hold:

1. **One defensible answer.** A well-informed person would give the same
   answer. Reject questions with several valid answers where the gold lists
   one (e.g. "Where did the dewey decimal system come from?": Melvil Dewey,
   the USA, Amherst College, 1876 are all defensible).
2. **No unstated context.** The answer must not depend on location or
   jurisdiction (legal ages, laws, prices), on when it is asked ("last",
   "current", "new", records that can change), or on an unnamed reference
   ("the tv series", "the war").
3. **The gold answer fits the question.** A "where" question needs a place,
   a "when" question a date, and so on. Reject gold answers that are wrong,
   incomplete, contradictory (e.g. "1998" and "1996" for the same event), or
   of the wrong kind.
4. **A real question with an open answer.** No yes/no or two-option
   questions ("Is aluminium ferrous or non-ferrous?" is a 50% guess, and
   "Yes" is a technically true non-answer). No fragments whose answer is in
   the question.
5. **The victim answers it correctly** at baseline (temperature 0). You
   cannot attack a fact the victim does not know. The current main victim is
   `qwen3:4b-instruct` (no reasoning); screen against the victim you will
   actually attack.

When in doubt, leave it out: borderline questions have repeatedly produced
fake "successes".

## Allowed edits

Questions come from NQ-Open (real search queries). Only **minimal clarity
edits** are allowed, e.g. "Where in the constitution..." ->
"Where in the American constitution...". Never add hints to the answer and
never change the gold answer. Keep the NQ wording in `original_question`.

## Process

1. **Generate candidates** (context check + victim screen, automatic):

   ```
   python3 utils/download_nq_open.py --n 100 --victim-model qwen3:4b-instruct \
       --judge-model gemma3:12b --output /tmp/candidates.json
   ```

   Questions in `EXCLUDED_IDS` (in that script) are skipped automatically.
   The automatic checks are not enough on their own; step 2 is required.

2. **Review by hand** against the rules above, e.g. in a spreadsheet with
   columns `id, question, answer, original_question`. Delete rejected rows;
   for each rejection, add the id and a one-line reason to `EXCLUDED_IDS` in
   `utils/download_nq_open.py`, so it can never come back.

3. **Re-check the victim** on the final wording (it must still answer
   correctly after any edit), then save the set as JSON: a list of
   `{"id", "question", "answers", "original_question"}`.

## Existing sets

| File | Contents |
|---|---|
| `accepted_questions.json` | 22 hand-curated questions, screened against `qwen3:4b` (reasoning) |
| `accepted_questions_qwen3_4b_instruct.json` | the 16 of those that `qwen3:4b-instruct` answers correctly |
| `nq_open_qwen3_4b_correct.json` | automatic screen only (not hand-reviewed) |
| `nq_open_sample.json` | random NQ-Open sample, context-checked only |
