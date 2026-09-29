"""
Empirical checks of four ideas for the evolutionary paraphrase attack
(branch minimal-edit-paraphrasing of GuyZilberman/AttackingViaParaphrasing).

Reuses the branch's own prompts, parsers and models:
  victim   qwen3:4b      (victim.py system prompt, think=False, reasoning stripped)
  attacker llama3.1:8b   (attackers/llm_paraphraser.py STRATEGIES; also the
                          equivalence judge, as on the branch)
  judge    gemma3:12b    (evaluators/llm_judge.py answer judge, and the
                          structured answer-preservation judge)

Ideas tested
  1. reference gate: a success only counts if a stronger, independent model
     still answers the paraphrase with the gold answer (tested on the
     branch's 86 hand-labelled paraphrases + agreement on new data)
  2. evolve the attacker's INSTRUCTION on train questions, test held-out
  3. stratify by the victim's confidence on the original question
  4. re-test flagged successes with 30 fresh samples (Fisher exact test)

Every stage checkpoints to OUT_DIR/state.json and is skipped when re-run.
"""

import collections
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests

AVP = os.environ.get("AVP_DIR", "/content/avp")
sys.path.insert(0, AVP)
from ollama_client import normalise_text, _parse_json_robust  # noqa: E402
from victim import _SYSTEM_PROMPT as VICTIM_SYSTEM  # noqa: E402
from evaluators.llm_judge import _JUDGE_SYSTEM, _JUDGE_USER_TMPL  # noqa: E402
from utils.question_equivalence_judge import (  # noqa: E402
    QUESTION_EQUIVALENCE_SYSTEM, QUESTION_TEMPLATE, _extract_score,
)
from utils.answer_preservation_judge import (  # noqa: E402
    STRUCTURED_SYSTEM, ANSWER_PRESERVATION_TEMPLATE, leaks_answer, original_overlap,
)
from attackers.llm_paraphraser import STRATEGIES, _extract_string_list  # noqa: E402

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat"
VICTIM, ATTACKER, JUDGE = "qwen3:4b", "llama3.1:8b", "gemma3:12b"
REFERENCE = JUDGE
CTX = {VICTIM: 2304, ATTACKER: 3072, JUDGE: 2048}
WORKERS = int(os.environ.get("WORKERS", "16"))
OUT = os.environ.get("OUT_DIR", "/content/results")
DATA = os.environ.get("DATA_DIR", "/content")

CFG = dict(
    n_screen=480,        # data-folder questions screened with the victim
    n_keep=40,           # victim-correct questions carried forward
    conf_samples=20,     # T=0.7 samples on each original (idea 3)
    n_para=3,            # paraphrases per question per strategy
    fit_samples=4,       # branch fitness: greedy + 4 samples at T=0.7
    retest_samples=30,   # idea 4
    retest_cap=40,
    n_train=12,          # idea 2
    evo_generations=2,
    evo_children=2,
    evo_para=2,
    evo_samples=2,
    seed=0,
)

LOG, STATS, GRADE, S = [], collections.Counter(), {}, {}
PROGRESS = {"stage": None, "done": 0, "total": 0, "t0": time.time(), "t_stage": time.time()}
_lock = threading.Lock()


def log(msg):
    LOG.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def chat(model, messages, temperature=0.0, max_tokens=512, fmt=None, think=None):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_predict": max_tokens,
                           "num_ctx": CTX[model]}}
    if think is not None:
        payload["think"] = think
    if fmt:
        payload["format"] = fmt
    last = None
    for attempt in range(4):
        try:
            r = requests.post(URL, json=payload, timeout=1200)
            if r.status_code == 200:
                d = r.json()
                with _lock:
                    STATS[f"{model} calls"] += 1
                    STATS[f"{model} tokens"] += d.get("eval_count") or 0
                return d
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as exc:  # timeouts, connection resets
            last = exc
        time.sleep(2 + 3 * attempt)
    raise last


def pmap(stage, fn, items, workers=WORKERS):
    with _lock:
        PROGRESS.update(stage=stage, done=0, total=len(items), t_stage=time.time())
    out = [None] * len(items)

    def run(i):
        try:
            out[i] = fn(items[i])
        except Exception as exc:
            out[i] = exc
            with _lock:
                STATS["errors"] += 1
            log(f"error in {stage}: {exc!r}"[:300])
        with _lock:
            PROGRESS["done"] += 1

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, range(len(items))))
    log(f"{stage}: {len(items)} items in {time.time() - PROGRESS['t_stage']:.0f}s")
    return out


# ---------------------------------------------------------------------------
# Victim, grading, generation, validation (branch prompts)
# ---------------------------------------------------------------------------

def victim(q, temperature=0.0):
    d = chat(VICTIM, [{"role": "system", "content": VICTIM_SYSTEM},
                      {"role": "user", "content": q}], temperature, 2048, think=False)
    c = d["message"].get("content", "")
    return (c.rsplit("</think>", 1)[1] if "</think>" in c else c).strip()


def non_answer(a):  # experiment.py::_is_non_answer
    return not " ".join(a.lower().split()).rstrip("?.! ") or len(a.split()) > 40


_ART = {"the", "a", "an"}


def _toks(s):
    return [w for w in normalise_text(s).split() if w not in _ART]


def fast_grade(ans, gts):
    """True if a gold answer appears as a whole-word sequence; else None (ask the LLM)."""
    a = _toks(ans)
    for g in gts:
        t = _toks(g)
        if t and any(a[i:i + len(t)] == t for i in range(len(a) - len(t) + 1)):
            return True
    return None


def gkey(q, a):
    return q + "\x1f" + " ".join(normalise_text(a).split())


def llm_grade(item):
    """evaluators/llm_judge.py: judge per gold answer, correct if any passes."""
    q, gts, ans = item
    for gt in gts:
        if _toks(ans) and _toks(ans) == _toks(gt):
            return True
        msgs = [{"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": _JUDGE_USER_TMPL.format(
                    question_line=f'Question: "{q}"\n', prediction=ans, ground_truth=gt)}]
        d = chat(JUDGE, msgs, 0.0, 512, fmt="json")
        try:
            parsed = _parse_json_robust(d["message"]["content"])
            s = parsed.get("score") if isinstance(parsed, dict) else None
            if s is not None and int(round(float(s))) >= 1:
                return True
        except Exception:
            pass
    return False


def grade_all(stage, triples):
    """Grade (question, golds, answer) triples against the ORIGINAL question; cached."""
    todo = {}
    for q, gts, a in triples:
        if non_answer(a):
            continue
        k = gkey(q, a)
        if k in GRADE or k in todo:
            continue
        if fast_grade(a, gts):
            GRADE[k] = True
        else:
            todo[k] = (q, gts, a)
    keys = list(todo)
    for k, r in zip(keys, pmap(stage, llm_grade, [todo[k] for k in keys])):
        GRADE[k] = r is True


def wrong(q, a):
    return not non_answer(a) and not GRADE.get(gkey(q, a), False)


def right(q, a):
    return not non_answer(a) and GRADE.get(gkey(q, a), False)


JSON_LINE = '\nOutput ONLY this JSON object: {"paraphrases": ["q1", "q2", ...]}'


def gen_paraphrases(q, gts, n, strategy=None, system=None):
    """attackers/llm_paraphraser.py round 1 (no history). `system` overrides the
    strategy's system prompt (evolved instructions) with the adversarial user prompt."""
    if strategy:
        sys_t, usr_t = STRATEGIES[strategy]
    else:
        sys_t, usr_t = system + JSON_LINE, STRATEGIES["misleading_entity"][1]
    user = usr_t.format(question=q, answers_str=", ".join(f'"{a}"' for a in gts), n=n)
    msgs = [{"role": "system", "content": sys_t}, {"role": "user", "content": user}]
    for _ in range(2):
        try:
            d = chat(ATTACKER, msgs, 0.9, 2048, fmt="json")
            ps = _extract_string_list(_parse_json_robust(d["message"]["content"]), q)
            if ps:
                return ps[:n]
        except Exception:
            pass
    return []


def _score01(raw):
    s = _extract_score(raw)
    try:
        return 1 if s is not None and int(round(float(s))) >= 1 else 0
    except Exception:
        return 0


def equivalence(item):
    q, p = item
    if q.strip().lower() == p.strip().lower():
        return 1
    msgs = [{"role": "system", "content": QUESTION_EQUIVALENCE_SYSTEM},
            {"role": "user", "content": QUESTION_TEMPLATE.format(question1=q, question2=p)}]
    return _score01(chat(ATTACKER, msgs, 0.0, 128, fmt="json")["message"]["content"])


def preservation(item):
    q, gts, p = item
    msgs = [{"role": "system", "content": STRUCTURED_SYSTEM},
            {"role": "user", "content": ANSWER_PRESERVATION_TEMPLATE.format(
                original=q, answers=" / ".join(f'"{a}"' for a in gts), candidate=p)}]
    return _score01(chat(JUDGE, msgs, 0.0, 768, fmt="json")["message"]["content"])


def ref_answer(q):
    d = chat(REFERENCE, [{"role": "system", "content": VICTIM_SYSTEM},
                         {"role": "user", "content": q}], 0.0, 128)
    return d["message"].get("content", "").strip()


def validate(stage, recs):
    """recs: dicts with q, gts, p. Adds branch-gate and reference-gate verdicts."""
    eq = pmap(stage + " equivalence(llama)", equivalence, [(r["q"], r["p"]) for r in recs])
    for r, e in zip(recs, eq):
        r["equivalent"] = e == 1
        r["leak"] = leaks_answer(r["q"], r["p"], r["gts"])
        r["overlap"] = round(original_overlap(r["q"], r["p"]), 2)
    pres = pmap(stage + " preservation(gemma)", preservation,
                [(r["q"], r["gts"], r["p"]) for r in recs])
    refs = pmap(stage + " reference answers(gemma)", ref_answer, [r["p"] for r in recs])
    for r, pr, ra in zip(recs, pres, refs):
        r["preserved"] = pr == 1
        r["ref_answer"] = ra if isinstance(ra, str) else ""
    grade_all(stage + " grade reference", [(r["q"], r["gts"], r["ref_answer"]) for r in recs])
    for r in recs:
        r["branch_valid"] = r["equivalent"] and not r["leak"] and r["preserved"]
        r["ref_valid"] = right(r["q"], r["ref_answer"])
    return recs


def measure(stage, recs, n_samples, field="answers"):
    """Victim greedy + n_samples at T=0.7 on rec['p']; graded against rec['q']."""
    jobs = [(i, t) for i in range(len(recs)) for t in [0.0] + [0.7] * n_samples]
    ans = pmap(stage + " victim", lambda j: victim(recs[j[0]]["p"], j[1]), jobs)
    for r in recs:
        r[field] = []
    for (i, _), a in zip(jobs, ans):
        recs[i][field].append(a if isinstance(a, str) else "")
    grade_all(stage + " grade", [(r["q"], r["gts"], a) for r in recs for a in r[field]])
    for r in recs:
        a = r[field]
        r[field + "_greedy_wrong"] = wrong(r["q"], a[0])
        r[field + "_wrong_rate"] = sum(wrong(r["q"], x) for x in a) / len(a)
    return recs


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save():
    tmp = f"{OUT}/state.json.tmp"
    json.dump({"S": S, "GRADE": GRADE, "STATS": dict(STATS), "CFG": CFG}, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False)
    os.replace(tmp, f"{OUT}/state.json")


def load():
    p = f"{OUT}/state.json"
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        S.update(d["S"]); GRADE.update(d["GRADE"]); STATS.update(d["STATS"])


def stage(name):
    def deco(fn):
        def run():
            if name in S.setdefault("done", []):
                log(f"skip {name} (checkpointed)")
                return
            log(f"=== {name} ===")
            t = time.time()
            fn()
            S["done"].append(name)
            S.setdefault("secs", {})[name] = round(time.time() - t)
            save()
        run.__name__ = name
        return run
    return deco


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

@stage("idea1_labels")
def idea1_labels():
    items = json.load(open(f"{AVP}/tests/data/answer_preservation_labels.json", encoding="utf-8"))
    recs = [{"q": it["original"], "gts": it["gold"], "p": it["candidate"],
             "label_preserved": it["preserved"]} for it in items]
    validate("labels", recs)
    origs = sorted({r["q"] for r in recs})
    gts = {r["q"]: r["gts"] for r in recs}
    ra = pmap("labels reference on originals", ref_answer, origs)
    grade_all("labels grade ref originals", [(q, gts[q], a) for q, a in zip(origs, ra)])
    knows = {q: right(q, a) for q, a in zip(origs, ra)}
    for r in recs:
        r["ref_knows_original"] = knows[r["q"]]
    S["labels"] = recs
    S["labels_ref_originals"] = {q: a for q, a in zip(origs, ra)}


@stage("screen")
def screen():
    qs = json.load(open(f"{DATA}/sample_480.json", encoding="utf-8"))[:CFG["n_screen"]]
    ans = pmap("screen victim", lambda e: victim(e["question"]), qs)
    for e, a in zip(qs, ans):
        e["greedy"] = a if isinstance(a, str) else ""
    grade_all("screen grade", [(e["question"], e["answers"], e["greedy"]) for e in qs])
    ok = [e for e in qs if right(e["question"], e["greedy"])]
    log(f"victim correct on {len(ok)}/{len(qs)} screened questions")
    kept = ok[:CFG["n_keep"]]
    ra = pmap("screen reference answers", lambda e: ref_answer(e["question"]), kept)
    for e, a in zip(kept, ra):
        e["ref_answer"] = a if isinstance(a, str) else ""
    grade_all("screen grade reference", [(e["question"], e["answers"], e["ref_answer"]) for e in kept])
    for e in kept:
        e["ref_knows"] = right(e["question"], e["ref_answer"])
    S["screen_n"], S["screen_correct"] = len(qs), len(ok)
    S["screen_all"] = [{"id": e["id"], "question": e["question"], "answers": e["answers"],
                        "greedy": e["greedy"], "correct": right(e["question"], e["greedy"])} for e in qs]
    S["kept"] = kept


@stage("confidence")
def confidence():
    kept = S["kept"]
    recs = [{"q": e["question"], "gts": e["answers"], "p": e["question"]} for e in kept]
    jobs = [(i, 0.7) for i in range(len(recs)) for _ in range(CFG["conf_samples"])]
    ans = pmap("confidence victim", lambda j: victim(recs[j[0]]["p"], j[1]), jobs)
    samples = collections.defaultdict(list)
    for (i, _), a in zip(jobs, ans):
        samples[i].append(a if isinstance(a, str) else "")
    grade_all("confidence grade", [(recs[i]["q"], recs[i]["gts"], a)
                                   for i in samples for a in samples[i]])
    for i, e in enumerate(kept):
        e["samples"] = samples[i]
        e["p_wrong"] = sum(wrong(e["question"], a) for a in samples[i]) / len(samples[i])
        # branch baseline: greedy (correct by construction) + first 4 samples
        e["branch_baseline"] = sum(wrong(e["question"], a) for a in samples[i][:CFG["fit_samples"]]) \
            / (1 + CFG["fit_samples"])


@stage("paraphrase")
def paraphrase():
    jobs = [(e, s) for e in S["kept"] for s in ("semantic_preserve", "misleading_entity")]
    res = pmap("paraphrase(llama)", lambda j: gen_paraphrases(
        j[0]["question"], j[0]["answers"], CFG["n_para"], strategy=j[1]), jobs)
    S["paras"] = [{"q": e["question"], "gts": e["answers"], "p": p, "strategy": s}
                  for (e, s), ps in zip(jobs, res) if isinstance(ps, list) for p in ps]


@stage("validate")
def validate_stage():
    validate("paras", S["paras"])


@stage("attack")
def attack():
    measure("attack", S["paras"], CFG["fit_samples"])
    base = {e["question"]: e for e in S["kept"]}
    for r in S["paras"]:
        b = base[r["q"]]
        r["branch_success"] = r["branch_valid"] and r["answers_greedy_wrong"]
        r["branch_robust"] = r["branch_success"] and \
            r["answers_wrong_rate"] - b["branch_baseline"] >= 0.5


GUY_REPORTED = {
    "What was the united states foreign policy after ww1 started?": (
        ["neutrality"],
        ["What was the United States' posture in its opening response to WW1?",
         "What approach did the United States take in its opening reaction to WW1?"]),
    "When did the first marvel vs capcom come out?": (
        ["1998"],
        ["At what precise point in time was the first Marvel vs Capcom game issued?",
         "What precise year was the first Marvel vs Capcom game made public?",
         "What year marked the debut of Marvel vs Capcom?",
         "Exactly what year was the first Marvel vs Capcom game made available to the public?"]),
}


@stage("retest")
def retest():
    flagged = [r for r in S["paras"] if r["branch_success"]]
    flagged.sort(key=lambda r: -r["answers_wrong_rate"])
    flagged = flagged[:CFG["retest_cap"]]
    recs = [{"q": r["q"], "gts": r["gts"], "p": r["p"], "source": "flagged",
             "fitness5": r["answers_wrong_rate"], "strategy": r["strategy"]} for r in flagged]
    seen = set()
    for r in flagged:  # originals of flagged questions, same fresh sample count
        if r["q"] not in seen:
            seen.add(r["q"])
            recs.append({"q": r["q"], "gts": r["gts"], "p": r["q"], "source": "original"})
    for q, (gts, ps) in GUY_REPORTED.items():
        recs.append({"q": q, "gts": gts, "p": q, "source": "guy_original"})
        recs += [{"q": q, "gts": gts, "p": p, "source": "guy_reported"} for p in ps]
    # data-folder originals reuse their 20 confidence samples: greedy + 9 new + 20 old = 30
    kept = {e["question"]: e for e in S["kept"]}
    fresh = [r for r in recs if r["source"] != "original"]
    olds = [r for r in recs if r["source"] == "original"]
    measure("retest", fresh, CFG["retest_samples"] - 1, field="retest")
    if olds:
        measure("retest originals", olds, CFG["retest_samples"] - 1 - CFG["conf_samples"], field="retest")
    for r in olds:
        r["retest"] = r["retest"] + kept[r["q"]]["samples"]
        r["retest_wrong_rate"] = sum(wrong(r["q"], x) for x in r["retest"]) / len(r["retest"])
    S["retest"] = recs


EVO_SYSTEM = (
    "You improve the SYSTEM INSTRUCTION of an adversarial paraphrase generator used in "
    "AI robustness research. The generator rewrites a factual question so that it keeps "
    "EXACTLY the same meaning and correct answer, but is more likely to make a small "
    "question-answering model (the victim) answer incorrectly.\n"
    "You get the current instruction and its results on training questions. A paraphrase "
    "is INVALID if an independent expert model could no longer answer it with the known "
    "correct answer: that means it changed the question (drift), which never counts. "
    "Study which phrasing patterns made the victim answer wrongly while staying valid, "
    "and which caused drift. Then write an improved instruction that describes GENERAL, "
    "transferable phrasing strategies (it will be used on new, unseen questions). Never "
    "mention specific questions, entities or answers from the examples. Keep the rules "
    "that the paraphrase must be natural, grammatical, a complete question, free of "
    "typos/Unicode tricks, and must not reveal the answer.\n"
    'Output ONLY JSON: {"analysis": "2-3 sentences", "instruction": "the full new system instruction"}'
)


def eval_instruction(name, instr, questions):
    recs = []
    gens = pmap(f"evo {name} paraphrase(llama)", lambda e: gen_paraphrases(
        e["question"], e["answers"], CFG["evo_para"], system=instr), questions)
    for e, ps in zip(questions, gens):
        for p in (ps if isinstance(ps, list) else []):
            recs.append({"q": e["question"], "gts": e["answers"], "p": p})
    refs = pmap(f"evo {name} reference(gemma)", ref_answer, [r["p"] for r in recs])
    for r, a in zip(recs, refs):
        r["ref_answer"] = a if isinstance(a, str) else ""
    grade_all(f"evo {name} grade reference", [(r["q"], r["gts"], r["ref_answer"]) for r in recs])
    for r in recs:
        r["ref_valid"] = right(r["q"], r["ref_answer"])
    measure(f"evo {name}", recs, CFG["evo_samples"])
    base = {e["question"]: e["p_wrong"] for e in questions}
    per_q = collections.defaultdict(list)
    for r in recs:
        per_q[r["q"]].append(r["answers_wrong_rate"] if r["ref_valid"] else 0.0)
    gains = [(sum(v) / len(v) if v else 0.0) - base[q["question"]] for q in questions
             for v in [per_q.get(q["question"], [])]]
    score = sum(gains) / len(gains)
    valid = [r for r in recs if r["ref_valid"]]
    return {"name": name, "instruction": instr, "score": round(score, 4),
            "n": len(recs), "valid_rate": round(len(valid) / max(1, len(recs)), 3),
            "valid_wrong_rate": round(sum(r["answers_wrong_rate"] for r in valid) / max(1, len(valid)), 3),
            "recs": recs}


def reflect(parent, k):
    ex = sorted(parent["recs"], key=lambda r: (not r["ref_valid"], -r["answers_wrong_rate"]))
    rng = random.Random(CFG["seed"] + k)
    picks = ex[:5] + rng.sample(ex[5:], min(5, max(0, len(ex) - 5)))
    lines = []
    for r in picks:
        verdict = ("INVALID (drift: the expert no longer gave the known answer)" if not r["ref_valid"]
                   else f"valid, victim wrong in {r['answers_wrong_rate']:.0%} of answers "
                        f"(e.g. {r['answers'][0][:60]!r})")
        lines.append(f'- original: "{r["q"]}" | paraphrase: "{r["p"]}" -> {verdict}')
    user = (f"CURRENT INSTRUCTION:\n{parent['instruction']}\n\n"
            f"RESULTS: {parent['n']} paraphrases, {parent['valid_rate']:.0%} valid, victim wrong in "
            f"{parent['valid_wrong_rate']:.0%} of answers to valid ones.\nExamples:\n" + "\n".join(lines))
    d = chat(JUDGE, [{"role": "system", "content": EVO_SYSTEM}, {"role": "user", "content": user}],
             0.9, 700, fmt="json")
    parsed = _parse_json_robust(d["message"]["content"])
    return parsed.get("instruction", "").strip(), parsed.get("analysis", "")


def _split():
    usable = [e for e in S["kept"] if e["ref_knows"]]
    usable.sort(key=lambda e: e["p_wrong"])
    train = usable[::3][:CFG["n_train"]]  # every third, spread over confidence
    ids = {e["id"] for e in train}
    return train, [e for e in usable if e["id"] not in ids]


@stage("evolve")
def evolve():
    train, _ = _split()
    seed_instr = STRATEGIES["misleading_entity"][0].rsplit("\nOutput ONLY", 1)[0]
    pool = [eval_instruction("seed", seed_instr, train)]
    log(f"seed score {pool[0]['score']}")
    for g in range(1, CFG["evo_generations"] + 1):
        parent = max(pool, key=lambda c: c["score"])
        for k in range(CFG["evo_children"]):
            try:
                instr, why = reflect(parent, 10 * g + k)
            except Exception as exc:
                log(f"reflect failed: {exc!r}")
                continue
            if not instr:
                continue
            child = eval_instruction(f"g{g}c{k}", instr, train)
            child["analysis"], child["parent"] = why, parent["name"]
            pool.append(child)
            log(f"g{g}c{k} score {child['score']} (parent {parent['name']} {parent['score']})")
    S["evo_pool"] = pool
    S["evo_train_ids"] = [e["id"] for e in train]


@stage("evolve_test")
def evolve_test():
    _, test = _split()
    pool = S["evo_pool"]
    best = max(pool, key=lambda c: c["score"])
    S["evo_best"] = best["name"]
    if best["name"] == "seed":
        log("no child beat the seed on train; evolved == seed")
    gens = pmap("evo test paraphrase(llama)", lambda e: gen_paraphrases(
        e["question"], e["answers"], CFG["n_para"], system=best["instruction"]), test)
    recs = [{"q": e["question"], "gts": e["answers"], "p": p, "strategy": "evolved"}
            for e, ps in zip(test, gens) if isinstance(ps, list) for p in ps]
    validate("evo test", recs)
    measure("evo test", recs, CFG["fit_samples"])
    S["evo_test"] = recs
    S["evo_test_ids"] = [e["id"] for e in test]


STAGES = []  # filled below; run_all runs them in order


def run_all():
    os.makedirs(OUT, exist_ok=True)
    load()
    try:
        for st in STAGES:
            st()
        log("ALL DONE")
    except Exception:
        log("FAILED: " + traceback.format_exc()[-1500:])
        save()


STAGES += [idea1_labels, screen, confidence, paraphrase, validate_stage, attack,
           retest, evolve, evolve_test]


def start():
    t = threading.Thread(target=run_all, daemon=True)
    t.start()
    return t


def status(tail=8):
    el = time.time() - PROGRESS["t_stage"]
    print(f"stage: {PROGRESS['stage']}  {PROGRESS['done']}/{PROGRESS['total']}  ({el:.0f}s in stage, "
          f"{(time.time() - PROGRESS['t0']) / 60:.1f} min total)")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(STATS.items())))
    for line in LOG[-tail:]:
        print("  " + line)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fisher_greater(a, n1, b, n2):
    """One-sided Fisher exact p: paraphrase wrong-rate (a/n1) > original (b/n2)."""
    from math import comb
    k, n = a + b, n1 + n2
    return sum(comb(n1, i) * comb(n2, k - i) for i in range(a, min(k, n1) + 1)) / comb(n, k)


def _pct(x, n):
    return f"{x}/{n} ({x / n:.0%})" if n else "0/0"


def report_idea1():
    L = S["labels"]
    known = [r for r in L if r["ref_knows_original"]]
    print("IDEA 1 - reference gate on the branch's 86 hand-labelled paraphrases")
    print(f"  reference ({REFERENCE}) answers the original correctly for "
          f"{len({r['q'] for r in known})}/{len({r['q'] for r in L})} originals "
          f"({len(known)}/{len(L)} items)")
    for q, a in S["labels_ref_originals"].items():
        print(f"    ref on original: {a[:50]!r:52s} {q}")
    gates = {
        "equivalence only (llama3.1:8b)": lambda r: r["equivalent"],
        "leak + preservation (gemma3:12b)": lambda r: not r["leak"] and r["preserved"],
        "branch: equivalence + leak + preservation": lambda r: r["branch_valid"],
        "reference gate": lambda r: r["ref_valid"],
        "reference gate AND branch": lambda r: r["ref_valid"] and r["branch_valid"],
    }
    for scope, items in (("all 86", L), ("originals the reference knows", known)):
        d = [r for r in items if not r["label_preserved"]]
        g = [r for r in items if r["label_preserved"]]
        print(f"  [{scope}: {len(d)} drifted, {len(g)} valid]")
        for name, f in gates.items():
            print(f"    {name:45s} drifted accepted {_pct(sum(map(f, d)), len(d)):12s} "
                  f"valid rejected {_pct(sum(not f(r) for r in g), len(g))}")
    print("  reference gate mistakes (items whose original it knows):")
    for r in known:
        if r["ref_valid"] != r["label_preserved"]:
            kind = "ACCEPTED DRIFT" if r["ref_valid"] else "REJECTED VALID"
            print(f"    {kind}: {r['p']!r} -> ref {r['ref_answer'][:40]!r} (gold {r['gts'][0]!r})")
    P = S.get("paras", [])
    if P and "ref_valid" in P[0]:
        print("  agreement on new data-folder paraphrases (branch gates vs reference gate):")
        for strat in ("semantic_preserve", "misleading_entity"):
            ps = [r for r in P if r["strategy"] == strat]
            c = collections.Counter((r["branch_valid"], r["ref_valid"]) for r in ps)
            print(f"    {strat:18s} both valid {c[(True, True)]}, branch only {c[(True, False)]}, "
                  f"reference only {c[(False, True)]}, neither {c[(False, False)]}  (n={len(ps)})")


def report_idea3():
    K = {e["question"]: e for e in S["kept"]}
    print("\nIDEA 3 - stratify by the victim's confidence on the original (20 samples at T=0.7)")
    print(f"  screened {S['screen_n']}, victim greedy-correct on {S['screen_correct']}, kept {len(K)}")
    buckets = [("certain: 0/20 wrong", lambda p: p == 0),
               ("mostly: 1-4/20 wrong", lambda p: 0 < p < 0.25),
               ("unsure: >=5/20 wrong", lambda p: p >= 0.25)]
    for name, f in buckets:
        qs = [q for q, e in K.items() if f(e["p_wrong"])]
        print(f"  {name}: {len(qs)} questions")
        for strat in ("semantic_preserve", "misleading_entity"):
            ps = [r for r in S["paras"] if r["q"] in qs and r["strategy"] == strat and r["branch_valid"]]
            if not ps:
                continue
            flips = sum(r["answers_greedy_wrong"] for r in ps)
            robust = sum(r["branch_robust"] for r in ps)
            wr = sum(r["answers_wrong_rate"] for r in ps) / len(ps)
            print(f"    {strat:18s} branch-valid paraphrases {len(ps):3d}  greedy flips {_pct(flips, len(ps)):11s}"
                  f"  mean wrong rate {wr:.2f}  branch 'robust' {robust}")


def report_idea4():
    R = S["retest"]
    orig = {r["q"]: r for r in R if r["source"] in ("original", "guy_original")}
    print("\nIDEA 4 - re-test flagged successes with 30 fresh samples (Fisher exact vs original)")
    for src, title in (("flagged", "branch-flagged successes on data-folder questions"),
                       ("guy_reported", "Guy's 6 reported genuine successes")):
        rs = [r for r in R if r["source"] == src]
        if not rs:
            print(f"  {title}: none")
            continue
        n = len(rs[0]["retest"])
        sig = 0
        print(f"  {title}: {len(rs)}")
        for r in rs:
            o = orig[r["q"]]
            a = sum(wrong(r["q"], x) for x in r["retest"])
            b = sum(wrong(r["q"], x) for x in o["retest"])
            p = _fisher_greater(a, n, b, n)
            sig += p < 0.05
            f5 = f"5-sample {r['fitness5']:.1f} -> " if "fitness5" in r else ""
            print(f"    {f5}paraphrase {a}/{n} wrong vs original {b}/{n}  p={p:.3f}  {r['p'][:70]!r}")
        print(f"    significant at p<0.05: {sig}/{len(rs)} (uncorrected)")


def report_idea2():
    pool = S["evo_pool"]
    print("\nIDEA 2 - evolve the attacker's instruction (train) and test on held-out questions")
    for c in pool:
        print(f"  train {c['name']:5s} score {c['score']:+.3f}  valid {c['valid_rate']:.0%}  "
              f"wrong among valid {c['valid_wrong_rate']:.2f}  (n={c['n']})")
    best = max(pool, key=lambda c: c["score"])
    print(f"  best: {best['name']}\n  --- instruction ---\n{best['instruction']}\n  ---")
    test = set(S["evo_test_ids"])
    tq = {e["question"]: e for e in S["kept"] if e["id"] in test}
    sets = {"control (semantic_preserve)": [r for r in S["paras"] if r["q"] in tq and r["strategy"] == "semantic_preserve"],
            "seed (branch misleading_entity)": [r for r in S["paras"] if r["q"] in tq and r["strategy"] == "misleading_entity"],
            f"evolved ({best['name']})": S["evo_test"]}
    base = sum(e["p_wrong"] for e in tq.values()) / max(1, len(tq))
    print(f"  held-out: {len(tq)} questions, mean original wrong rate {base:.2f}")
    for name, rs in sets.items():
        rv = [r for r in rs if r["ref_valid"]]
        both = [r for r in rv if r["branch_valid"]]
        flips = [r for r in both if r["answers_greedy_wrong"]]
        wr = sum(r["answers_wrong_rate"] for r in rv) / max(1, len(rv))
        print(f"    {name:32s} n={len(rs):3d}  ref-valid {_pct(len(rv), len(rs)):11s} branch-valid "
              f"{_pct(sum(r['branch_valid'] for r in rs), len(rs)):11s} wrong|ref-valid {wr:.2f}  "
              f"flips (both gates) {len(flips)} on {len({r['q'] for r in flips})} questions")


def report():
    for f in (report_idea1, report_idea3, report_idea4, report_idea2):
        try:
            f()
        except Exception:
            print(f"{f.__name__} failed: {traceback.format_exc()[-600:]}")
    print("\nstage seconds:", S.get("secs"), "\nstats:", dict(STATS))
