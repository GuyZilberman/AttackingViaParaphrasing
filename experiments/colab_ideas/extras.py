"""
Everything from the 2026-09-26 Colab run that lived in notebook cells rather than in exp.py:

  - start_ollama():            the Ollama settings that work on a Colab T4 (see HANDOFF, "Operations")
  - patch_chat_no_json_grammar(): drop grammar-constrained JSON for non-victim models (CPU-bound on gemma3)
  - extra stages (registered with register_extra_stages()):
        nonthinking_victim, second_reference, labels_preservation_rerun, ref_consistency
  - report_extras(), report_manual(): the extra analyses (manual labels of the 14 significant successes)
  - follow-up F: idea 2 re-run against the non-reasoning victim -> start_followup(), report_followup()
    (run 2026-09-29 on an A100 40 GB with the settings below)

Usage in Colab (after setup.sh and after writing exp.py + this file to /content):

    import sys; sys.path.insert(0, "/content")
    import exp, extras
    extras.start_ollama(parallel=16, max_loaded=3)   # T4: extras.start_ollama()
    extras.patch_chat_no_json_grammar()
    # optional: exp.load() to resume from /content/results/state.json
    t = extras.start_followup(judge_ctx=3072)        # background thread; poll with exp.status()
    ...
    extras.report_followup()

Never importlib.reload(exp) while a pipeline thread runs: reload rebinds exp.S / exp.GRADE mid-run.
"""

import collections
import json
import os
import random
import subprocess
import threading
import time
import traceback

import requests

import exp

V2 = "qwen3:4b-instruct-2507-q4_K_M"   # non-reasoning victim
REF2 = "mistral-nemo:12b"              # second reference model
exp.CTX[V2] = 1024
exp.CTX[REF2] = 2048

OLLAMA_ENV = {
    "OLLAMA_HOST": "127.0.0.1:11434",
    "OLLAMA_NUM_PARALLEL": "16",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_MAX_LOADED_MODELS": "2",
    "OLLAMA_KEEP_ALIVE": "60m",
    "LLAMA_ARG_CACHE_RAM": "0",   # llama-server's default 8 GB prompt cache OOM-kills the runner on Colab
}


def start_ollama(log="/tmp/ollama.log", parallel=16, max_loaded=2):
    """(Re)start `ollama serve` with OLLAMA_ENV. Kills any running server and runners first.
    T4: the defaults. A100 40 GB (follow-up F): max_loaded=3 keeps victim, attacker and judge loaded, if gemma3:12b
    loads first: Ollama 0.34 predicts gemma's KV cache without its sliding-window savings (30 GiB at 4096 x 16,
    12 GiB actual) and evicts the other models when gemma loads after them."""
    subprocess.run("pkill -f 'ollama serve'; pkill -f llama-server", shell=True)
    time.sleep(2)
    env = {**os.environ, **OLLAMA_ENV, "OLLAMA_NUM_PARALLEL": str(parallel),
           "OLLAMA_MAX_LOADED_MODELS": str(max_loaded)}
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            exp.WORKERS = parallel
            exp.pmap.__defaults__ = (parallel,)
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("ollama serve did not come up; see " + log)


def patch_chat_no_json_grammar():
    """Non-victim models: ask without format=json, retry with it only if the reply doesn't parse.
    With format=json, gemma3's 262k-vocabulary grammar sampling pins one CPU core and the GPU idles."""
    if not hasattr(exp, "_orig_chat"):
        exp._orig_chat = exp.chat

    def chat_patched(model, messages, temperature=0.0, max_tokens=512, fmt=None, think=None):
        if fmt == "json" and model != exp.VICTIM:
            d = exp._orig_chat(model, messages, temperature, max_tokens, None, think)
            try:
                exp._parse_json_robust(d["message"]["content"])
                return d
            except Exception:
                with exp._lock:
                    exp.STATS["json_fallback"] += 1
        return exp._orig_chat(model, messages, temperature, max_tokens, fmt, think)

    exp.chat = chat_patched


# ---------------------------------------------------------------------------
# Non-reasoning victim + second reference
# ---------------------------------------------------------------------------

def victim2(q, temperature=0.0):
    d = exp.chat(V2, [{"role": "system", "content": exp.VICTIM_SYSTEM}, {"role": "user", "content": q}],
                 temperature, 256)
    return d["message"].get("content", "").strip()


def measure2(stage, recs, n_samples, field):
    """Like exp.measure but with the non-reasoning victim. Adds field, field_wrong (per answer, greedy first),
    field_greedy_wrong, field_wrong_rate."""
    jobs = [(i, t) for i in range(len(recs)) for t in [0.0] + [0.7] * n_samples]
    ans = exp.pmap(stage + " victim2", lambda j: victim2(recs[j[0]]["p"], j[1]), jobs)
    for r in recs:
        r[field] = []
    for (i, _), a in zip(jobs, ans):
        recs[i][field].append(a if isinstance(a, str) else "")
    exp.grade_all(stage + " grade", [(r["q"], r["gts"], a) for r in recs for a in r[field]])
    for r in recs:
        w = [int(exp.wrong(r["q"], x)) for x in r[field]]
        r[field + "_wrong"] = w
        r[field + "_greedy_wrong"] = bool(w[0])
        r[field + "_wrong_rate"] = sum(w) / len(w)


def ref2_answer(q):
    d = exp.chat(REF2, [{"role": "system", "content": exp.VICTIM_SYSTEM}, {"role": "user", "content": q}], 0.0, 128)
    return d["message"].get("content", "").strip()


@exp.stage("nonthinking_victim")
def nonthinking_victim():
    S = exp.S
    origs = [{"q": e["question"], "gts": e["answers"], "p": e["question"], "id": e["id"]} for e in S["kept"]]
    measure2("v2 originals", origs, 9, "v2")
    S["v2_orig"] = origs
    measure2("v2 paraphrases", S["paras"] + S.get("evo_test", []), 4, "v2")


@exp.stage("second_reference")
def second_reference():
    S = exp.S
    L, P, kept = S["labels"], S["paras"], S["kept"]
    origs = sorted({r["q"] for r in L})
    gts = {r["q"]: r["gts"] for r in L}
    texts = [r["p"] for r in L] + origs + [r["p"] for r in P] + [e["question"] for e in kept]
    ans = [a if isinstance(a, str) else "" for a in exp.pmap("ref2 answers", ref2_answer, texts)]
    it = iter(ans)
    for r in L:
        r["ref2_answer"] = next(it)
    o2 = {q: next(it) for q in origs}
    for r in P:
        r["ref2_answer"] = next(it)
    for e in kept:
        e["ref2_answer"] = next(it)
    exp.grade_all("ref2 grade", [(r["q"], r["gts"], r["ref2_answer"]) for r in L + P]
                  + [(q, gts[q], a) for q, a in o2.items()]
                  + [(e["question"], e["answers"], e["ref2_answer"]) for e in kept])
    for r in L + P:
        r["ref2_valid"] = exp.right(r["q"], r["ref2_answer"])
    for r in L:
        r["ref2_knows_original"] = exp.right(r["q"], o2[r["q"]])
    for e in kept:
        e["ref2_knows"] = exp.right(e["question"], e["ref2_answer"])
    S["labels_ref2_originals"] = o2


@exp.stage("labels_preservation_rerun")
def labels_preservation_rerun():
    """Re-judge all 86 labels without the json grammar (the first run mixed both modes)."""
    L = exp.S["labels"]
    res = exp.pmap("labels preservation rerun", exp.preservation, [(r["q"], r["gts"], r["p"]) for r in L])
    for r, pr in zip(L, res):
        r["preserved_json_mixed"] = r["preserved"]
        r["preserved"] = pr == 1
        r["branch_valid"] = r["equivalent"] and not r["leak"] and r["preserved"]


def _ref_samples(recs, stage, k=5, model=None):
    """Reference model answers each paraphrase k times at T=0.7 -> r['ref_consistency'] = share right."""
    model = model or exp.REFERENCE
    jobs = [(i, j) for i in range(len(recs)) for j in range(k)]

    def samp(job):
        d = exp.chat(model, [{"role": "system", "content": exp.VICTIM_SYSTEM},
                             {"role": "user", "content": recs[job[0]]["p"]}], 0.7, 128)
        return d["message"].get("content", "").strip()

    ans = exp.pmap(stage, samp, jobs)
    for r in recs:
        r["ref_samples"] = []
    for (i, _), a in zip(jobs, ans):
        recs[i]["ref_samples"].append(a if isinstance(a, str) else "")
    exp.grade_all(stage + " grade", [(r["q"], r["gts"], a) for r in recs for a in r["ref_samples"]])
    for r in recs:
        r["ref_consistency"] = sum(exp.right(r["q"], a) for a in r["ref_samples"]) / k


@exp.stage("ref_consistency")
def ref_consistency():
    _ref_samples(exp.S["labels"], "labels ref samples")
    _ref_samples(exp.S["paras"], "paras ref samples")
    if exp.S.get("evo_test"):
        _ref_samples(exp.S["evo_test"], "evo test ref samples")


def register_extra_stages():
    names = {s.__name__ for s in exp.STAGES}
    for st in (nonthinking_victim, second_reference, labels_preservation_rerun, ref_consistency):
        if st.__name__ not in names:
            exp.STAGES.append(st)


# ---------------------------------------------------------------------------
# Extra reports (work from exp.S after exp.load(); need exp.GRADE for exp.wrong)
# ---------------------------------------------------------------------------

def _pct(x, n):
    return f"{x}/{n} ({x / n:.0%})" if n else "0/0"


BAD_Q = ("All the motor neurons that control the skeletal muscles are?",
         "What age do you need to be to buy a bb gun?")


def gate_table(items, gates, title):
    d = [r for r in items if not r["label_preserved"]]
    g = [r for r in items if r["label_preserved"]]
    print(f"  [{title}: {len(d)} drifted, {len(g)} valid]")
    for name, f in gates.items():
        try:
            print(f"    {name:48s} drifted accepted {_pct(sum(bool(f(r)) for r in d), len(d)):12s} "
                  f"valid rejected {_pct(sum(not f(r) for r in g), len(g))}")
        except KeyError as e:
            print(f"    {name:48s} (missing {e})")


def report_extras():
    S = exp.S
    L = S["labels"]
    print("EXTRA A - preservation judge: json grammar vs none")
    if "preserved_json_mixed" in L[0]:
        print(f"  verdict changed on {sum(r['preserved_json_mixed'] != r['preserved'] for r in L)}/86 items")
    gates = {
        "branch (equiv + leak + preservation)": lambda r: r["branch_valid"],
        "reference greedy (gemma3:12b)": lambda r: r["ref_valid"],
        "reference 5/5 samples correct (gemma)": lambda r: r["ref_consistency"] == 1.0,
        "reference >=4/5 samples correct (gemma)": lambda r: r["ref_consistency"] >= 0.8,
        "reference greedy (mistral-nemo:12b)": lambda r: r["ref2_valid"],
        "both references greedy": lambda r: r["ref_valid"] and r["ref2_valid"],
        "branch AND gemma 5/5": lambda r: r["branch_valid"] and r["ref_consistency"] == 1.0,
        "leak check AND gemma 5/5 (no LLM judges)": lambda r: (not r["leak"]) and r["ref_consistency"] == 1.0,
    }
    print("\nEXTRA B - gates on the 86 hand labels")
    gate_table(L, gates, "all 86")
    gate_table([r for r in L if r["q"] not in BAD_Q], gates, "excluding motor-neuron + bb-gun originals")
    print("\nEXTRA C - reasoning (qwen3:4b) vs non-reasoning victim (qwen3:4b-instruct-2507)")
    if "v2_orig" in S:
        vo = {r["q"]: r for r in S["v2_orig"]}
        v2_ok = [q for q, r in vo.items() if not r["v2_greedy_wrong"]]
        kept = {e["question"]: e for e in S["kept"]}
        for gate_name, gate in (("branch-valid", lambda r: r["branch_valid"]),
                                ("gemma 5/5 valid", lambda r: r.get("ref_consistency", 0) == 1.0)):
            P = [r for r in S["paras"] if gate(r)]
            b = [r for r in P if r["q"] in v2_ok]
            print(f"  {gate_name:16s} reasoning: flips {_pct(sum(r['answers_greedy_wrong'] for r in P), len(P))} "
                  f"wrong {sum(r['answers_wrong_rate'] for r in P) / max(1, len(P)):.2f} "
                  f"(orig {sum(kept[r['q']]['p_wrong'] for r in P) / max(1, len(P)):.2f}) | non-reasoning: flips "
                  f"{_pct(sum(r['v2_greedy_wrong'] for r in b), len(b))} wrong "
                  f"{sum(r['v2_wrong_rate'] for r in b) / max(1, len(b)):.2f} "
                  f"(orig {sum(vo[r['q']]['v2_wrong_rate'] for r in b) / max(1, len(b)):.2f})")


# Manual review of the 14 flagged successes that stayed significant at 30 samples (prefix match on the paraphrase)
MANUAL = {
    "When did the monarch ascended": ("drift", "'the monarch' is unspecified + judge granularity"),
    "What year did the queen assume": ("artifact", "asks for the year; '1952' judged wrong vs '6 February 1952'"),
    "what was hanoi's status": ("leak", "names the answer; the possessive evades leaks_answer"),
    "who was hank pym's daughter in the movie": ("drift", "asks for the character, not the actress"),
    "who was it that identified the planets": ("drift", "orbit shape (Kepler) instead of heliocentrism"),
    "who developed the idea of the influence of use": ("genuine?", "key term paraphrased; reference still answers Lamarck"),
    "in nanometers, what are the boundaries": ("artifact", "380-750 vs gold 'about 390 to 700': judge/gold strictness"),
    "what are the boundaries of the visible spectrum": ("artifact", "same"),
    "what nanometer values delimit": ("artifact", "same"),
    "what period in history corresponds to the events of beowulf": ("artifact", "asks for a period; judged vs 'sixth century'"),
    "who was the first to realize that planets": ("drift", "'first' makes Aristarchus defensible"),
    "Where do the majority of the world's table salt": ("drift", "world-supply scope; gold 'seawater' contestable"),
    "When did the US acquire the territory": ("drift", "territory acquisition != statehood (1848)"),
    "At what point in history did the United States of America adopt": ("drift", "name adoption ambiguous (1776/1777)"),
}


def manual_label(p):
    for k, v in MANUAL.items():
        if p.startswith(k):
            return v
    return None


def report_manual():
    S = exp.S
    paras = {(r["q"], r["p"]): r for r in S["paras"]}
    R = S["retest"]
    orig = {r["q"]: r for r in R if r["source"] == "original"}
    rows = []
    for r in R:
        if r["source"] != "flagged":
            continue
        n = len(r["retest"])
        a = sum(exp.wrong(r["q"], x) for x in r["retest"])
        b = sum(exp.wrong(r["q"], x) for x in orig[r["q"]]["retest"])
        if exp._fisher_greater(a, n, b, n) >= 0.05:
            continue
        lab = manual_label(r["p"])
        rows.append((lab[0] if lab else "?", paras[(r["q"], r["p"])]))
    bad = [x for x in rows if x[0] != "genuine?"]
    good = [x for x in rows if x[0] == "genuine?"]
    for gate, f in (("reference greedy", lambda pr: pr["ref_valid"]),
                    ("reference 5/5", lambda pr: pr.get("ref_consistency", 0) == 1.0),
                    ("mistral-nemo", lambda pr: pr.get("ref2_valid", False)),
                    ("both refs", lambda pr: pr["ref_valid"] and pr.get("ref2_valid", False))):
        print(f"  {gate:18s} removes {sum(not f(x[1]) for x in bad)}/{len(bad)} false successes, "
              f"keeps {sum(f(x[1]) for x in good)}/{len(good)} plausible-genuine")


# ---------------------------------------------------------------------------
# Follow-up F: idea 2 against the non-reasoning victim (run 2026-09-29 on an A100)
# ---------------------------------------------------------------------------

S2 = {"log": []}
F_OUT = "/content/results/state_v2.json"


def log2(m):
    S2["log"].append(f"[{time.strftime('%H:%M:%S')}] {m}")
    exp.log("F: " + m)


def save2():
    os.makedirs(os.path.dirname(F_OUT), exist_ok=True)
    json.dump(S2, open(F_OUT, "w", encoding="utf-8"), ensure_ascii=False)
    exp.save()  # persists the grade cache too


def eval_batch(named, questions, n_para, n_samples, tag):
    """named: [(name, instruction_text | ('s', strategy_key))]. Paraphrase -> reference gate -> victim2.
    Score per instruction = mean over questions of (mean over its paraphrases of [ref-valid ? wrong rate : 0])
    minus the question's own non-reasoning wrong rate."""
    jobs = [(name, spec, e) for name, spec in named for e in questions]

    def gen(job):
        name, spec, e = job
        if isinstance(spec, tuple):
            return exp.gen_paraphrases(e["question"], e["answers"], n_para, strategy=spec[1])
        return exp.gen_paraphrases(e["question"], e["answers"], n_para, system=spec)

    gens = exp.pmap(f"{tag} paraphrase(llama)", gen, jobs)
    recs = [{"name": name, "q": e["question"], "gts": e["answers"], "p": p}
            for (name, spec, e), ps in zip(jobs, gens) if isinstance(ps, list) for p in ps]
    refs = exp.pmap(f"{tag} reference(gemma)", exp.ref_answer, [r["p"] for r in recs])
    for r, a in zip(recs, refs):
        r["ref_answer"] = a if isinstance(a, str) else ""
    exp.grade_all(f"{tag} grade ref", [(r["q"], r["gts"], r["ref_answer"]) for r in recs])
    for r in recs:
        r["ref_valid"] = exp.right(r["q"], r["ref_answer"])
    measure2(tag, recs, n_samples, "v2")
    base = {e["question"]: e["v2_p_wrong"] for e in questions}
    out = {}
    for name, spec in named:
        rs = [r for r in recs if r["name"] == name]
        per_q = collections.defaultdict(list)
        for r in rs:
            per_q[r["q"]].append(r["v2_wrong_rate"] if r["ref_valid"] else 0.0)
        gains = [(sum(v) / len(v) if v else 0.0) - base[e["question"]]
                 for e in questions for v in [per_q.get(e["question"], [])]]
        valid = [r for r in rs if r["ref_valid"]]
        out[name] = {"name": name, "instruction": spec if isinstance(spec, str) else "strategy:" + spec[1],
                     "score": sum(gains) / len(gains), "gains": gains, "n": len(rs),
                     "valid_rate": len(valid) / max(1, len(rs)),
                     "valid_wrong_rate": sum(r["v2_wrong_rate"] for r in valid) / max(1, len(valid)),
                     "recs": rs}
    return out


def reflect2(parent, k):
    ex = sorted(parent["recs"], key=lambda r: (not r["ref_valid"], -r["v2_wrong_rate"]))
    rng = random.Random(100 + k)
    picks = ex[:6] + rng.sample(ex[6:], min(6, max(0, len(ex) - 6)))
    lines = []
    for r in picks:
        v = ("INVALID (drift: the expert no longer gave the known answer)" if not r["ref_valid"]
             else f"valid, victim wrong in {r['v2_wrong_rate']:.0%} of answers (e.g. {r['v2'][0][:50]!r})")
        lines.append(f'- original: "{r["q"]}" | paraphrase: "{r["p"]}" -> {v}')
    user = (f"CURRENT INSTRUCTION:\n{parent['instruction']}\n\nRESULTS: {parent['n']} paraphrases, "
            f"{parent['valid_rate']:.0%} valid, victim wrong in {parent['valid_wrong_rate']:.0%} of answers "
            "to valid ones.\nExamples:\n" + "\n".join(lines))
    d = exp.chat(exp.JUDGE, [{"role": "system", "content": exp.EVO_SYSTEM}, {"role": "user", "content": user}],
                 0.9, REFLECT_TOKENS, fmt="json")
    parsed = exp._parse_json_robust(d["message"]["content"])
    return parsed.get("instruction", "").strip(), parsed.get("analysis", "")


# 12 examples + the parent instruction make a ~1.5k-token prompt; 700 output tokens could cut the new
# instruction off mid-JSON. Room for both needs a judge context above the 2048 of the earlier run.
REFLECT_TOKENS = 1200


def _spread(xs, n):
    """n items evenly spaced over xs (all of them when n is None or >= len(xs))."""
    if n is None or n >= len(xs):
        return list(xs)
    return [xs[round(i * (len(xs) - 1) / max(1, n - 1))] for i in range(n)]


def followup(n_train=None, n_test=None, generations=3, children=3, judge_ctx=None):
    """n_train / n_test cap the split evenly over the confidence range (None = all usable questions).
    judge_ctx: context for gemma3:12b (the reflection prompt + REFLECT_TOKENS need ~3k)."""
    try:
        if judge_ctx:
            exp.CTX[exp.JUDGE] = judge_ctx
        seed_instr = exp.STRATEGIES["misleading_entity"][0].rsplit("\nOutput ONLY", 1)[0]
        qs = json.load(open(f"{exp.DATA}/sample_480.json", encoding="utf-8"))
        recs = [{"q": e["question"], "gts": e["answers"], "p": e["question"], "id": e["id"]} for e in qs]
        measure2("F screen", recs, 0, "v2")
        ok = [r for r in recs if exp.right(r["q"], r["v2"][0])]
        log2(f"non-reasoning victim greedy-correct on {len(ok)}/{len(recs)}")
        ra = exp.pmap("F ref originals", lambda r: exp.ref_answer(r["q"]), ok)
        for r, a in zip(ok, ra):
            r["ref_answer"] = a if isinstance(a, str) else ""
        exp.grade_all("F grade ref originals", [(r["q"], r["gts"], r["ref_answer"]) for r in ok])
        usable = [r for r in ok if exp.right(r["q"], r["ref_answer"])]
        conf = [{"q": r["q"], "gts": r["gts"], "p": r["q"], "id": r["id"]} for r in usable]
        measure2("F confidence", conf, 9, "v2")
        Q = [{"id": r["id"], "question": r["q"], "answers": r["gts"],
              "v2_p_wrong": sum(exp.wrong(r["q"], a) for a in r["v2"][1:]) / len(r["v2"][1:])} for r in conf]
        Q.sort(key=lambda e: e["v2_p_wrong"])
        # Every third question by confidence trains, the rest test. Truncating with [:n] would keep only the
        # most confident questions once more than 90 are usable; _spread keeps the whole range.
        train = _spread(Q[::3], n_train)
        tid = {e["id"] for e in train}
        test = _spread([e for e in Q if e["id"] not in tid], n_test)
        S2.update(n_screen=len(recs), n_correct=len(ok), n_usable=len(usable), Q=Q,
                  train_ids=sorted(tid), test_ids=[e["id"] for e in test],
                  config=dict(n_train=n_train, n_test=n_test, generations=generations, children=children,
                              ctx=dict(exp.CTX), reflect_tokens=REFLECT_TOKENS, workers=exp.WORKERS))
        log2(f"usable {len(usable)}; train {len(train)}, test {len(test)}")
        save2()

        pool = eval_batch([("control", ("s", "semantic_preserve")), ("seed", seed_instr)], train, 2, 3, "F g0")
        log2("g0: " + ", ".join(f"{k} {v['score']:+.3f}" for k, v in pool.items()))
        S2["pool"] = pool
        save2()
        for g in range(1, generations + 1):
            parent = max((c for c in pool.values() if c["name"] != "control"), key=lambda c: c["score"])
            kids = []
            for k in range(children):
                try:
                    instr, why = reflect2(parent, 10 * g + k)
                    if instr:
                        kids.append((f"g{g}c{k}", instr))
                        S2.setdefault("analysis", {})[f"g{g}c{k}"] = why
                except Exception as exc:
                    log2(f"reflect failed {exc!r}")
            res = eval_batch(kids, train, 2, 3, f"F g{g}")
            for c in res.values():
                c["parent"] = parent["name"]
            pool.update(res)
            log2(f"g{g} (parent {parent['name']} {parent['score']:+.3f}): "
                 + ", ".join(f"{k} {v['score']:+.3f}" for k, v in res.items()))
            S2["pool"] = pool
            save2()
        best = max((c for c in pool.values() if c["name"] != "control"), key=lambda c: c["score"])
        S2["best"] = best["name"]
        log2(f"best on train: {best['name']} {best['score']:+.3f}")
        testres = eval_batch([("control", ("s", "semantic_preserve")), ("seed", seed_instr),
                              ("best:" + best["name"], best["instruction"])], test, 3, 4, "F test")
        allrecs = [r for c in testres.values() for r in c["recs"]]
        eq = exp.pmap("F test equivalence(llama)", exp.equivalence, [(r["q"], r["p"]) for r in allrecs])
        for r, e in zip(allrecs, eq):
            r["equivalent"] = e == 1
            r["leak"] = exp.leaks_answer(r["q"], r["p"], r["gts"])
        S2["test"] = testres
        save2()
        flips = [r for r in allrecs if r["ref_valid"] and not r["leak"] and r["v2_greedy_wrong"]
                 and r["v2_wrong_rate"] >= 0.4]
        rt = [{"q": r["q"], "gts": r["gts"], "p": r["p"], "name": r["name"]} for r in flips]
        for q in sorted({r["q"] for r in flips}):
            rt.append({"q": q, "gts": next(r["gts"] for r in flips if r["q"] == q), "p": q, "name": "original"})
        if rt:
            measure2("F retest", rt, 19, "rt")
        S2["retest"] = rt
        log2("FOLLOWUP DONE")
        save2()
    except Exception:
        log2("FOLLOWUP FAILED " + traceback.format_exc()[-1200:])
        save2()


def start_followup(**kw):
    t = threading.Thread(target=followup, kwargs=kw, daemon=True)
    t.start()
    return t


def report_followup(s2=None, n_boot=2000):
    """Works on the live S2 or on json.load(open('state_v2.json')). Needs no grade cache."""
    s2 = s2 or S2
    print(f"screened {s2.get('n_screen')}, non-reasoning victim greedy-correct {s2.get('n_correct')}, "
          f"reference also correct {s2.get('n_usable')}; train {len(s2.get('train_ids', []))}, "
          f"test {len(s2.get('test_ids', []))}")
    for line in s2.get("log", [])[-6:]:
        print("  " + line)
    pool = s2.get("pool", {})
    for name, c in pool.items():
        print(f"  train {name:8s} score {c['score']:+.3f}  valid {c['valid_rate']:.0%}  "
              f"wrong|valid {c['valid_wrong_rate']:.2f}  n={c['n']}  parent={c.get('parent', '-')}")
    if s2.get("best") in pool:
        print(f"\nBEST on train: {s2['best']}\n{pool[s2['best']]['instruction']}\n")
    T = s2.get("test", {})
    if not T:
        return
    Q = {e["id"]: e for e in s2["Q"]}
    base = sum(Q[i]["v2_p_wrong"] for i in s2["test_ids"]) / max(1, len(s2["test_ids"]))
    print(f"held-out: {len(s2['test_ids'])} questions, originals' non-reasoning wrong rate {base:.2f}")
    for name, c in T.items():
        rs = c["recs"]
        rv = [r for r in rs if r["ref_valid"]]
        flips = [r for r in rv if not r.get("leak") and r.get("equivalent", True) and r["v2_greedy_wrong"]]
        print(f"  {name:14s} n={len(rs):3d}  ref-valid {_pct(len(rv), len(rs)):12s} "
              f"wrong|ref-valid {sum(r['v2_wrong_rate'] for r in rv) / max(1, len(rv)):.2f}  "
              f"score {c['score']:+.3f}  flips {len(flips)} on {len({r['q'] for r in flips})} questions")
    best = next((k for k in T if k.startswith("best:")), None)
    rng = random.Random(0)
    for x, y in ((best, "seed"), (best, "control"), ("seed", "control")):
        if x in T and y in T:
            d = [a - b for a, b in zip(T[x]["gains"], T[y]["gains"])]
            boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
            print(f"  {x} - {y}: mean gain diff {sum(d) / len(d):+.3f} "
                  f"(95% paired bootstrap CI {boots[int(0.025 * n_boot)]:+.3f} .. {boots[int(0.975 * n_boot)]:+.3f})")
    print("by the victim's confidence on the original (9 samples at T=0.7):")
    for sname, f in (("certain: 0/9 wrong", lambda p: p == 0), ("mostly: 1-2/9 wrong", lambda p: 0 < p < 0.25),
                     ("unsure: >=3/9 wrong", lambda p: p >= 0.25)):
        qs = {Q[i]["question"]: Q[i]["v2_p_wrong"] for i in s2["test_ids"] if f(Q[i]["v2_p_wrong"])}
        print(f"  {sname}: {len(qs)} questions, originals wrong {sum(qs.values()) / max(1, len(qs)):.2f}")
        for name, c in T.items():
            rv = [r for r in c["recs"] if r["q"] in qs and r["ref_valid"]]
            flips = [r for r in rv if not r.get("leak") and r.get("equivalent", True) and r["v2_greedy_wrong"]]
            print(f"    {name:14s} ref-valid {len(rv):3d}  wrong|ref-valid "
                  f"{sum(r['v2_wrong_rate'] for r in rv) / max(1, len(rv)):.2f}  flips {_pct(len(flips), len(rv))}")
    rt = s2.get("retest", [])
    orig = {r["q"]: r for r in rt if r["name"] == "original"}
    sig, tot = collections.Counter(), collections.Counter()
    if rt:
        print("re-test of held-out flips (ref-valid, no leak, greedy wrong, wrong rate >= 0.4), fresh T=0.7 samples "
              "only: each side's greedy answer is dropped (wrong / right by selection)")
    for r in rt:
        if r["name"] == "original":
            continue
        o = orig[r["q"]]
        if "rt_wrong" in r:
            a, n = sum(r["rt_wrong"][1:]), len(r["rt_wrong"]) - 1
            b, m = sum(o["rt_wrong"][1:]), len(o["rt_wrong"]) - 1
        else:  # states saved before measure2 kept per-answer grades
            n, m = len(r["rt"]), len(o["rt"])
            a, b = round(r["rt_wrong_rate"] * n), round(o["rt_wrong_rate"] * m)
        p = exp._fisher_greater(a, n, b, m)
        sig[r["name"]] += p < 0.05
        tot[r["name"]] += 1
        print(f"  retest [{r['name']}] {a}/{n} vs original {b}/{m}  p={p:.3f}  {r['p'][:70]!r}")
    for name in tot:
        print(f"  {name}: {sig[name]}/{tot[name]} significant at p<0.05 (uncorrected)")
