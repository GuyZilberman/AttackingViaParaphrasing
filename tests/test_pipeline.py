"""
Offline tests of the attack pipeline's checks and measurements, with fake
models in place of Ollama (the other scripts in tests/ need a live server).

Run from the project root:
    python -m pytest tests/test_pipeline.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import experiment
from config import ExperimentConfig
from evaluators.base import BaseEvaluator, EvalResult
from utils.answer_preservation_judge import leaks_answer

QUESTION = "who wrote the novel moby dick"
GOLD = ["Herman Melville"]


class FakeAttacker:
    """Returns one fixed candidate list per round."""

    def __init__(self, rounds):
        self.rounds = list(rounds)

    def generate_paraphrases(self, question, answers, n=10, history=None, parents=None):
        return self.rounds.pop(0) if self.rounds else []


class FakeVictim:
    """Answers each question with a fixed reply, or cycles through a list of replies."""

    def __init__(self, replies=None, default="Herman Melville"):
        self.replies = replies or {}
        self.default = default
        self.calls = {}

    def answer(self, question, temperature=None):
        reply = self.replies.get(question, self.default)
        if isinstance(reply, list):
            i = self.calls.get(question, 0)
            self.calls[question] = i + 1
            return reply[i % len(reply)]
        return reply


class FakeJudge(BaseEvaluator):
    @property
    def name(self):
        return "llm_judge"

    def evaluate(self, prediction, ground_truths, question=None):
        ok = prediction.strip().lower() in {g.lower() for g in ground_truths}
        return EvalResult(correct=ok, score=float(ok))


@pytest.fixture
def judges(monkeypatch):
    """Replace the two LLM judges with ones that accept everything and log their calls."""
    calls = {"equivalence": [], "preservation": []}

    def fake_equivalent(question1, question2, model_name=None, client=None):
        calls["equivalence"].append(question2)
        return 1

    def fake_preservation(original, candidate, answers, model_name=None, client=None):
        calls["preservation"].append(candidate)
        return 1, ""

    monkeypatch.setattr(experiment, "questions_equivalent", fake_equivalent)
    monkeypatch.setattr(experiment, "answer_preservation_verdict", fake_preservation)
    return calls


def attack(cfg, attacker, victim, question=QUESTION, gts=GOLD):
    return experiment._iterative_attack(
        cfg, None, attacker, victim, [FakeJudge()], question, gts, {"llm_judge": True},
    )


def test_free_checks_run_before_the_equivalence_judge(judges):
    cfg = ExperimentConfig(max_rounds=1, fitness_samples=0)
    attacker = FakeAttacker([[
        "what is the capital of france",
        "who wrote moby dick, herman melville?",
        "who was the author of the novel moby dick",
    ]])
    rounds, queried, _ = attack(cfg, attacker, FakeVictim())
    rejected_by = {c["paraphrase"]: c.get("rejected_by") for c in rounds[0]["candidates"]}
    assert rejected_by["what is the capital of france"] == "too_different"
    assert rejected_by["who wrote moby dick, herman melville?"] == "answer_leak"
    assert judges["equivalence"] == ["who was the author of the novel moby dick"]
    assert [r["paraphrase"] for r in queried] == ["who was the author of the novel moby dick"]


@pytest.mark.parametrize("candidate, leaks", [
    # The paraphrase that scored 27/30 "wrong" in the 2026-09-26 run by naming the answer
    ("what was hanoi's status as a city in north vietnam", True),
    ("What was Hanoi’s status as a city in North Vietnam?", True),
    ("which city served as the seat of government of north vietnam", False),
    ("what was the capital of north vietnam's government", False),
])
def test_leaks_answer_sees_through_possessives(candidate, leaks):
    assert leaks_answer("what was the capital of north vietnam", candidate, ["Hanoi"]) is leaks


@pytest.mark.parametrize("wrong_rate, group", [
    (None, None), (0.0, "certain"), (0.05, "mostly"), (0.2, "mostly"), (0.25, "unsure"), (1.0, "unsure"),
])
def test_confidence_group_boundaries(wrong_rate, group):
    assert experiment.confidence_group(wrong_rate) == group


def test_confidence_is_measured_on_fresh_samples_of_the_original(judges):
    cfg = ExperimentConfig(max_rounds=1, fitness_samples=0, confidence_samples=4)
    victim = FakeVictim({QUESTION: ["Herman Melville", "Mark Twain", "Herman Melville", "Herman Melville"]})
    _, _, baseline = attack(cfg, FakeAttacker([[]]), victim)
    assert len(baseline["confidence_samples"]) == 4
    assert baseline["confidence_wrong_rate"] == 0.25


def test_summary_groups_questions_by_confidence(judges):
    cfg = ExperimentConfig(max_rounds=1, fitness_samples=0, confidence_samples=4)
    sure = {"id": "q1", "question": QUESTION, "answers": GOLD}
    unsure = {"id": "q2", "question": "who wrote the novel moby-dick in 1851", "answers": GOLD}
    paraphrase = "who was the author of the novel moby dick"
    victim = FakeVictim({
        unsure["question"]: ["Herman Melville", "Mark Twain"],  # greedy right, then 2 of 4 samples wrong
        paraphrase: "Mark Twain",
    })
    records = [
        experiment._run_question(cfg, sure, None, victim, FakeAttacker([[paraphrase]]),
                                 [FakeJudge()], ["llm_judge"]),
        experiment._run_question(cfg, unsure, None, victim, FakeAttacker([[]]),
                                 [FakeJudge()], ["llm_judge"]),
    ]
    assert [r["confidence_group"] for r in records] == ["certain", "unsure"]
    groups = experiment._summarize(cfg, records, ["llm_judge"])["by_confidence_group"]
    assert groups == {
        "certain": {"n_questions": 1, "n_queried": 1, "n_successes": 1, "n_questions_with_success": 1,
                    "n_confirmed_successes": 1, "n_questions_with_confirmed_success": 1},
        "unsure": {"n_questions": 1, "n_queried": 0, "n_successes": 0, "n_questions_with_success": 0,
                   "n_confirmed_successes": 0, "n_questions_with_confirmed_success": 0},
    }


def test_fisher_greater():
    assert experiment.fisher_greater(3, 5, 0, 5) == pytest.approx(10 / 120)  # C(5,3) / C(10,3)
    assert experiment.fisher_greater(0, 19, 0, 19) == 1.0
    assert experiment.fisher_greater(5, 30, 12, 30) > 0.9  # the Marvel vs Capcom "wins": noise
    assert experiment.fisher_greater(18, 30, 0, 30) < 0.001  # the WWI paraphrase: real


def test_retest_confirms_only_successes_that_hold_up_on_fresh_samples(judges):
    cfg = ExperimentConfig(max_rounds=1, fitness_samples=0, confidence_samples=5, retest_samples=20)
    holds_up = "who was the author of the novel moby dick"
    lucky = "who penned the novel moby dick"
    victim = FakeVictim({
        holds_up: "Mark Twain",
        lucky: ["Mark Twain"] + ["Herman Melville"] * 30,  # wrong only on the greedy answer
    })
    _, queried, _ = attack(cfg, FakeAttacker([[holds_up, lucky]]), victim)
    by_text = {r["paraphrase"]: r for r in queried}
    assert all(r["attack_success"]["llm_judge"] for r in queried)
    assert by_text[holds_up]["confirmed_success"] is True
    assert by_text[holds_up]["retest"]["n_wrong"] == 20
    assert by_text[lucky]["confirmed_success"] is False
    assert by_text[lucky]["retest"]["n_wrong"] == 0
    assert by_text[lucky]["retest"]["original_n"] == 5


def test_retest_off_leaves_successes_unconfirmed(judges):
    cfg = ExperimentConfig(max_rounds=1, fitness_samples=0, confidence_samples=5, retest_samples=0)
    paraphrase = "who was the author of the novel moby dick"
    _, queried, _ = attack(cfg, FakeAttacker([[paraphrase]]), FakeVictim({paraphrase: "Mark Twain"}))
    assert queried[0]["confirmed_success"] is None and "retest" not in queried[0]


def test_retest_needs_confidence_samples():
    with pytest.raises(ValueError, match="confidence_samples"):
        ExperimentConfig(confidence_samples=0, retest_samples=20).validate()
