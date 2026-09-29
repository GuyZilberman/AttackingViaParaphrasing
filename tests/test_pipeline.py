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
