"""Grader name -> function, and task-level scoring.

Deterministic graders are the midicore.verify checks, so the agent's midi_verify
tool and the grader can never disagree. The LLM rubric judge (T4) plugs in here
later under its own name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from midicore.model import Score
from midicore.verify import CHECKS, CheckError, run_check

GRADERS = set(CHECKS)


@dataclass
class TaskGrade:
    passed: bool
    partial: float
    results: list[dict] = field(default_factory=list)


def grade(graders: list[dict], weights: dict[str, float], pass_rule: str, out: Score, inp: Score) -> TaskGrade:
    """Run every grader; pass = all hard graders pass (or weighted score >= threshold)."""
    results = []
    total_w = weighted = 0.0
    for g in graders:
        name = g["name"]
        try:
            r = run_check(name, out, inp, g.get("args", {})).to_dict()
        except (CheckError, ValueError) as e:
            r = {"passed": False, "score": 0.0, "detail": f"grader error: {e}", "error": True}
        w = float(g.get("weight", weights.get(name, 1.0)))
        r.update(name=name, args=g.get("args", {}), weight=w, hard=g.get("hard", True))
        results.append(r)
        total_w += w
        weighted += w * r["score"]
    partial = weighted / total_w if total_w else 0.0
    if pass_rule == "all":
        passed = all(r["passed"] for r in results if r["hard"])
    elif pass_rule.startswith("threshold:"):
        passed = partial >= float(pass_rule.split(":", 1)[1])
    else:
        raise ValueError(f"unknown pass_rule {pass_rule!r}")
    return TaskGrade(passed, round(partial, 4), results)
