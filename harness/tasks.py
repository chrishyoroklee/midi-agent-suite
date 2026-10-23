"""Task YAML schema, loading, and the check-tasks sanity check."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from graders.registry import GRADERS, TaskGrade, grade
from midicore import io
from midicore.model import Score
from midicore.ops import apply_ops

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


class GraderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    args: dict = Field(default_factory=dict)
    weight: float | None = None
    hard: bool = True

    @field_validator("name")
    @classmethod
    def _known(cls, v):
        if v not in GRADERS:
            raise ValueError(f"unknown grader {v!r}; known: {', '.join(sorted(GRADERS))}")
        return v


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tier: int = Field(ge=1, le=4)
    category: str
    input: str
    instruction: str
    graders: list[GraderSpec]
    weights: dict[str, float] = Field(default_factory=dict)
    pass_rule: str = "all"
    split: str = Field("dev", pattern="^(dev|test)$")
    notes: str = ""
    gold_ops: list[dict] | None = None
    gold: str | None = None  # path to a gold .mid, e.g. the clean original of a repair task
    path: Path | None = None

    @property
    def input_path(self) -> Path:
        return TASKS_DIR / self.input

    def load_input(self) -> Score:
        return io.load(self.input_path)

    def gold_score(self) -> Score | None:
        if self.gold:
            return io.load(TASKS_DIR / self.gold)
        if self.gold_ops is not None:
            return apply_ops(self.load_input(), self.gold_ops)
        return None

    def grade(self, out: Score, inp: Score | None = None) -> TaskGrade:
        inp = inp if inp is not None else self.load_input()
        specs = [g.model_dump(exclude_none=True) for g in self.graders]
        return grade(specs, self.weights, self.pass_rule, out, inp)


def load_tasks(selector: str | None = None, split: str | None = None) -> list[Task]:
    """selector: None (all), a tier dir like 't1', or comma-separated task ids / glob patterns."""
    tasks = []
    for p in sorted(TASKS_DIR.glob("t[1-4]/*.yaml")):
        data = yaml.safe_load(p.read_text())
        t = Task(**data, path=p)
        if p.stem != t.id:
            raise ValueError(f"{p}: file name must match id {t.id!r}")
        tasks.append(t)
    if selector:
        import fnmatch

        pats = [s.strip() for s in selector.split(",") if s.strip()]
        tasks = [t for t in tasks if any(t.path.parent.name == s or fnmatch.fnmatch(t.id, s) for s in pats)]
    if split:
        tasks = [t for t in tasks if t.split == split]
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return tasks


def check_task(task: Task) -> list[str]:
    """Problems with a task: it must fail on its unmodified input and pass on its gold."""
    problems = []
    if not task.input_path.exists():
        return [f"input not found: {task.input}"]
    inp = task.load_input()
    g_input = task.grade(inp, inp)
    if g_input.passed:
        problems.append("passes on the unmodified input")
    errors = [r for r in g_input.results if r.get("error")]
    problems += [f"grader error on input: {r['name']}: {r['detail']}" for r in errors]
    gold = task.gold_score()
    if gold is None:
        problems.append("no gold solution (gold_ops or gold)")
    else:
        g_gold = task.grade(gold, inp)
        if not g_gold.passed:
            failing = [f"{r['name']}: {r['detail']}" for r in g_gold.results if not r["passed"]]
            problems.append("gold does not pass: " + "; ".join(failing))
    return problems
