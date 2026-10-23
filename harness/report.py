"""Markdown results tables for a run."""

from __future__ import annotations

import json

import pandas as pd

from .metrics import summarize
from .run import load_rows, run_dir


def _fmt(x) -> str:
    if isinstance(x, tuple):
        return "-" if any(v != v for v in x) else f"[{x[0]:.2f}, {x[1]:.2f}]"
    if isinstance(x, float):
        return "-" if x != x else f"{x:.2f}"
    return str(x)


def _table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_report(run_id: str) -> str:
    rows = load_rows(run_id)
    if not rows:
        return f"No results for run {run_id}."
    df = pd.DataFrame(rows)
    trials_per_cell = int(df.groupby(["task_id", "config"]).size().max())
    ks = tuple(k for k in (1, 3, 5) if k <= trials_per_cell)
    metric_cols = [m for k in ks for m in (f"pass@{k}", f"pass^{k}")]
    out = [f"# MIDIBench results: {run_id}", ""]
    out.append(f"{len(df)} trials, {df['task_id'].nunique()} tasks, {df['config'].nunique()} configs, up to {trials_per_cell} trials per cell.")
    status = df["status"].value_counts().to_dict()
    if set(status) != {"ok"}:
        out.append(f"Agent status counts: {status}. Non-ok trials count as failures.")
    out += ["", "## By config", ""]
    s = summarize(df, ["model", "view", "verify"], ks)
    cost = df.groupby(["model", "view", "verify"]).apply(
        lambda g: pd.Series({
            "tools/trial": g["n_tool_calls"].mean(),
            "tokens/trial": (g["input_tokens"] + g["output_tokens"]).mean(),
            "cost/success": g["cost_usd"].sum() / max(int(g["passed"].sum()), 1),
        }), include_groups=False,
    ).reset_index()
    s = s.merge(cost, on=["model", "view", "verify"])
    out.append(_table(s, ["model", "view", "verify", "tasks", "trials"] + metric_cols + ["pass@1_ci", "partial", "tools/trial", "tokens/trial", "cost/success"]))

    out += ["", "## By tier and config", ""]
    out.append(_table(summarize(df, ["tier", "model", "view", "verify"], ks), ["tier", "model", "view", "verify", "tasks"] + metric_cols + ["partial"]))

    out += ["", "## Per task", ""]
    pt = df.groupby(["task_id", "config"]).agg(passed=("passed", "sum"), trials=("passed", "size"), partial=("partial", "mean"), tools=("n_tool_calls", "mean")).reset_index()
    out.append(_table(pt, ["task_id", "config", "passed", "trials", "partial", "tools"]))

    fails = df[~df["passed"].astype(bool)]
    if len(fails):
        out += ["", "## Failed graders", ""]
        counts: dict[str, int] = {}
        for gr in fails["grader_results"]:
            for r in json.loads(gr or "[]"):
                if not r["passed"]:
                    counts[r["name"]] = counts.get(r["name"], 0) + 1
        out += [f"- {name}: {n}" for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    text = "\n".join(out) + "\n"
    (run_dir(run_id) / "report.md").write_text(text)
    return text
