"""pass@k, pass^k, partial credit, bootstrap CIs over tasks, paired comparisons.

Every metric is computed per task first and then averaged over tasks, so a
task with more trials does not weigh more. CIs bootstrap over tasks.
"""

from __future__ import annotations

import random
from math import comb

import pandas as pd


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k sampled trials passes), from c passes in n trials."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(all k sampled trials pass): the reliability metric."""
    if k > n:
        return float("nan")
    return comb(c, k) / comb(n, k)


def per_task(df: pd.DataFrame, by: list[str], ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """One row per (group, task) with n, c and each metric."""
    rows = []
    for key, g in df.groupby(by + ["task_id"], dropna=False):
        n, c = len(g), int(g["passed"].sum())
        row = dict(zip(by + ["task_id"], key if isinstance(key, tuple) else (key,)))
        row.update(n=n, c=c, partial=g["partial"].mean(), tier=g["tier"].iloc[0])
        for k in ks:
            row[f"pass@{k}"] = pass_at_k(n, c, k)
            row[f"pass^{k}"] = pass_hat_k(n, c, k)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: list[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    vals = [v for v in values if v == v]  # drop NaN
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(iters))
    return means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters) - 1]


def summarize(df: pd.DataFrame, by: list[str], ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """Task-averaged metrics per group, with a bootstrap CI on pass@1."""
    pt = per_task(df, by, ks)
    out = []
    for key, g in pt.groupby(by, dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["tasks"] = len(g)
        row["trials"] = int(g["n"].sum())
        for k in ks:
            for m in (f"pass@{k}", f"pass^{k}"):
                if g[m].notna().any():
                    row[m] = g[m].mean()
        lo, hi = bootstrap_ci(g["pass@1"].tolist())
        row["pass@1_ci"] = (lo, hi)
        row["partial"] = g["partial"].mean()
        out.append(row)
    return pd.DataFrame(out)


def paired_diff(df: pd.DataFrame, condition: str, a, b, metric: str = "pass@1", ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """Mean per-task difference (b - a) in a metric between two values of a condition column."""
    pt = per_task(df, [condition], ks)
    wide = pt.pivot(index="task_id", columns=condition, values=metric).dropna()
    if a not in wide or b not in wide:
        return {"tasks": 0}
    d = (wide[b] - wide[a]).tolist()
    lo, hi = bootstrap_ci(d)
    return {"tasks": len(d), "mean_diff": sum(d) / len(d) if d else float("nan"), "ci": (lo, hi)}
