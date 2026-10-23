"""Grid runner: tasks x configs x trials, resumable, one JSONL row per trial."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from midicore import io

from .agent import ROOT, run_agent, trajectory_metrics
from .tasks import Task

RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class Config:
    model: str
    view: str
    verify: bool

    @property
    def id(self) -> str:
        slug = self.model.replace("/", "_").replace(":", "-")
        return f"{slug}__{self.view}__verify{int(self.verify)}"


def run_dir(run_id: str) -> Path:
    return RESULTS_DIR / run_id


def load_rows(run_id: str) -> list[dict]:
    path = run_dir(run_id) / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def run_trial(run_id: str, task: Task, cfg: Config, trial: int, timeout_s: float, max_tool_calls: int, thinking: str | None) -> dict:
    sandbox = run_dir(run_id) / "trials" / task.id / cfg.id / str(trial)
    if sandbox.exists():
        shutil.rmtree(sandbox)  # a partial trial from a crashed run; start clean
    sandbox.mkdir(parents=True)
    shutil.copy(task.input_path, sandbox / "input.mid")
    shutil.copy(task.input_path, sandbox / "work.mid")
    (sandbox / "input.mid").chmod(0o444)

    agent = run_agent(sandbox, task.instruction.strip(), cfg.model, cfg.view, cfg.verify, timeout_s, max_tool_calls, thinking)
    traj = trajectory_metrics(sandbox / "tools.jsonl")

    grade_error = ""
    try:
        out = io.load(sandbox / "work.mid")
        g = task.grade(out)
        passed, partial, grader_results = g.passed, g.partial, g.results
    except Exception as e:  # an unreadable output is a failed trial, not a harness crash
        passed, partial, grader_results, grade_error = False, 0.0, [], f"{type(e).__name__}: {e}"

    return {
        "run_id": run_id,
        "task_id": task.id,
        "tier": task.tier,
        "category": task.category,
        "split": task.split,
        "config": cfg.id,
        "model": cfg.model,
        "view": cfg.view,
        "verify": cfg.verify,
        "trial": trial,
        "passed": passed,
        "partial": partial,
        "grader_results": json.dumps(grader_results),
        **traj,
        "status": agent.status,
        "agent_error": agent.error,
        "grade_error": grade_error,
        "timeout": agent.status == "timeout",
        "input_tokens": agent.input_tokens,
        "output_tokens": agent.output_tokens,
        "cache_read_tokens": agent.cache_read_tokens,
        "cost_usd": agent.cost_usd,
        "n_turns": agent.n_turns,
        "wall_s": agent.wall_s,
        "final_text": agent.final_text,
        "trajectory_path": str(sandbox.relative_to(ROOT)),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def run_grid(
    run_id: str,
    tasks: list[Task],
    configs: list[Config],
    trials: int,
    timeout_s: float = 300,
    max_tool_calls: int = 30,
    workers: int = 1,
    thinking: str | None = None,
    progress=lambda msg: print(msg, flush=True),
) -> list[dict]:
    out_dir = run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "run_id": run_id,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "configs": [asdict(c) | {"id": c.id} for c in configs],
            "tasks": [t.id for t in tasks],
            "trials": trials,
            "timeout_s": timeout_s,
            "max_tool_calls": max_tool_calls,
            "thinking": thinking,
        }, indent=2))

    done = {(r["task_id"], r["config"], r["trial"]) for r in load_rows(run_id)}
    todo = [(t, c, i) for t in tasks for c in configs for i in range(trials) if (t.id, c.id, i) not in done]
    progress(f"run {run_id}: {len(done)} trials already done, {len(todo)} to run")

    lock = threading.Lock()
    results_path = out_dir / "results.jsonl"
    rows: list[dict] = []

    def one(t: Task, c: Config, i: int) -> dict:
        try:
            return run_trial(run_id, t, c, i, timeout_s, max_tool_calls, thinking)
        except Exception:
            # Harness bugs are recorded, not fatal, so the rest of the grid still runs.
            return {"run_id": run_id, "task_id": t.id, "config": c.id, "trial": i, "status": "harness_error", "agent_error": traceback.format_exc()[-2000:]}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, t, c, i): (t, c, i) for t, c, i in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if row.get("status") == "harness_error":
                progress(f"[{n}/{len(todo)}] HARNESS ERROR {row['task_id']} {row['config']} #{row['trial']}\n{row['agent_error']}")
                continue  # not written, so a resume retries it
            with lock:
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
            rows.append(row)
            mark = "PASS" if row["passed"] else "fail"
            extra = f" [{row['status']}: {row['agent_error'][:120]}]" if row["status"] != "ok" else ""
            progress(f"[{n}/{len(todo)}] {mark} {row['task_id']} {row['config']} #{row['trial']} partial={row['partial']:.2f} tools={row['n_tool_calls']} {row['wall_s']}s{extra}")
    write_parquet(run_id)
    return rows


def write_parquet(run_id: str) -> Path | None:
    import pandas as pd

    rows = load_rows(run_id)
    if not rows:
        return None
    path = run_dir(run_id) / "results.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
