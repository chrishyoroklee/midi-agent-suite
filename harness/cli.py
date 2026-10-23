"""midibench CLI: run, check-tasks, report, list-tasks."""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from .run import Config, run_grid
from .tasks import check_task, load_tasks


def cmd_check_tasks(a) -> int:
    tasks = load_tasks(a.tasks, a.split)
    bad = 0
    for t in tasks:
        problems = check_task(t)
        if problems:
            bad += 1
            print(f"FAIL {t.id}")
            for p in problems:
                print(f"     - {p}")
        else:
            print(f"ok   {t.id}")
    print(f"\n{len(tasks) - bad}/{len(tasks)} tasks fail on their input and pass on their gold solution")
    return 1 if bad else 0


def cmd_list(a) -> int:
    for t in load_tasks(a.tasks, a.split):
        print(f"{t.id:24} tier {t.tier}  {t.split:4}  {t.category:12} {t.instruction.strip()[:80]}")
    return 0


def cmd_run(a) -> int:
    tasks = load_tasks(a.tasks, a.split)
    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 1
    configs = [Config(m, v, bool(int(x))) for m in a.model for v in a.views.split(",") for x in a.verify.split(",")]
    run_id = a.run_id or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = run_grid(run_id, tasks, configs, a.trials, a.timeout, a.max_tool_calls, a.workers, a.thinking)
    if rows or a.report:
        from .report import build_report

        print()
        print(build_report(run_id))
    return 0


def cmd_report(a) -> int:
    from .report import build_report

    print(build_report(a.run_id))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="midibench")
    sub = p.add_subparsers(dest="cmd", required=True)

    def task_args(sp):
        sp.add_argument("--tasks", help="tier dir (t1), task ids or globs, comma-separated; default all")
        sp.add_argument("--split", choices=["dev", "test"])

    c = sub.add_parser("check-tasks", help="every task must fail on its input and pass on its gold")
    task_args(c)
    c.set_defaults(fn=cmd_check_tasks)

    ls = sub.add_parser("list-tasks")
    task_args(ls)
    ls.set_defaults(fn=cmd_list)

    r = sub.add_parser("run", help="run the agent over tasks x configs x trials")
    task_args(r)
    r.add_argument("--model", action="append", required=True, help="pi model id, e.g. anthropic/claude-sonnet-5 or ollama/llama3.1:8b; repeatable")
    r.add_argument("--views", default="notes", help="comma-separated: raw,notes,musical")
    r.add_argument("--verify", default="0", help="comma-separated: 0,1")
    r.add_argument("--trials", type=int, default=3)
    r.add_argument("--run-id", help="reuse to resume a run")
    r.add_argument("--timeout", type=float, default=300)
    r.add_argument("--max-tool-calls", type=int, default=30)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--thinking", help="pi thinking level, e.g. off, low, medium")
    r.add_argument("--report", action="store_true", help="print the report even if nothing new ran")
    r.set_defaults(fn=cmd_run)

    rep = sub.add_parser("report")
    rep.add_argument("run_id")
    rep.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
