"""python -m midicore read|analyze|edit|verify ... ; always prints {"ok": ..., "result" | "error": ...}."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from . import io
from .analysis import analyze
from .ops import OpError, apply_ops, diff
from .verify import run_checks
from .views import VIEWS, parse_bars, render, resolve_tracks


class CliError(Exception):
    pass


def _load(path: str):
    p = Path(path)
    if not p.exists():
        raise CliError(f"file not found: {path}")
    try:
        return io.load(p)
    except Exception as e:  # mido raises a variety of errors on bad files
        raise CliError(f"could not read MIDI file {path}: {e}") from None


def _json_arg(text: str, what: str):
    if text.startswith("@"):
        text = Path(text[1:]).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise CliError(f"{what} is not valid JSON: {e}") from None


def cmd_read(a) -> dict:
    score = _load(a.file)
    tracks = resolve_tracks(score, a.tracks.split(",") if a.tracks else None)
    return {"view": a.view, "text": render(score, a.view, bars=parse_bars(a.bars), tracks=tracks)}


def cmd_analyze(a) -> dict:
    return analyze(_load(a.file))


def cmd_edit(a) -> dict:
    path = Path(a.file)
    if path.name in (a.protect or "").split(","):
        raise CliError(f"{path.name} is read-only; edit the working file instead")
    before = _load(a.file)
    ops = _json_arg(a.ops, "ops")
    if isinstance(ops, dict):
        ops = [ops]
    try:
        after = apply_ops(before, ops)
    except OpError as e:
        return {"__error__": e.to_dict()}
    # Write to a temp file and rename so a crash never leaves a half-written file.
    fd, tmp = tempfile.mkstemp(suffix=".mid", dir=path.parent)
    os.close(fd)
    try:
        io.save(after, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"applied": len(ops), "diff": diff(before, after)}


def cmd_verify(a) -> dict:
    out = _load(a.file)
    ref = _load(a.reference) if a.reference and Path(a.reference).exists() else None
    constraints = _json_arg(a.constraints, "constraints")
    if isinstance(constraints, dict):
        constraints = [constraints]
    results = run_checks(constraints, out, ref)
    return {"all_passed": all(r["passed"] for r in results), "results": results}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="midicore")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("read")
    r.add_argument("--file", required=True)
    r.add_argument("--view", choices=list(VIEWS), default="notes")
    r.add_argument("--bars")
    r.add_argument("--tracks")
    r.set_defaults(fn=cmd_read)

    an = sub.add_parser("analyze")
    an.add_argument("--file", required=True)
    an.set_defaults(fn=cmd_analyze)

    e = sub.add_parser("edit")
    e.add_argument("--file", required=True)
    e.add_argument("--ops", required=True, help="JSON array of ops, or @path")
    e.add_argument("--protect", default="", help="comma-separated file names that may not be edited")
    e.set_defaults(fn=cmd_edit)

    v = sub.add_parser("verify")
    v.add_argument("--file", required=True)
    v.add_argument("--constraints", required=True, help="JSON array of {name, args}, or @path")
    v.add_argument("--reference", help="original file, for checks that compare against it")
    v.set_defaults(fn=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.fn(args)
        if isinstance(result, dict) and "__error__" in result:
            out = {"ok": False, "error": result["__error__"]}
        else:
            out = {"ok": True, "result": result}
    except (CliError, ValueError) as e:
        out = {"ok": False, "error": {"message": str(e)}}
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
