"""Run one Pi agent trial headlessly. The only module that knows how Pi is invoked.

Swapping the agent means reimplementing run_agent(); the harness only needs the
final work.mid, the extension's JSONL log and the AgentResult.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PI_EXT = ROOT / "pi-ext" / "index.ts"
LOCAL_MODELS_EXT = ROOT / "pi-ext" / "local-models.ts"

# Fixed across every condition (spec 6.4): no musical hints, so views stay the only variable.
SYSTEM_PROMPT = (
    "You are editing a MIDI file using the midi_* tools. The file to edit is work.mid in the "
    "current directory; input.mid is a read-only copy of the original. Complete the user's "
    "request by calling the tools. When you are finished, stop calling tools and reply with a "
    "one-line summary of what you changed."
)


@dataclass
class AgentResult:
    status: str  # ok | timeout | agent_error | crash
    error: str = ""
    wall_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    n_turns: int = 0
    final_text: str = ""
    extra: dict = field(default_factory=dict)


def pi_command(model: str, instruction: str, thinking: str | None = None) -> list[str]:
    pi = shutil.which("pi")
    if not pi:
        raise RuntimeError("pi not found on PATH; install @earendil-works/pi-coding-agent")
    cmd = [
        pi, "-p", "--mode", "json",
        "--no-session", "--no-builtin-tools", "--no-extensions", "--no-skills",
        "--no-context-files", "--no-prompt-templates", "--no-themes", "--no-approve", "--offline",
        "-e", str(PI_EXT),
    ]
    if model.startswith("ollama/"):
        cmd += ["-e", str(LOCAL_MODELS_EXT)]
    cmd += ["--model", model, "--system-prompt", SYSTEM_PROMPT]
    if thinking:
        cmd += ["--thinking", thinking]
    cmd.append(instruction)
    return cmd


def run_agent(
    sandbox: Path,
    instruction: str,
    model: str,
    view: str,
    verify: bool,
    timeout_s: float = 300,
    max_tool_calls: int = 30,
    thinking: str | None = None,
) -> AgentResult:
    env = dict(os.environ)
    env.update(
        MIDIBENCH_VIEW=view,
        MIDIBENCH_VERIFY="1" if verify else "0",
        MIDIBENCH_LOG=str(sandbox / "tools.jsonl"),
        MIDIBENCH_PYTHON=sys.executable,
        MIDIBENCH_MAX_TOOL_CALLS=str(max_tool_calls),
        MIDIBENCH_PROTECT="input.mid",
        MIDIBENCH_REFERENCE="input.mid",
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    if model.startswith("ollama/"):
        env["MIDIBENCH_OLLAMA_MODEL"] = model.split("/", 1)[1]

    events_path = sandbox / "pi_events.jsonl"
    t0 = time.monotonic()
    with open(events_path, "w") as out, open(sandbox / "pi_stderr.txt", "w") as err:
        # stdin must be closed: with an open pipe, pi -p waits for piped input.
        proc = subprocess.Popen(pi_command(model, instruction, thinking), cwd=sandbox, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        try:
            code = proc.wait(timeout=timeout_s)
            status = "ok" if code == 0 else "crash"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code, status = None, "timeout"
    result = parse_events(events_path)
    result.wall_s = round(time.monotonic() - t0, 2)
    if status != "ok":
        result.status = status
        if status == "crash" and not result.error:
            result.error = (sandbox / "pi_stderr.txt").read_text()[-2000:]
        if status == "timeout":
            result.error = f"timed out after {timeout_s}s"
    return result


def parse_events(path: Path) -> AgentResult:
    """Token usage, cost and errors from Pi's JSON event stream.

    Pi exits 0 even when the provider call fails, so a model error is detected
    from the final assistant message's stopReason, not the exit code.
    """
    r = AgentResult(status="ok")
    last_assistant = None
    if not path.exists():
        return AgentResult(status="crash", error="no pi output")
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "message_end":
            continue
        msg = e.get("message", {})
        if msg.get("role") != "assistant":
            continue
        last_assistant = msg
        r.n_turns += 1
        u = msg.get("usage") or {}
        r.input_tokens += int(u.get("input", 0))
        r.output_tokens += int(u.get("output", 0))
        r.cache_read_tokens += int(u.get("cacheRead", 0))
        r.cost_usd += float((u.get("cost") or {}).get("total", 0.0))
    if last_assistant is not None:
        r.final_text = " ".join(c.get("text", "") for c in last_assistant.get("content", []) if c.get("type") == "text").strip()[:2000]
        if last_assistant.get("stopReason") == "error":
            r.status = "agent_error"
            r.error = str(last_assistant.get("errorMessage", ""))[:2000]
    r.cost_usd = round(r.cost_usd, 6)
    return r


def trajectory_metrics(log_path: Path) -> dict:
    """Behavioral features from the extension's tool log (RQ3/RQ4 inputs)."""
    events = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    calls = [e for e in events if e.get("type") == "tool_result"]
    edits = [e for e in calls if e["tool"] == "midi_edit"]
    first_edit = next((i for i, e in enumerate(calls) if e["tool"] == "midi_edit"), None)
    before_edit = calls[:first_edit] if first_edit is not None else calls
    edit_errors = [i for i, e in enumerate(calls) if e["tool"] == "midi_edit" and e.get("isError")]
    recovered = bool(edit_errors) and any(e["tool"] == "midi_edit" and not e.get("isError") for e in calls[edit_errors[0] + 1:])
    return {
        "n_tool_calls": len(calls),
        "n_read_calls": sum(e["tool"] == "midi_read" for e in calls),
        "n_analyze_calls": sum(e["tool"] == "midi_analyze" for e in calls),
        "n_edit_calls": len(edits),
        "verify_calls": sum(e["tool"] == "midi_verify" for e in calls),
        "edit_errors": len(edit_errors),
        "tool_errors": sum(bool(e.get("isError")) for e in calls),
        "recovered_after_error": recovered,
        "analyzed_before_first_edit": any(e["tool"] == "midi_analyze" for e in before_edit) if first_edit is not None else False,
        "read_before_first_edit": any(e["tool"] == "midi_read" for e in before_edit) if first_edit is not None else False,
        "hit_budget": any(e.get("type") == "budget_exhausted" for e in events),
    }
