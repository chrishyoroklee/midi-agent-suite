import json
import re
from pathlib import Path

import pytest

from harness.agent import parse_events, trajectory_metrics
from harness.metrics import pass_at_k, pass_hat_k, per_task, summarize
from harness.tasks import check_task, load_tasks
from midicore.ops import OP_NAMES, apply_ops
from midicore.verify import CHECKS

ROOT = Path(__file__).resolve().parent.parent
TASKS = {t.id: t for t in load_tasks()}


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_task_fails_on_input_and_passes_on_gold(task_id):
    assert check_task(TASKS[task_id]) == []


# Plausible wrong answers an agent might produce. Each must fail its task.
NEAR_MISSES = {
    "t1-transpose-001": [{"op": "transpose", "semitones": -1}],
    "t1-transpose-002": [{"op": "transpose", "semitones": 5, "scope": {"tracks": ["Melody"]}}],  # whole melody, not bars 5-8
    "t1-transpose-002#2": [{"op": "transpose", "semitones": 5, "scope": {"tracks": ["Melody"], "bars": [5, 9]}}],  # off by one bar
    "t1-transpose-003": [{"op": "transpose", "semitones": 2, "scope": {"tracks": ["Melody", "Piano"]}}],  # bass left behind
    "t1-octave-001": [{"op": "transpose", "semitones": 12}],  # drums moved too
    "t1-tempo-001": [{"op": "set_tempo", "bpm": 90, "at_bar": 2}],
    "t1-tempo-002": [{"op": "set_tempo", "bpm": 120, "at_bar": 1}],
    "t1-tempo-002#2": [{"op": "set_tempo", "bpm": 120, "at_bar": 8}],
    "t1-quantize-001": [{"op": "quantize", "grid": 0.25}],  # bass touched (it is already on the grid, so only velocity-safe)
    "t1-quantize-001#2": [{"op": "quantize", "grid": 0.5, "strength": 0.5, "scope": {"tracks": ["Drums"]}}],
    "t1-delete-001": [{"op": "delete_notes", "selector": {"track": "Alto", "bars": [3, 5]}}],
    "t1-delete-001#2": [{"op": "delete_notes", "selector": {"bars": [3, 4]}}],
    "t1-velocity-001": [{"op": "set_velocity", "selector": {}, "value": 64}],
    "t1-insert-001": [{"op": "insert_note", "track": "Bass", "bar": 16, "pitch": "C3", "dur": 4}],
    "t1-insert-001#2": [{"op": "insert_note", "track": "Bass", "bar": 15, "pitch": "C2", "dur": 4}],
}


@pytest.mark.parametrize("key", sorted(NEAR_MISSES))
def test_near_miss_fails(key):
    task = TASKS[key.split("#")[0]]
    out = apply_ops(task.load_input(), NEAR_MISSES[key])
    g = task.grade(out)
    if key == "t1-quantize-001":
        # The bass is already on the 16th grid, so quantizing everything is harmless: a valid answer.
        assert g.passed
        return
    assert not g.passed, g.results
    assert 0 <= g.partial < 1


def test_pass_metrics():
    assert pass_at_k(5, 0, 1) == 0
    assert pass_at_k(5, 5, 3) == 1
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(5, 1, 5) == 1.0
    assert pass_hat_k(5, 5, 5) == 1
    assert pass_hat_k(5, 4, 5) == 0
    assert pass_hat_k(5, 3, 3) == pytest.approx(1 / 10)
    assert pass_hat_k(5, 4, 3) == pytest.approx(4 / 10)


def test_summarize_averages_over_tasks():
    import pandas as pd

    rows = [{"task_id": "a", "tier": 1, "cfg": "x", "passed": p, "partial": float(p)} for p in (1, 1, 1)]
    rows += [{"task_id": "b", "tier": 1, "cfg": "x", "passed": p, "partial": float(p)} for p in (0, 0, 1)]
    df = pd.DataFrame(rows)
    s = summarize(df, ["cfg"], (1, 3)).iloc[0]
    assert s["pass@1"] == pytest.approx((1 + 1 / 3) / 2)
    assert s["pass^3"] == pytest.approx(0.5)
    assert len(per_task(df, ["cfg"], (1,))) == 2


def test_extension_documents_every_op_and_check():
    ts = (ROOT / "pi-ext" / "index.ts").read_text()
    for name in OP_NAMES:
        assert f'"op":"{name}"' in ts, f"op {name} missing from midi_edit description"
    for name in CHECKS:
        assert re.search(rf"^\s+{name}\s", ts, re.M), f"check {name} missing from midi_verify description"


def test_parse_events_detects_provider_error(tmp_path):
    p = tmp_path / "ev.jsonl"
    msgs = [
        {"type": "message_end", "message": {"role": "assistant", "content": [], "usage": {"input": 10, "output": 2, "cost": {"total": 0.01}}, "stopReason": "toolUse"}},
        {"type": "message_end", "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "401 bad key"}},
    ]
    p.write_text("\n".join(json.dumps(m) for m in msgs))
    r = parse_events(p)
    assert r.status == "agent_error" and "401" in r.error
    assert r.input_tokens == 10 and r.cost_usd == 0.01


def test_trajectory_metrics(tmp_path):
    p = tmp_path / "tools.jsonl"
    events = [
        {"type": "tool_result", "tool": "midi_read", "isError": False},
        {"type": "tool_result", "tool": "midi_analyze", "isError": False},
        {"type": "tool_result", "tool": "midi_edit", "isError": True},
        {"type": "tool_result", "tool": "midi_edit", "isError": False},
        {"type": "tool_result", "tool": "midi_verify", "isError": False},
    ]
    p.write_text("\n".join(json.dumps(e) for e in events))
    m = trajectory_metrics(p)
    assert m["n_tool_calls"] == 5 and m["n_edit_calls"] == 2 and m["edit_errors"] == 1
    assert m["recovered_after_error"] and m["analyzed_before_first_edit"] and m["verify_calls"] == 1
