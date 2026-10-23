import json

import pytest

from midicore import cli, io
from midicore.model import Note, Score, TimeSig
from midicore.ops import OpError, apply_ops, diff
from midicore.verify import run_check, run_checks


@pytest.fixture
def chorale() -> Score:
    """Two voices, 4 bars of 4/4, quarter notes."""
    s = Score(tracks={"S": [], "B": []}, channels={"S": 0, "B": 1})
    for bar in range(4):
        for beat in range(4):
            t = bar * 4 + beat
            s.tracks["S"].append(Note(float(t), 72 + (beat % 3), 1.0, 80, "S", 0))
            s.tracks["B"].append(Note(float(t), 48 + beat, 1.0, 80, "B", 1))
    return s


def test_transpose_scoped(chorale):
    out = apply_ops(chorale, [{"op": "transpose", "semitones": -2, "scope": {"tracks": ["S"], "bars": [2, 3]}}])
    assert run_check("transpose_exact", out, chorale, {"semitones": -2, "tracks": ["S"], "bars": [2, 3]}).passed
    assert run_check("preserve", out, chorale, {"tracks": ["S"], "bars": [[1, 1], [4, None]]}).passed
    assert run_check("preserve", out, chorale, {"tracks": ["B"]}).passed
    assert not run_check("preserve", out, chorale, {}).passed


def test_ops_are_atomic(chorale):
    ops = [
        {"op": "transpose", "semitones": 2},
        {"op": "delete_notes", "selector": {"track": "Alto"}},  # unknown track
    ]
    with pytest.raises(OpError) as e:
        apply_ops(chorale, ops)
    assert e.value.op_index == 1
    assert "tracks are: S, B" in e.value.message
    assert chorale.tracks["S"][0].pitch == 72  # input untouched


def test_invalid_op_reports_field(chorale):
    with pytest.raises(OpError) as e:
        apply_ops(chorale, [{"op": "insert_note", "track": "S", "bar": 1, "pitch": "C5"}])
    assert "dur" in e.value.message
    with pytest.raises(OpError):
        apply_ops(chorale, [{"op": "explode"}])
    with pytest.raises(OpError, match="past the end of bar"):
        apply_ops(chorale, [{"op": "insert_note", "track": "S", "bar": 1, "beat": 5, "pitch": "C5", "dur": 1}])


def test_out_of_range_transpose_fails(chorale):
    with pytest.raises(OpError, match="outside MIDI range"):
        apply_ops(chorale, [{"op": "transpose", "semitones": 60}])


def test_insert_delete_velocity_tempo(chorale):
    out = apply_ops(chorale, [
        {"op": "insert_note", "track": "B", "bar": 5, "beat": 1, "pitch": "C2", "dur": 4},
        {"op": "delete_notes", "selector": {"track": "S", "bars": [3, 4]}},
        {"op": "set_velocity", "selector": {"track": "B"}, "value": 64},
        {"op": "set_tempo", "bpm": 90},
        {"op": "set_tempo", "bpm": 120, "at_bar": 3},
    ])
    assert run_check("contains", out, chorale, {"track": "B", "bar": 5, "pitch": "C2", "dur": 4}).passed
    assert run_check("empty", out, chorale, {"tracks": ["S"], "bars": [3, 4]}).passed
    assert run_check("velocity", out, chorale, {"tracks": ["B"], "value": 64}).passed
    assert run_check("tempo", out, chorale, {"bpm": 90, "bars": [1, 2]}).passed
    assert run_check("tempo", out, chorale, {"bpm": 120, "bars": [3, None]}).passed
    assert not run_check("tempo", out, chorale, {"bpm": 90}).passed


def test_quantize_and_check():
    s = Score(tracks={"D": [Note(0.03, 36, 0.25, 90, "D", 9), Note(0.52, 42, 0.25, 90, "D", 9), Note(0.98, 38, 0.25, 90, "D", 9)]})
    assert not run_check("quantize", s, None, {"grid": 0.25}).passed
    out = apply_ops(s, [{"op": "quantize", "grid": 0.25}])
    assert [n.start for n in out.tracks["D"]] == [0.0, 0.5, 1.0]
    assert run_check("quantize", out, s, {"grid": 0.25}).passed
    assert run_check("preserve_pitches", out, s, {}).passed
    half = apply_ops(s, [{"op": "quantize", "grid": 0.25, "strength": 0.5}])
    assert half.tracks["D"][0].start == pytest.approx(0.015)


def test_replace_bars_and_add_track(chorale):
    out = apply_ops(chorale, [
        {"op": "add_track", "name": "Alto", "channel": 3},
        {"op": "replace_bars", "track": "Alto", "bars": [1, 1], "notes": [{"bar": 1, "beat": 1, "pitch": "G4", "dur": 4}]},
        {"op": "replace_bars", "track": "B", "bars": [2, 2], "notes": [{"bar": 2, "pitch": "F2", "dur": 4}]},
    ])
    assert [n.pitch for n in out.tracks["Alto"]] == [67]
    assert out.channels["Alto"] == 2
    assert [n.pitch for n in out.tracks["B"] if 4 <= n.start < 8] == [41]
    with pytest.raises(OpError, match="outside bars"):
        apply_ops(chorale, [{"op": "replace_bars", "track": "B", "bars": [2, 2], "notes": [{"bar": 3, "pitch": "C3", "dur": 1}]}])


def test_move_note_selector_by_position(chorale):
    out = apply_ops(chorale, [{"op": "move_note", "selector": {"track": "S", "bar": 2, "beat": 3}, "dpitch": 1}])
    changed = [n for n in out.tracks["S"] if n.start == 6.0]
    assert changed[0].pitch == chorale.tracks["S"][6].pitch + 1


def test_diff_summary(chorale):
    out = apply_ops(chorale, [{"op": "transpose", "semitones": 1, "scope": {"tracks": ["S"], "bars": [2, 2]}}, {"op": "set_tempo", "bpm": 100}])
    d = diff(chorale, out)
    assert d["tracks"]["S"] == {"added": 4, "removed": 4, "bars_touched": "2"}
    assert "B" not in d["tracks"]
    assert d["tempo"] == ["100 bpm at bar 1 beat 1"]
    assert len(d["listed"]) == 8
    assert diff(chorale, chorale)["summary"] == "no changes"


def test_transpose_exact_allow_octave(chorale):
    down = apply_ops(chorale, [{"op": "transpose", "semitones": -10}])
    assert not run_check("transpose_exact", down, chorale, {"semitones": 2}).passed
    r = run_check("transpose_exact", down, chorale, {"semitones": 2, "allow_octave": True})
    assert r.passed and r.data["semitones"] == -10


def test_key_and_range_checks():
    c_major = Score(tracks={"M": [Note(float(i), p, 1.0, 80, "M") for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60])]})
    assert run_check("key", c_major, None, {"key": "C major", "diatonic": True}).passed
    assert not run_check("key", c_major, None, {"key": "D major"}).passed
    assert run_check("range", c_major, None, {"track": "M", "low": "C4", "high": "C5"}).passed
    r = run_check("range", c_major, None, {"track": "M", "low": "D4", "high": "C5"})
    assert not r.passed and r.score == pytest.approx(1 - 2 / 11)


def test_run_checks_reports_errors_per_constraint(chorale):
    res = run_checks([{"name": "nope"}, {"name": "preserve", "args": {}}, {"name": "range", "args": {"low": "C2"}}], chorale, None)
    assert res[0]["error"] and "unknown check" in res[0]["detail"]
    assert res[1]["error"] and "original file" in res[1]["detail"]
    assert res[2]["passed"]


def test_cli_edit_is_atomic_and_protects_input(chorale, tmp_path, capsys):
    f = tmp_path / "work.mid"
    io.save(chorale, f)
    before = f.read_bytes()
    code = cli.main(["edit", "--file", str(f), "--ops", json.dumps([{"op": "transpose", "semitones": 1}, {"op": "set_velocity", "selector": {"track": "X"}, "value": 1}])])
    out = json.loads(capsys.readouterr().out)
    assert code == 1 and not out["ok"] and out["error"]["op_index"] == 1
    assert f.read_bytes() == before

    code = cli.main(["edit", "--file", str(f), "--ops", json.dumps([{"op": "transpose", "semitones": 1}])])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["result"]["diff"]["tracks"]["S"]["added"] == 16

    inp = tmp_path / "input.mid"
    io.save(chorale, inp)
    cli.main(["edit", "--file", str(inp), "--protect", "input.mid", "--ops", "[]"])
    assert "read-only" in json.loads(capsys.readouterr().out)["error"]["message"]

    cli.main(["verify", "--file", str(f), "--reference", str(inp), "--constraints", json.dumps([{"name": "transpose_exact", "args": {"semitones": 1}}])])
    assert json.loads(capsys.readouterr().out)["result"]["all_passed"]


def test_cli_read_errors_are_json(tmp_path, capsys):
    assert cli.main(["read", "--file", str(tmp_path / "missing.mid")]) == 1
    assert "file not found" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_bar_scoped_check_with_meter_change():
    s = Score(tracks={"A": [Note(0.0, 60, 1, 80, "A"), Note(4.0, 62, 1, 80, "A"), Note(7.0, 64, 1, 80, "A")]}, time_sigs=[TimeSig(0, 4, 4), TimeSig(4, 3, 4)])
    out = apply_ops(s, [{"op": "transpose", "semitones": 12, "scope": {"bars": [3, 3]}}])
    assert [n.pitch for n in out.tracks["A"]] == [60, 62, 76]  # beat 7 is bar 3 because bar 2 is 3/4


@pytest.mark.parametrize("bars", [[2, 3], "2-3", "[2, 3]", "2 - 3"])
def test_bar_range_formats_are_equivalent(chorale, bars):
    out = apply_ops(chorale, [{"op": "delete_notes", "selector": {"track": "S", "bars": bars}}])
    assert run_check("empty", out, chorale, {"tracks": ["S"], "bars": bars}).passed
    assert len(out.tracks["S"]) == 8


def test_single_bar_string(chorale):
    out = apply_ops(chorale, [{"op": "delete_notes", "selector": {"track": "S", "bars": "2"}}])
    assert len(out.tracks["S"]) == 12
