"""Views must carry identical information: parse(render(score)) reproduces the score."""

import pytest

from midicore import io
from midicore.views import VIEWS, parse_bars, render, resolve_tracks, select

from .conftest import roundtrip


def _canon(score):
    """Everything task-relevant, rounded to tick resolution."""
    q = 1 / score.ppq

    def r(x):
        return round(round(x / q) * q, 4)

    return {
        "notes": sorted((n.track, n.pitch, r(n.start), r(n.dur), n.vel) for n in score.all_notes()),
        "tempos": [(r(t.beat), round(t.bpm, 2)) for t in score.tempos],
        "time_sigs": [(r(t.beat), t.num, t.den) for t in score.time_sigs],
        "key_sigs": [(r(k.beat), k.key) for k in score.key_sigs],
        "pickup": score.pickup,
        "ppq": score.ppq,
        "tracks": list(score.tracks),
        "channels": {t: score.channels.get(t) for t in score.tracks},
        "programs": dict(score.programs),
    }


def test_io_round_trip(tricky_score, tmp_path):
    assert _canon(roundtrip(tricky_score, tmp_path)) == _canon(tricky_score)


def test_fixture_files_round_trip(fixture_paths, tmp_path):
    assert fixture_paths, "fixtures missing; run tasks/fixtures/build_fixtures.py"
    for p in fixture_paths:
        s = io.load(p)
        assert _canon(roundtrip(s, tmp_path)) == _canon(s), p.name


@pytest.mark.parametrize("view", list(VIEWS))
def test_view_is_lossless(view, tricky_score, tmp_path):
    s = roundtrip(tricky_score, tmp_path)
    parsed = VIEWS[view].parse(render(s, view))
    assert _canon(parsed) == _canon(s)


@pytest.mark.parametrize("view", list(VIEWS))
def test_views_lossless_on_fixtures(view, fixture_paths):
    for p in fixture_paths:
        s = io.load(p)
        assert _canon(VIEWS[view].parse(render(s, view))) == _canon(s), p.name


def test_notes_view_format(tricky_score):
    text = render(tricky_score, "notes")
    assert "[Lead] bar 0 beat 1  G4  dur 1  vel 90" in text
    assert "[Lead] bar 1 beat 1  C5  dur 0.5  vel 80" in text
    assert "time_signature 3/4 at bar 4 beat 1" in text
    assert "tempo 120 at bar 4 beat 1" in text


def test_musical_view_marks_rests_overlaps_and_chords(tricky_score):
    text = render(tricky_score, "musical")
    assert "estimated key:" in text
    assert "r(0.25)" in text  # gap before E5
    assert "@" in text  # overlapping G5 starts inside E5
    assert "C4+E4+G4(2)" in text
    assert "D5(0.25,v64)" in text


def test_raw_view_has_ticks_not_bars(tricky_score):
    text = render(tricky_score, "raw")
    assert "bar " not in text
    assert "note_on  ch=0 n=67 v=90" in text
    assert "set_tempo bpm=120" in text


def test_select_by_bars_and_tracks(tricky_score):
    sub = select(tricky_score, bars=(1, 1), tracks=["Lead"])
    assert list(sub.tracks) == ["Lead"]
    assert [n.pitch for n in sub.tracks["Lead"]] == [72, 74, 76, 79]


def test_parse_bars_and_tracks(tricky_score):
    assert parse_bars("5-8") == (5, 8)
    assert parse_bars("3") == (3, 3)
    assert parse_bars(None) is None
    with pytest.raises(ValueError):
        parse_bars("8-5")
    assert resolve_tracks(tricky_score, ["lead", "Keys"]) == ["Lead", "Keys"]
    with pytest.raises(ValueError, match="tracks are"):
        resolve_tracks(tricky_score, ["Violin"])
