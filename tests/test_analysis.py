import pytest

from midicore import io
from midicore.analysis import Key, analyze, estimate_key, label_chord, roman, roman_base
from midicore.model import Note

from .conftest import FIXTURES


def _chord(*pitches, dur=1.0):
    return [Note(0.0, p, dur, 80, "x") for p in pitches]


@pytest.mark.parametrize(
    "pitches,name",
    [
        ((60, 64, 67), "C"),
        ((57, 60, 64), "Am"),
        ((64, 67, 72), "C/E"),
        ((55, 59, 62, 65), "G7"),
        ((59, 62, 65), "Bdim"),
        ((48, 60, 64), "C"),  # missing fifth is fine
        ((50, 53, 57, 60), "Dm7"),
    ],
)
def test_label_chord(pitches, name):
    assert label_chord(_chord(*pitches)).name == name


def test_passing_tone_does_not_make_a_seventh():
    notes = _chord(60, 64, 67) + [Note(0.5, 70, 0.1, 80, "x")]
    assert label_chord(notes, 0.0, 1.0).name == "C"


@pytest.mark.parametrize(
    "pitches,key,expected",
    [
        ((55, 59, 62, 65), "C major", "V7"),
        ((62, 65, 69), "C major", "ii"),
        ((59, 62, 65), "C major", "vii°"),
        ((65, 69, 72), "C major", "IV"),
        ((64, 67, 72), "C major", "I6"),
        ((55, 59, 62), "A minor", "VII"),
        ((52, 56, 59), "A minor", "V"),
        ((60, 63, 67), "Bb major", "ii"),
    ],
)
def test_roman(pitches, key, expected):
    assert roman(label_chord(_chord(*pitches)), Key.parse(key)) == expected


def test_roman_base_equivalence():
    assert roman_base("ii7") == roman_base("ii65") == roman_base("ii") == "ii"
    assert roman_base("V7") == "V"
    assert roman_base("I64") == "I"
    assert roman_base("viiø7") == "vii°"


@pytest.mark.parametrize("text,name", [("Bb major", "Bb major"), ("Gm", "G minor"), ("F#m", "F# minor"), ("c minor", "C minor"), ("Eb", "Eb major")])
def test_key_parse(text, name):
    assert Key.parse(text).name == name


def test_fixture_keys():
    expected = {"chorale_bwv347.mid": "A major", "chorale_bwv40_8.mid": "F minor", "leadsheet_c.mid": "C major"}
    for fname, key in expected.items():
        s = io.load(FIXTURES / fname)
        assert estimate_key(s.all_notes())[0].name == key, fname
        assert analyze(s)["key"]["name"] == key


def test_analyze_reports_grid_and_drums():
    a = analyze(io.load(FIXTURES / "groove_em.mid"))
    assert a["tracks"]["Drums"]["drums"] is True
    assert a["tracks"]["Drums"]["quantize_grid"] == "unquantized"
    assert a["tracks"]["Bass"]["quantize_grid"] == "0.5"
