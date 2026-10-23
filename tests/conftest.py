from pathlib import Path

import pytest

from midicore import io
from midicore.model import KeySig, Note, Score, Tempo, TimeSig

FIXTURES = Path(__file__).resolve().parent.parent / "tasks" / "fixtures"


@pytest.fixture
def tricky_score() -> Score:
    """Pickup, meter change, tempo change, chords, overlaps, rests, varied velocities, drums."""
    s = Score(
        tracks={
            "Lead": [
                Note(0.0, 67, 1.0, 90, "Lead", 0),
                Note(1.0, 72, 0.5, 80, "Lead", 0),
                Note(1.5, 74, 0.25, 64, "Lead", 0),
                Note(2.5, 76, 2.0, 80, "Lead", 0),  # gap before, crosses a bar line
                Note(3.0, 79, 0.333333, 80, "Lead", 0),  # overlaps the previous note
                Note(19.0, 60, 1.0, 80, "Lead", 0),
            ],
            "Keys": [Note(1.0, p, 2.0, 70, "Keys", 1) for p in (60, 64, 67)] + [Note(5.0, p, 4.0, 70, "Keys", 1) for p in (57, 60, 64)],
            "Drums": [Note(1.0 + i * 0.5, 42, 0.25, 60 + i, "Drums", 9) for i in range(8)],
        },
        tempos=[Tempo(0.0, 96.0), Tempo(13.0, 120.0)],
        time_sigs=[TimeSig(0.0, 4, 4), TimeSig(13.0, 3, 4)],
        key_sigs=[KeySig(0.0, "G")],
        ppq=480,
        pickup=1.0,
        channels={"Lead": 0, "Keys": 1, "Drums": 9},
        programs={"Keys": 4},
    )
    s.sort()
    return s


@pytest.fixture
def fixture_paths() -> list[Path]:
    return sorted(FIXTURES.glob("*.mid"))


def roundtrip(score: Score, tmp_path) -> Score:
    p = tmp_path / "rt.mid"
    io.save(score, p)
    return io.load(p)
