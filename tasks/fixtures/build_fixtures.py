"""Regenerate the input fixtures. Deterministic: rerunning produces identical files.

    uv run python tasks/fixtures/build_fixtures.py

Chorales come from the music21 corpus (public domain). Lead sheet and groove are
written here so their ground truth is known exactly.
"""

from __future__ import annotations

import random
from pathlib import Path

from midicore import io
from midicore.model import KeySig, Note, Score, Tempo, TimeSig, parse_pitch

OUT = Path(__file__).parent


def chorale(bwv: str, bpm: float) -> Score:
    from music21 import corpus, key as m21key

    s = corpus.parse(f"bach/{bwv}").stripTies()
    first = s.parts[0].getElementsByClass("Measure")[0]
    pickup = float(first.duration.quarterLength) if first.number == 0 else 0.0
    ts = s.recurse().getElementsByClass("TimeSignature").first()
    ks = s.recurse().getElementsByClass("KeySignature").first()
    score = Score(tracks={}, tempos=[Tempo(0.0, bpm)], time_sigs=[TimeSig(0.0, ts.numerator, ts.denominator)], pickup=pickup)
    if ks is not None:
        k = ks.asKey("major") if not isinstance(ks, m21key.Key) else ks
        score.key_sigs = [KeySig(0.0, k.tonic.name.replace("-", "b") + ("m" if k.mode == "minor" else ""))]
    for ch, part in enumerate(s.parts):
        name = part.partName
        notes = []
        for n in part.flatten().notes:
            for p in n.pitches:
                notes.append(Note(round(float(n.offset), 6), p.midi, round(float(n.quarterLength), 6), 80, name, ch))
        score.tracks[name] = sorted(notes)
        score.channels[name] = ch
    return score


def _seq(track: str, ch: int, start_bar: int, bars: list[str], vel: int = 80) -> list[Note]:
    """bars: one string per 4/4 bar, tokens PITCH:DUR, 'r:DUR' for rests."""
    out, t = [], (start_bar - 1) * 4.0
    for bar in bars:
        b0 = t
        for tok in bar.split():
            p, d = tok.split(":")
            if p != "r":
                out.append(Note(t, parse_pitch(p), float(d), vel, track, ch))
            t += float(d)
        assert abs(t - b0 - 4.0) < 1e-9, f"bar does not add up to 4 beats: {bar}"
    return out


def leadsheet() -> Score:
    melody = [
        "E5:1 D5:1 C5:1 D5:1", "E5:1 E5:1 E5:2", "D5:1 D5:1 D5:2", "E5:1 G5:1 G5:2",
        "E5:1 D5:1 C5:1 D5:1", "E5:1 E5:1 E5:1 E5:1", "D5:1 D5:1 E5:1 D5:1", "C5:4",
        "G5:1.5 F5:0.5 E5:1 C5:1", "A5:2 G5:2", "F5:1 A5:1 C6:1 A5:1", "G5:3 r:1",
        "E5:1 G5:1 C6:1.5 B5:0.5", "A5:1 G5:1 E5:2", "F5:1 D5:1 G5:1 B4:1", "C5:4",
    ]
    roots = ["C3", "A2", "F2", "G2", "C3", "E2", "F2", "C3", "C3", "F2", "F2", "G2", "C3", "A2", "G2", "C3"]
    bass = [f"{r}:2 {r}:2" for r in roots[:-1]] + ["r:4"]  # bass drops out in the last bar
    triads = {"C": "C4 E4 G4", "A": "A3 C4 E4", "F": "F3 A3 C4", "G": "G3 B3 D4", "E": "E3 G3 B3"}
    piano = []
    for i, r in enumerate(roots):
        for p in triads[r[0]].split():
            piano.append(Note(i * 4.0, parse_pitch(p), 4.0, 64, "Piano", 2))
    return Score(
        tracks={"Melody": _seq("Melody", 0, 1, melody, 90), "Bass": _seq("Bass", 1, 1, bass, 85), "Piano": sorted(piano)},
        tempos=[Tempo(0.0, 100.0)], time_sigs=[TimeSig(0.0, 4, 4)], key_sigs=[KeySig(0.0, "C")],
        channels={"Melody": 0, "Bass": 1, "Piano": 2}, programs={"Melody": 73, "Bass": 33, "Piano": 0},
    )


def groove() -> Score:
    rng = random.Random(7)
    drums = []
    for bar in range(8):
        b0 = bar * 4.0
        for i in range(8):
            t = b0 + i * 0.5
            hits = [42]
            if i in (0, 3, 4):
                hits.append(36)
            if i in (2, 6):
                hits.append(38)
            for p in hits:
                jitter = round(rng.uniform(-0.06, 0.06), 3)
                vel = rng.randint(70, 110) if p != 42 else rng.randint(50, 80)
                drums.append(Note(round(max(0.0, t + jitter), 6), p, 0.25, vel, "Drums", 9))
    roots = ["E2", "E2", "C2", "C2", "G2", "G2", "D2", "D2"]
    bass = []
    for bar, r in enumerate(roots):
        for i in range(8):
            p = parse_pitch(r) + (12 if i in (3, 7) else 0)
            bass.append(Note(bar * 4.0 + i * 0.5, p, 0.5, 96 if i % 2 == 0 else 72, "Bass", 1))
    return Score(
        tracks={"Drums": sorted(drums), "Bass": bass},
        tempos=[Tempo(0.0, 96.0)], time_sigs=[TimeSig(0.0, 4, 4)], key_sigs=[KeySig(0.0, "Em")],
        channels={"Drums": 9, "Bass": 1}, programs={"Bass": 33},
    )


def main() -> None:
    fixtures = {
        "chorale_bwv347.mid": chorale("bwv347", 80.0),
        "chorale_bwv40_8.mid": chorale("bwv40.8", 80.0),
        "leadsheet_c.mid": leadsheet(),
        "groove_em.mid": groove(),
    }
    for name, score in fixtures.items():
        io.save(score, OUT / name)
        print(f"wrote {name}: {', '.join(f'{t} ({len(n)})' for t, n in score.tracks.items())}")


if __name__ == "__main__":
    main()
