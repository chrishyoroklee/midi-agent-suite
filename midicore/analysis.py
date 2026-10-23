"""Musical analysis shared by the agent's tools and the graders.

Deliberately dependency-free: key finding is Krumhansl-Schmuckler correlation
with the Aarden-Essen profiles (the music21 default; on 20 Bach chorales it
agreed with music21 20/20, Krumhansl-Kessler only 13/20), chords are labelled by template matching over
duration-weighted pitch classes. Both are approximate, and the musical view
exposes their output to the agent as-is (over-trusting it is a failure mode).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import EPS, FLAT_NAMES, NOTE_NAMES, SHARP_NAMES, Note, Score, fmt_num, pitch_name

AARDEN_MAJOR = [17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122]
AARDEN_MINOR = [18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623]

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # natural minor; raised 6/7 handled as chromatic-but-expected

# Preferred spelling of tonic names per key (pc -> name) to avoid "A# major".
MAJOR_TONIC = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_TONIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class
    mode: str  # "major" | "minor"

    @property
    def name(self) -> str:
        names = MAJOR_TONIC if self.mode == "major" else MINOR_TONIC
        return f"{names[self.tonic]} {self.mode}"

    @property
    def scale(self) -> list[int]:
        steps = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE
        return [(self.tonic + s) % 12 for s in steps]

    def scale_pcs(self, harmonic: bool = True) -> set[int]:
        pcs = set(self.scale)
        if self.mode == "minor" and harmonic:
            pcs |= {(self.tonic + 9) % 12, (self.tonic + 11) % 12}  # melodic/harmonic 6 and 7
        return pcs

    @staticmethod
    def parse(text: str) -> Key:
        """Accept 'Bb major', 'G minor', 'Bb', 'Gm', 'F#m', 'c minor'."""
        t = text.strip().replace("-", " ")
        mode = "major"
        low = t.lower()
        if low.endswith(" minor") or low.endswith(" min"):
            mode, t = "minor", t.rsplit(" ", 1)[0]
        elif low.endswith(" major") or low.endswith(" maj"):
            t = t.rsplit(" ", 1)[0]
        elif t.endswith("m") and len(t) >= 2:
            mode, t = "minor", t[:-1]
        t = t.strip()
        if not t or t[0].upper() not in "ABCDEFG":
            raise ValueError(f"cannot parse key {text!r}")
        pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[t[0].upper()]
        pc += t[1:].count("#") - t[1:].count("b")
        return Key(pc % 12, mode)


_SHARP_MAJOR_TONICS = {7, 2, 9, 4, 11, 6}  # G D A E B F#
_SHARP_MINOR_TONICS = {4, 11, 6, 1, 8, 3}  # E B F# C# G# D#
_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1}  # F Bb Eb Ab Db
_FLAT_MINOR_TONICS = {2, 7, 0, 5, 10}  # D G C F Bb


def names_for_key(key: Key | None) -> list[str]:
    """Pitch-class spelling that follows the key signature (sharps or flats)."""
    if key is None:
        return NOTE_NAMES
    sharps = _SHARP_MAJOR_TONICS if key.mode == "major" else _SHARP_MINOR_TONICS
    flats = _FLAT_MAJOR_TONICS if key.mode == "major" else _FLAT_MINOR_TONICS
    if key.tonic in sharps:
        return SHARP_NAMES
    if key.tonic in flats:
        return FLAT_NAMES
    return NOTE_NAMES


def declared_key(score: Score) -> Key | None:
    """The key signature at the start of the file, if the file declares one."""
    if not score.key_sigs:
        return None
    try:
        return Key.parse(score.key_sigs[0].key)
    except ValueError:
        return None


def _pc_weights(notes: list[Note]) -> list[float]:
    w = [0.0] * 12
    for n in notes:
        w[n.pitch % 12] += n.dur
    return w


def _corr(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / 12, sum(y) / 12
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den > 0 else 0.0


def key_candidates(notes: list[Note]) -> list[tuple[Key, float]]:
    """All 24 keys ranked by profile correlation."""
    w = _pc_weights(notes)
    out = []
    for tonic in range(12):
        for mode, profile in (("major", AARDEN_MAJOR), ("minor", AARDEN_MINOR)):
            rotated = [profile[(pc - tonic) % 12] for pc in range(12)]
            out.append((Key(tonic, mode), _corr(w, rotated)))
    return sorted(out, key=lambda kc: -kc[1])


def estimate_key(notes: list[Note]) -> tuple[Key, float]:
    return key_candidates(notes)[0]


# ---------------------------------------------------------------- chords

# name -> (intervals above root, display suffix)
CHORD_TEMPLATES: dict[str, tuple[tuple[int, ...], str]] = {
    "maj": ((0, 4, 7), ""),
    "min": ((0, 3, 7), "m"),
    "dim": ((0, 3, 6), "dim"),
    "aug": ((0, 4, 8), "aug"),
    "dom7": ((0, 4, 7, 10), "7"),
    "maj7": ((0, 4, 7, 11), "maj7"),
    "min7": ((0, 3, 7, 10), "m7"),
    "hdim7": ((0, 3, 6, 10), "m7b5"),
    "dim7": ((0, 3, 6, 9), "dim7"),
    "sus4": ((0, 5, 7), "sus4"),
}
_TEMPLATE_PRIORITY = {name: i for i, name in enumerate(CHORD_TEMPLATES)}


@dataclass(frozen=True)
class Chord:
    root: int
    quality: str
    bass: int  # pitch class of the lowest note struck at the start of the window
    score: float

    def label(self, names: list[str] | None = None) -> str:
        names = names or NOTE_NAMES
        base = f"{names[self.root]}{CHORD_TEMPLATES[self.quality][1]}"
        if self.bass != self.root:
            base += f"/{names[self.bass]}"
        return base

    @property
    def name(self) -> str:
        return self.label()

    @property
    def pcs(self) -> set[int]:
        return {(self.root + i) % 12 for i in CHORD_TEMPLATES[self.quality][0]}


def label_chord(notes: list[Note], start: float | None = None, end: float | None = None) -> Chord | None:
    """Label the sonority sounding in [start, end) (or of all given notes).

    Notes already sounding when the window opens count 1.5x, so passing tones
    in the middle of a beat do not outweigh the chord they decorate. The bass
    is the lowest note sounding at the start of the window.
    """
    sounding: list[tuple[int, float, bool]] = []
    for n in notes:
        if start is not None and end is not None:
            overlap = min(n.end, end) - max(n.start, start)
            if overlap <= EPS:
                continue
            at_start = n.start <= start + EPS
        else:
            overlap, at_start = n.dur, True
        sounding.append((n.pitch, overlap * (1.5 if at_start else 1.0), at_start))
    if not sounding:
        return None
    w = [0.0] * 12
    for p, weight, _ in sounding:
        w[p % 12] += weight
    total = sum(w)
    w = [x / total for x in w]
    struck = [p for p, _, at_start in sounding if at_start]
    bass_pc = min(struck or [p for p, _, _ in sounding]) % 12
    present = {pc for pc in range(12) if w[pc] > 0.02}

    best: tuple[float, int, int, str] | None = None
    for root in range(12):
        if root not in present:
            continue
        for quality, (ivs, _) in CHORD_TEMPLATES.items():
            # A seventh chord needs a seventh that is more than a passing tone.
            if len(ivs) == 4 and w[(root + ivs[3]) % 12] < 0.12:
                continue
            tpl = {(root + i) % 12 for i in ivs}
            covered = sum(w[pc] for pc in tpl)
            # A missing fifth is common and cheap; a missing third or seventh is not.
            penalty = sum(0.08 if i == 7 else 0.25 for i in ivs if (root + i) % 12 not in present)
            s = covered - (1 - covered) - penalty + (0.06 if root == bass_pc else 0.0)
            cand = (s, -_TEMPLATE_PRIORITY[quality], root, quality)
            if best is None or cand > best:
                best = cand
    if best is None:
        return None
    return Chord(best[2], best[3], bass_pc, round(best[0], 3))


# ---------------------------------------------------------------- roman numerals

_MAJOR_DEGREES = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
_MINOR_DEGREES = {0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV", 6: "#IV", 7: "V", 8: "VI", 9: "#VI", 10: "VII", 11: "#VII"}
_QUALITY_SUFFIX = {"maj": "", "min": "", "dim": "°", "aug": "+", "dom7": "7", "maj7": "maj7", "min7": "7", "hdim7": "ø7", "dim7": "°7", "sus4": "sus4"}
_LOWER = {"min", "dim", "min7", "hdim7", "dim7"}
_INVERSION_FIGURES_TRIAD = {1: "6", 2: "64"}
_INVERSION_FIGURES_7TH = {1: "65", 2: "43", 3: "42"}


def roman(chord: Chord, key: Key) -> str:
    degrees = _MAJOR_DEGREES if key.mode == "major" else _MINOR_DEGREES
    numeral = degrees[(chord.root - key.tonic) % 12]
    acc = numeral[0] if numeral[0] in "b#" else ""
    body = numeral[len(acc):]
    if chord.quality in _LOWER:
        body = body.lower()
    suffix = _QUALITY_SUFFIX[chord.quality]
    ivs = CHORD_TEMPLATES[chord.quality][0]
    if chord.bass != chord.root:
        inv = next((i for i, iv in enumerate(ivs) if (chord.root + iv) % 12 == chord.bass), None)
        if inv:
            if len(ivs) == 4:
                suffix = suffix.replace("7", "") + _INVERSION_FIGURES_7TH.get(inv, "")
            else:
                suffix += _INVERSION_FIGURES_TRIAD.get(inv, "")
    return f"{acc}{body}{suffix}"


def roman_base(numeral: str) -> str:
    """Strip sevenths and inversion figures so ii7 == ii65 == ii, V7 == V, viiø7 == vii°."""
    out = numeral.replace("ø7", "°").replace("°7", "°").replace("ø", "°").replace("maj7", "").replace("sus4", "")
    for fig in ("65", "43", "42", "64", "7", "6"):
        out = out.replace(fig, "")
    return out


# ---------------------------------------------------------------- whole-score summary

QUANT_GRIDS = [1.0, 0.5, 1 / 3, 0.25, 1 / 6, 0.125, 1 / 12, 0.0625]


def onset_deviation(beat: float, grid: float) -> float:
    return abs(beat - round(beat / grid) * grid)


def estimate_grid(notes: list[Note], tol: float = 0.02, coverage: float = 0.95) -> float | None:
    if not notes:
        return None
    for g in QUANT_GRIDS:
        ok = sum(1 for n in notes if onset_deviation(n.start, g) <= tol)
        if ok / len(notes) >= coverage:
            return g
    return None


def chord_segments(score: Score, tracks: list[str] | None = None) -> list[tuple[int, float, float, Chord | None]]:
    """One chord per beat-sized window: (bar, beat_in_bar, start, chord).

    The window is one quarter note for x/4 and x/2 meters and a dotted quarter
    for compound x/8 meters (6/8, 9/8, 12/8).
    """
    notes = [n for name, ns in score.tracks.items() if tracks is None or name in tracks for n in ns if not is_drum(score, name)]
    bars = score.bars
    out = []
    for bar in range(score.first_bar(), score.n_bars() + 1):
        ts = bars.time_sig_of_bar(bar)
        if ts.den == 8 and ts.num % 3 == 0:
            step = 1.5
        elif ts.den >= 8:
            step = 4.0 / ts.den
        else:
            step = 1.0
        b0, b1 = bars.bar_start(bar), bars.bar_end(bar)
        t = b0
        while t < b1 - EPS:
            t1 = min(t + step, b1)
            window = [n for n in notes if n.start < t1 - EPS and n.end > t + EPS]
            out.append((bar, t - b0 + 1.0, t, label_chord(window, t, t1)))
            t = t1
    return out


def is_drum(score: Score, track: str) -> bool:
    return score.channels.get(track) == 9 or "drum" in track.lower()


def analyze(score: Score) -> dict:
    notes = [n for name, ns in score.tracks.items() if not is_drum(score, name) for n in ns]
    cands = key_candidates(notes) if notes else []
    key = cands[0][0] if cands else None
    bars = score.bars
    spelling = names_for_key(declared_key(score) or key)
    result: dict = {
        "key": {"name": key.name, "confidence": round(cands[0][1], 3)} if key else None,
        "key_alternatives": [{"name": k.name, "confidence": round(c, 3)} for k, c in cands[1:3]],
        "declared_key_signatures": [{"bar_beat": _bb(bars, k.beat), "key": k.key} for k in score.key_sigs],
        "meter": [{"bar": bars.bar_of(t.beat), "time_signature": f"{t.num}/{t.den}"} for t in score.time_sigs],
        "tempo": [{"bar_beat": _bb(bars, t.beat), "bpm": t.bpm} for t in score.tempos],
        "pickup_beats": score.pickup,
        "bars": {"first": score.first_bar(), "last": score.n_bars()},
        "ppq": score.ppq,
        "tracks": {},
    }
    for name, ns in score.tracks.items():
        if not ns:
            result["tracks"][name] = {"notes": 0}
            continue
        lo, hi = min(n.pitch for n in ns), max(n.pitch for n in ns)
        span_bars = max(1, len({bars.bar_of(n.start) for n in ns}))
        grid = estimate_grid(ns)
        result["tracks"][name] = {
            "notes": len(ns),
            "channel": score.channels.get(name, ns[0].channel) + 1,
            "range": [pitch_name(lo, spelling), pitch_name(hi, spelling)],
            "notes_per_bar": round(len(ns) / span_bars, 2),
            "velocity_range": [min(n.vel for n in ns), max(n.vel for n in ns)],
            "quantize_grid": fmt_num(grid) if grid else "unquantized",
            "polyphonic": any(a.end > b.start + EPS for a, b in zip(ns, ns[1:])),
            "drums": is_drum(score, name),
        }
    if key:
        by_bar: dict[int, list[str]] = {}
        for bar, _, _, chord in chord_segments(score):
            label = f"{roman(chord, key)} ({chord.label(names_for_key(key))})" if chord else "-"
            by_bar.setdefault(bar, []).append(label)
        result["chords_per_beat"] = {str(b): labels for b, labels in by_bar.items()}
    return result


def _bb(bars, beat: float) -> str:
    bar, bib = bars.position(beat)
    return f"{bar}:{fmt_num(bib)}"
