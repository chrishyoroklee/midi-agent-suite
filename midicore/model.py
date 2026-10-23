"""Core data model. Everything above io.py works in beats (quarter notes) as floats.

Bars are 1-based. When a score has a pickup (anacrusis), the pickup is bar 0 and
bar 1 is the first full bar. Beat-in-bar positions are 1-based and measured in
quarter notes, so in 4/4 the beats of a bar are 1.0, 2.0, 3.0, 4.0 and in 6/8 a
bar spans beats 1.0 up to 4.0 (exclusive).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace

EPS = 1e-6

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]*)(-?\d+)$")


def pitch_name(pitch: int, names: list[str] | None = None) -> str:
    """MIDI number to scientific pitch name. Middle C (60) is C4.

    `names` picks the spelling (see analysis.names_for_key); MIDI itself has none.
    """
    return f"{(names or NOTE_NAMES)[pitch % 12]}{pitch // 12 - 1}"


def parse_pitch(value: int | str) -> int:
    """Accept a MIDI number or a name like C4, Bb3, F#5, Cb4."""
    if isinstance(value, int):
        if not 0 <= value <= 127:
            raise ValueError(f"pitch {value} out of MIDI range 0-127")
        return value
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return parse_pitch(int(text))
    m = _PITCH_RE.match(text)
    if not m:
        raise ValueError(f"cannot parse pitch {value!r}; use a name like C4, Bb3, F#5 or a MIDI number")
    letter, accidentals, octave = m.groups()
    pc = _LETTER_PC[letter.upper()] + accidentals.count("#") - accidentals.count("b")
    pitch = (int(octave) + 1) * 12 + pc
    return parse_pitch(pitch)


def coerce_bar_range(value):
    """Accept a bar range as [5, 8], "5-8", "[5, 8]" or "5"; other values pass through.

    midi_read takes bars as a "5-8" string, so agents reasonably send the same
    form to ops and checks; rejecting it would grade API trivia, not music.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("["):
        import json

        try:
            return json.loads(text)
        except ValueError:
            return value
    m = re.fullmatch(r"(-?\d+)\s*(?:-|to|\.\.)\s*(-?\d+)", text)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    if text.lstrip("-").isdigit():
        return [int(text), int(text)]
    return value


def fmt_num(x: float) -> str:
    """Compact, lossless-enough number formatting for views (6 decimals, trimmed)."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


@dataclass(frozen=True, order=True)
class Note:
    start: float
    pitch: int
    dur: float
    vel: int = 80
    track: str = ""
    channel: int = 0

    @property
    def end(self) -> float:
        return self.start + self.dur

    def key(self, with_vel: bool = False) -> tuple:
        """Identity used for set comparisons; rounded so float noise never matters."""
        k = (self.track, self.pitch, round(self.start, 4), round(self.dur, 4))
        return k + (self.vel,) if with_vel else k


@dataclass(frozen=True)
class TimeSig:
    beat: float
    num: int
    den: int

    @property
    def bar_len(self) -> float:
        return self.num * 4.0 / self.den


@dataclass(frozen=True)
class Tempo:
    beat: float
    bpm: float


@dataclass(frozen=True)
class KeySig:
    beat: float
    key: str  # mido notation: "Bb", "F#m", ...


@dataclass
class Score:
    tracks: dict[str, list[Note]] = field(default_factory=dict)
    tempos: list[Tempo] = field(default_factory=lambda: [Tempo(0.0, 120.0)])
    time_sigs: list[TimeSig] = field(default_factory=lambda: [TimeSig(0.0, 4, 4)])
    key_sigs: list[KeySig] = field(default_factory=list)
    ppq: int = 480
    pickup: float = 0.0  # length of the anacrusis in beats; 0 means none
    channels: dict[str, int] = field(default_factory=dict)
    programs: dict[str, int] = field(default_factory=dict)

    def copy(self) -> Score:
        return copy.deepcopy(self)

    def all_notes(self) -> list[Note]:
        return sorted(n for notes in self.tracks.values() for n in notes)

    @property
    def end(self) -> float:
        return max((n.end for n in self.all_notes()), default=0.0)

    @property
    def bars(self) -> BarMap:
        return BarMap(self.time_sigs, self.pickup)

    def n_bars(self) -> int:
        """Index of the last bar in which any note is sounding."""
        if self.end <= EPS:
            return 0
        # Step back less than one tick so a note ending exactly on a bar line stays in its bar.
        return self.bars.bar_of(self.end - 1e-3)

    def first_bar(self) -> int:
        return 0 if self.pickup > EPS else 1

    def sort(self) -> None:
        for name in self.tracks:
            self.tracks[name] = sorted(self.tracks[name])

    def with_notes(self, track: str, notes: list[Note]) -> Score:
        s = self.copy()
        s.tracks[track] = sorted(replace(n, track=track) for n in notes)
        return s


class BarMap:
    """Maps beats to (bar, beat-in-bar) and back using the time-signature map.

    Time-signature changes take effect at the first bar line at or after the
    change. A pickup of length p makes bar 0 span [0, p) and bar 1 start at p.
    """

    def __init__(self, time_sigs: list[TimeSig], pickup: float = 0.0):
        self.time_sigs = sorted(time_sigs, key=lambda t: t.beat) or [TimeSig(0.0, 4, 4)]
        self.pickup = pickup
        self._starts: list[float] = [pickup]  # _starts[i] is the start of bar i+1

    def _ts_at(self, beat: float) -> TimeSig:
        current = self.time_sigs[0]
        for ts in self.time_sigs:
            if ts.beat <= beat + EPS:
                current = ts
            else:
                break
        return current

    def _extend_to_bar(self, bar: int) -> None:
        while len(self._starts) < bar:
            last = self._starts[-1]
            self._starts.append(last + self._ts_at(last).bar_len)

    def _extend_to_beat(self, beat: float) -> None:
        while self._starts[-1] <= beat + EPS:
            last = self._starts[-1]
            self._starts.append(last + self._ts_at(last).bar_len)

    def bar_start(self, bar: int) -> float:
        if bar == 0:
            if self.pickup <= EPS:
                raise ValueError("bar 0 only exists when the score has a pickup")
            return 0.0
        if bar < 0:
            raise ValueError(f"bar {bar} is invalid; bars start at 1 (or 0 for a pickup)")
        self._extend_to_bar(bar)
        return self._starts[bar - 1]

    def bar_end(self, bar: int) -> float:
        if bar == 0:
            return self.pickup
        return self.bar_start(bar + 1)

    def bar_len(self, bar: int) -> float:
        return self.bar_end(bar) - self.bar_start(bar)

    def time_sig_of_bar(self, bar: int) -> TimeSig:
        return self._ts_at(self.bar_start(bar) if bar > 0 else 0.0)

    def bar_of(self, beat: float) -> int:
        if beat < self.pickup - EPS:
            return 0
        self._extend_to_beat(beat)
        for i in range(len(self._starts) - 1, -1, -1):
            if self._starts[i] <= beat + EPS:
                return i + 1
        return 1

    def position(self, beat: float) -> tuple[int, float]:
        """beat -> (bar, beat_in_bar), beat_in_bar 1-based in quarter notes."""
        bar = self.bar_of(beat)
        return bar, beat - self.bar_start(bar) + 1.0

    def beat_of(self, bar: int, beat_in_bar: float = 1.0) -> float:
        if beat_in_bar < 1.0 - EPS:
            raise ValueError(f"beat {beat_in_bar} is invalid; beats in a bar start at 1.0")
        if beat_in_bar - 1.0 >= self.bar_len(bar) - EPS:
            raise ValueError(
                f"beat {beat_in_bar} is past the end of bar {bar} "
                f"(bar {bar} has {fmt_num(self.bar_len(bar))} beats)"
            )
        return self.bar_start(bar) + beat_in_bar - 1.0
