"""Typed edit operations, applied atomically.

apply_ops() either returns a new Score with every op applied or raises OpError
naming the failing op; the input Score is never modified. Bars and beats in ops
are in the same coordinates the notes/musical views print (1-based, quarter notes).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from .model import EPS, Note, Score, Tempo, TimeSig, coerce_bar_range, fmt_num, parse_pitch, pitch_name

Pitch = Union[int, str]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Selector(_Strict):
    """Which notes an op touches. Every given field must match; omitted fields match all."""

    tracks: list[str] | None = None
    track: str | None = None
    bars: tuple[int, int] | None = Field(None, description="inclusive [first, last] bar of the note onset")
    bar: int | None = None
    beat: float | None = Field(None, description="beat-in-bar of the onset; requires bar")
    pitches: list[Pitch] | None = None
    pitch: Pitch | None = None

    @field_validator("bars", mode="before")
    @classmethod
    def _coerce(cls, v):
        return coerce_bar_range(v)

    @field_validator("bars")
    @classmethod
    def _ordered(cls, v):
        if v is not None and v[0] > v[1]:
            raise ValueError("bars must be [first, last] with first <= last")
        return v


class Transpose(_Strict):
    op: Literal["transpose"]
    semitones: int
    scope: Selector = Selector()


class Quantize(_Strict):
    op: Literal["quantize"]
    grid: float = Field(gt=0, description="grid in beats: 1=quarter, 0.5=eighth, 0.25=sixteenth, 1/3=eighth triplet")
    strength: float = Field(1.0, ge=0, le=1)
    ends: bool = Field(False, description="also snap note ends to the grid")
    scope: Selector = Selector()


class InsertNote(_Strict):
    op: Literal["insert_note"]
    track: str
    bar: int
    beat: float = 1.0
    pitch: Pitch
    dur: float = Field(gt=0)
    vel: int = Field(80, ge=1, le=127)


class DeleteNotes(_Strict):
    op: Literal["delete_notes"]
    selector: Selector


class MoveNote(_Strict):
    op: Literal["move_note"]
    selector: Selector
    dpitch: int = 0
    dbeat: float = 0.0


class SetVelocity(_Strict):
    op: Literal["set_velocity"]
    selector: Selector
    value: int = Field(ge=1, le=127)


class SetDuration(_Strict):
    op: Literal["set_duration"]
    selector: Selector
    dur: float = Field(gt=0)


class SetTempo(_Strict):
    op: Literal["set_tempo"]
    bpm: float = Field(gt=0, le=1000)
    at_bar: int = 1
    at_beat: float = 1.0


class SetTimeSignature(_Strict):
    op: Literal["set_time_signature"]
    num: int = Field(ge=1, le=32)
    den: Literal[1, 2, 4, 8, 16, 32]
    at_bar: int = 1


class BarNote(_Strict):
    bar: int
    beat: float = 1.0
    pitch: Pitch
    dur: float = Field(gt=0)
    vel: int = Field(80, ge=1, le=127)


class ReplaceBars(_Strict):
    op: Literal["replace_bars"]
    track: str
    bars: tuple[int, int]
    notes: list[BarNote]

    @field_validator("bars", mode="before")
    @classmethod
    def _coerce(cls, v):
        return coerce_bar_range(v)


class AddTrack(_Strict):
    op: Literal["add_track"]
    name: str = Field(min_length=1)
    channel: int = Field(1, ge=1, le=16, description="1-16; 10 is GM drums")
    program: int | None = Field(None, ge=0, le=127)


class RemoveTrack(_Strict):
    op: Literal["remove_track"]
    name: str


OP_TYPES = (
    Transpose, Quantize, InsertNote, DeleteNotes, MoveNote, SetVelocity, SetDuration,
    SetTempo, SetTimeSignature, ReplaceBars, AddTrack, RemoveTrack,
)
OP_NAMES = [t.model_fields["op"].annotation.__args__[0] for t in OP_TYPES]
Op = Annotated[Union[OP_TYPES], Field(discriminator="op")]
OpList = TypeAdapter(list[Op])


class OpError(Exception):
    def __init__(self, message: str, op_index: int | None = None, op: dict | None = None):
        super().__init__(message)
        self.message = message
        self.op_index = op_index
        self.op = op

    def to_dict(self) -> dict:
        d: dict = {"message": self.message}
        if self.op_index is not None:
            d["op_index"] = self.op_index
            d["op"] = self.op
        d["note"] = "No changes were written; the file is unchanged."
        return d


def parse_ops(raw: list[dict]) -> list:
    if not isinstance(raw, list):
        raise OpError("ops must be a JSON array of op objects")
    for i, item in enumerate(raw):
        try:
            OpList.validate_python([item])
        except ValidationError as e:
            errs = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:]) or 'op'}: {err['msg']}" for err in e.errors()[:5])
            raise OpError(f"invalid op: {errs}", i, item) from None
    return OpList.validate_python(raw)


# ---------------------------------------------------------------- selection


def _track_name(score: Score, name: str) -> str:
    if name in score.tracks:
        return name
    lower = {t.lower(): t for t in score.tracks}
    if name.lower() in lower:
        return lower[name.lower()]
    raise OpError(f"unknown track {name!r}; tracks are: {', '.join(score.tracks)} (use add_track to create one)")


def _matches(score: Score, n: Note, sel: Selector) -> bool:
    bm = score.bars
    tracks = list(sel.tracks or []) + ([sel.track] if sel.track else [])
    if tracks and n.track not in {_track_name(score, t) for t in tracks}:
        return False
    if sel.bars or sel.bar is not None:
        bar = bm.bar_of(n.start)
        if sel.bars and not sel.bars[0] <= bar <= sel.bars[1]:
            return False
        if sel.bar is not None and bar != sel.bar:
            return False
    if sel.beat is not None:
        if sel.bar is None:
            raise OpError("selector.beat requires selector.bar")
        if abs(bm.position(n.start)[1] - sel.beat) > EPS:
            return False
    pitches = list(sel.pitches or []) + ([sel.pitch] if sel.pitch is not None else [])
    if pitches and n.pitch not in {parse_pitch(p) for p in pitches}:
        return False
    return True


def _select(score: Score, sel: Selector) -> list[Note]:
    return [n for ns in score.tracks.values() for n in ns if _matches(score, n, sel)]


def _beat(score: Score, bar: int, beat: float) -> float:
    try:
        return score.bars.beat_of(bar, beat)
    except ValueError as e:
        raise OpError(str(e)) from None


def _replace_notes(score: Score, old: list[Note], new: list[Note]) -> None:
    gone = Counter(old)
    for name in score.tracks:
        kept = []
        for n in score.tracks[name]:
            if gone[n] > 0:
                gone[n] -= 1
            else:
                kept.append(n)
        score.tracks[name] = kept
    for n in new:
        score.tracks[n.track].append(n)
    score.sort()


def _check(n: Note) -> Note:
    if not 0 <= n.pitch <= 127:
        raise OpError(f"result pitch {n.pitch} is outside MIDI range 0-127 (from a note at beat {fmt_num(n.start)} in {n.track})")
    if n.start < -EPS:
        raise OpError(f"a note in {n.track} would start before the beginning of the piece")
    return n


# ---------------------------------------------------------------- apply


def _apply_one(score: Score, op) -> None:
    if isinstance(op, Transpose):
        sel = _select(score, op.scope)
        _replace_notes(score, sel, [_check(replace(n, pitch=n.pitch + op.semitones)) for n in sel])
    elif isinstance(op, Quantize):
        sel = _select(score, op.scope)
        new = []
        for n in sel:
            target = round(n.start / op.grid) * op.grid
            start = round(n.start + (target - n.start) * op.strength, 6)
            dur = n.dur
            if op.ends:
                end_t = round(n.end / op.grid) * op.grid
                end = n.end + (end_t - n.end) * op.strength
                dur = round(end - start, 6)
                if dur <= EPS:
                    dur = op.grid  # a note shorter than the grid snapped to zero length
            new.append(_check(replace(n, start=max(start, 0.0), dur=dur)))
        _replace_notes(score, sel, new)
    elif isinstance(op, InsertNote):
        track = _track_name(score, op.track)
        n = Note(round(_beat(score, op.bar, op.beat), 6), parse_pitch(op.pitch), op.dur, op.vel, track, score.channels.get(track, 0))
        _replace_notes(score, [], [_check(n)])
    elif isinstance(op, DeleteNotes):
        sel = _select(score, op.selector)
        if not sel:
            raise OpError("delete_notes selector matched no notes")
        _replace_notes(score, sel, [])
    elif isinstance(op, MoveNote):
        sel = _select(score, op.selector)
        if not sel:
            raise OpError("move_note selector matched no notes")
        _replace_notes(score, sel, [_check(replace(n, pitch=n.pitch + op.dpitch, start=round(n.start + op.dbeat, 6))) for n in sel])
    elif isinstance(op, SetVelocity):
        sel = _select(score, op.selector)
        _replace_notes(score, sel, [replace(n, vel=op.value) for n in sel])
    elif isinstance(op, SetDuration):
        sel = _select(score, op.selector)
        _replace_notes(score, sel, [replace(n, dur=op.dur) for n in sel])
    elif isinstance(op, SetTempo):
        at = round(_beat(score, op.at_bar, op.at_beat), 6)
        score.tempos = sorted([t for t in score.tempos if abs(t.beat - at) > EPS] + [Tempo(at, float(op.bpm))], key=lambda t: t.beat)
    elif isinstance(op, SetTimeSignature):
        at = round(_beat(score, op.at_bar, 1.0), 6)
        score.time_sigs = sorted([t for t in score.time_sigs if abs(t.beat - at) > EPS] + [TimeSig(at, op.num, op.den)], key=lambda t: t.beat)
    elif isinstance(op, ReplaceBars):
        track = _track_name(score, op.track)
        lo, hi = op.bars
        if lo > hi:
            raise OpError("bars must be [first, last] with first <= last")
        old = _select(score, Selector(track=track, bars=(lo, hi)))
        new = []
        for bn in op.notes:
            if not lo <= bn.bar <= hi:
                raise OpError(f"replace_bars note at bar {bn.bar} is outside bars {lo}-{hi}")
            start = round(_beat(score, bn.bar, bn.beat), 6)
            new.append(_check(Note(start, parse_pitch(bn.pitch), bn.dur, bn.vel, track, score.channels.get(track, 0))))
        _replace_notes(score, old, new)
    elif isinstance(op, AddTrack):
        if op.name in score.tracks:
            raise OpError(f"track {op.name!r} already exists")
        score.tracks[op.name] = []
        score.channels[op.name] = op.channel - 1
        if op.program is not None:
            score.programs[op.name] = op.program
    elif isinstance(op, RemoveTrack):
        name = _track_name(score, op.name)
        del score.tracks[name]
        score.channels.pop(name, None)
        score.programs.pop(name, None)
    else:  # pragma: no cover
        raise OpError(f"unhandled op {op!r}")


def apply_ops(score: Score, raw_ops: list[dict]) -> Score:
    ops = parse_ops(raw_ops)
    out = score.copy()
    for i, op in enumerate(ops):
        try:
            _apply_one(out, op)
        except OpError as e:
            e.op_index, e.op = i, raw_ops[i]
            raise
        except ValueError as e:
            raise OpError(str(e), i, raw_ops[i]) from None
    return out


# ---------------------------------------------------------------- diff


def diff(before: Score, after: Score, max_listed: int = 24) -> dict:
    """Notes added/removed per track and bar, plus tempo/meter/track changes."""
    bm_before, bm_after = before.bars, after.bars
    out: dict = {"tracks": {}, "listed": []}
    names = list(dict.fromkeys(list(before.tracks) + list(after.tracks)))
    listed = []
    for name in names:
        b = Counter(n.key(with_vel=True) for n in before.tracks.get(name, []))
        a = Counter(n.key(with_vel=True) for n in after.tracks.get(name, []))
        removed, added = b - a, a - b
        if not removed and not added:
            continue
        per_bar: dict[int, dict[str, int]] = defaultdict(lambda: {"added": 0, "removed": 0})
        for k, c in removed.items():
            per_bar[bm_before.bar_of(k[2])]["removed"] += c
            listed.append(("-", bm_before, k))
        for k, c in added.items():
            per_bar[bm_after.bar_of(k[2])]["added"] += c
            listed.append(("+", bm_after, k))
        out["tracks"][name] = {
            "added": sum(added.values()),
            "removed": sum(removed.values()),
            "bars_touched": _ranges(sorted(per_bar)),
        }
    tracks_added = [t for t in after.tracks if t not in before.tracks]
    tracks_removed = [t for t in before.tracks if t not in after.tracks]
    if tracks_added or tracks_removed:
        out["track_list"] = {"added": tracks_added, "removed": tracks_removed}
    if before.tempos != after.tempos:
        out["tempo"] = [f"{fmt_num(t.bpm)} bpm at bar {bm_after.position(t.beat)[0]} beat {fmt_num(bm_after.position(t.beat)[1])}" for t in after.tempos]
    if before.time_sigs != after.time_sigs:
        out["time_signatures"] = [f"{t.num}/{t.den} from bar {bm_after.bar_of(t.beat)}" for t in after.time_sigs]
    listed.sort(key=lambda x: (x[2][2], x[0]))
    for sign, bm, (track, pitch, start, dur, vel) in listed[:max_listed]:
        bar, beat = bm.position(start)
        out["listed"].append(f"{sign} [{track}] bar {bar} beat {fmt_num(beat)} {pitch_name(pitch)} dur {fmt_num(dur)} vel {vel}")
    if len(listed) > max_listed:
        out["listed"].append(f"... and {len(listed) - max_listed} more note changes")
    if not out["tracks"] and "tempo" not in out and "time_signatures" not in out and "track_list" not in out:
        out["summary"] = "no changes"
    return out


def _ranges(bars: list[int]) -> str:
    if not bars:
        return ""
    parts, start, prev = [], bars[0], bars[0]
    for b in bars[1:]:
        if b == prev + 1:
            prev = b
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = b
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)
