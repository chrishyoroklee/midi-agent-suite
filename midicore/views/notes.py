"""notes view: one line per note in musical time (bar, beat, pitch name)."""

from __future__ import annotations

import re

from ..analysis import declared_key, names_for_key
from ..model import KeySig, Note, Score, Tempo, TimeSig, fmt_num, parse_pitch, pitch_name
from .common import select


def header(score: Score) -> list[str]:
    bm = score.bars
    lines = [f"ppq {score.ppq} | pickup {fmt_num(score.pickup)} beats | bars {score.first_bar()}-{score.n_bars()}"]
    for ts in score.time_sigs:
        bar, beat = bm.position(ts.beat)
        lines.append(f"time_signature {ts.num}/{ts.den} at bar {bar} beat {fmt_num(beat)}")
    for t in score.tempos:
        bar, beat = bm.position(t.beat)
        lines.append(f"tempo {fmt_num(t.bpm)} at bar {bar} beat {fmt_num(beat)}")
    for k in score.key_sigs:
        bar, beat = bm.position(k.beat)
        lines.append(f"key_signature {k.key} at bar {bar} beat {fmt_num(beat)}")
    for name, ns in score.tracks.items():
        ch = score.channels.get(name, ns[0].channel if ns else 0)
        prog = f" | program {score.programs[name]}" if name in score.programs else ""
        lines.append(f"track [{name}] | channel {ch + 1}{prog} | {len(ns)} notes")
    return lines


def render(score: Score, bars=None, tracks=None) -> str:
    sub = select(score, bars, tracks)
    bm = score.bars
    names = names_for_key(declared_key(score))
    lines = header(sub)
    rows = sorted((n for ns in sub.tracks.values() for n in ns), key=lambda n: (n.start, list(sub.tracks).index(n.track), -n.pitch))
    for n in rows:
        bar, beat = bm.position(n.start)
        lines.append(f"[{n.track}] bar {bar} beat {fmt_num(beat)}  {pitch_name(n.pitch, names)}  dur {fmt_num(n.dur)}  vel {n.vel}")
    return "\n".join(lines) + "\n"


_AT = r"at bar (-?\d+) beat ([\d.]+)"
_TS = re.compile(rf"^time_signature (\d+)/(\d+) {_AT}$")
_TEMPO = re.compile(rf"^tempo ([\d.]+) {_AT}$")
_KEY = re.compile(rf"^key_signature (\S+) {_AT}$")
_TRACK = re.compile(r"^track \[(.+)\] \| channel (\d+)(?: \| program (\d+))? \|")
_NOTE = re.compile(r"^\[(.+)\] bar (-?\d+) beat ([\d.]+)\s+(\S+)\s+dur ([\d.]+)\s+vel (\d+)$")


def parse(text: str) -> Score:
    score = Score(tracks={}, tempos=[], time_sigs=[], key_sigs=[])
    lines = text.splitlines()
    m = re.match(r"^ppq (\d+) \| pickup ([\d.]+) beats", lines[0])
    score.ppq, score.pickup = int(m.group(1)), float(m.group(2))
    # Time signatures must be known before any bar position can be resolved.
    for line in lines[1:]:
        if m := _TS.match(line):
            beat = _resolve(score, int(m.group(3)), float(m.group(4)))
            score.time_sigs.append(TimeSig(beat, int(m.group(1)), int(m.group(2))))
    for line in lines[1:]:
        if m := _TEMPO.match(line):
            score.tempos.append(Tempo(_resolve(score, int(m.group(2)), float(m.group(3))), float(m.group(1))))
        elif m := _KEY.match(line):
            score.key_sigs.append(KeySig(_resolve(score, int(m.group(2)), float(m.group(3))), m.group(1)))
        elif m := _TRACK.match(line):
            name = m.group(1)
            score.tracks[name] = []
            score.channels[name] = int(m.group(2)) - 1
            if m.group(3) is not None:
                score.programs[name] = int(m.group(3))
        elif m := _NOTE.match(line):
            name = m.group(1)
            start = _resolve(score, int(m.group(2)), float(m.group(3)))
            score.tracks[name].append(Note(start, parse_pitch(m.group(4)), float(m.group(5)), int(m.group(6)), name, score.channels[name]))
    score.sort()
    return score


def _resolve(score: Score, bar: int, beat: float) -> float:
    """bar/beat -> absolute beat using the time signatures parsed so far."""
    bm = Score(time_sigs=score.time_sigs or [TimeSig(0.0, 4, 4)], pickup=score.pickup).bars
    return round(bm.bar_start(bar) + beat - 1.0, 6)
