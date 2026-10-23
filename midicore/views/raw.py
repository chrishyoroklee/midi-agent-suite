"""raw view: the MIDI event stream, delta ticks and all.

Generated from the same serialization io.save writes, so it is exactly what is
in the file (as midicore would save it). Bar/beat arithmetic is left to the reader.
"""

from __future__ import annotations

import re

import mido

from ..io import from_midifile, to_midifile
from ..model import Score
from .common import select


def _fmt(msg: mido.Message | mido.MetaMessage) -> str:
    t = msg.type
    if t in ("note_on", "note_off"):
        return f"{t:<8} ch={msg.channel} n={msg.note} v={msg.velocity}"
    if t == "program_change":
        return f"program_change ch={msg.channel} program={msg.program}"
    if t == "set_tempo":
        return f"set_tempo bpm={round(mido.tempo2bpm(msg.tempo), 3):g}"
    if t == "time_signature":
        return f"time_signature {msg.numerator}/{msg.denominator}"
    if t == "key_signature":
        return f"key_signature {msg.key}"
    if t == "marker":
        return f'marker "{msg.text}"'
    raise ValueError(f"unexpected message {msg}")


def render(score: Score, bars=None, tracks=None) -> str:
    sub = select(score, bars, tracks)
    mid = to_midifile(sub)
    lines = [f"midi type=1 ppq={mid.ticks_per_beat} tracks={len(mid.tracks)}"]
    if bars or tracks:
        lines.append(f"# filtered: bars={bars or 'all'} tracks={tracks or 'all'}; dt is still relative to the previous shown event, starting from tick 0")
    for i, track in enumerate(mid.tracks):
        lines.append(f'track {i} "{track.name}"')
        for msg in track:
            if msg.type in ("track_name", "end_of_track"):
                continue
            lines.append(f"dt={msg.time:<6} {_fmt(msg)}")
    return "\n".join(lines) + "\n"


_EVENT = re.compile(r"^dt=(\d+)\s+(\S+)\s*(.*)$")
_KV = re.compile(r"(\w+)=(\S+)")


def parse(text: str) -> Score:
    mid = None
    track = None
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("midi "):
            ppq = int(dict(_KV.findall(line))["ppq"])
            mid = mido.MidiFile(type=1, ticks_per_beat=ppq)
        elif line.startswith("track "):
            name = line.split('"', 1)[1].rsplit('"', 1)[0]
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("track_name", name=name, time=0))
            mid.tracks.append(track)
        else:
            m = _EVENT.match(line)
            if not m:
                raise ValueError(f"cannot parse raw line: {line!r}")
            dt, kind, rest = int(m.group(1)), m.group(2), m.group(3)
            kv = dict(_KV.findall(rest))
            if kind in ("note_on", "note_off"):
                msg = mido.Message(kind, channel=int(kv["ch"]), note=int(kv["n"]), velocity=int(kv["v"]), time=dt)
            elif kind == "program_change":
                msg = mido.Message(kind, channel=int(kv["ch"]), program=int(kv["program"]), time=dt)
            elif kind == "set_tempo":
                msg = mido.MetaMessage(kind, tempo=mido.bpm2tempo(float(kv["bpm"])), time=dt)
            elif kind == "time_signature":
                num, den = rest.split("/")
                msg = mido.MetaMessage(kind, numerator=int(num), denominator=int(den), time=dt)
            elif kind == "key_signature":
                msg = mido.MetaMessage(kind, key=rest.strip(), time=dt)
            elif kind == "marker":
                msg = mido.MetaMessage(kind, text=rest.strip().strip('"'), time=dt)
            else:
                raise ValueError(f"unknown event {kind!r}")
            track.append(msg)
    return from_midifile(mid)
