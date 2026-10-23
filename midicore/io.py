"""Load and save .mid files. The only module that knows about ticks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import mido

from .model import EPS, KeySig, Note, Score, Tempo, TimeSig

PICKUP_MARKER = "midibench:pickup="


def _r(x: float) -> float:
    return round(x, 6)


def load(path: str | Path) -> Score:
    return from_midifile(mido.MidiFile(str(path)))


def from_midifile(mid: mido.MidiFile) -> Score:
    ppq = mid.ticks_per_beat
    score = Score(tracks={}, tempos=[], time_sigs=[], key_sigs=[], ppq=ppq)
    names_seen: dict[str, int] = defaultdict(int)

    for ti, track in enumerate(mid.tracks):
        name = None
        tick = 0
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        notes: list[Note] = []
        programs: dict[int, int] = {}
        for msg in track:
            tick += msg.time
            beat = _r(tick / ppq)
            if msg.type == "track_name" and name is None:
                name = msg.name.strip() or None
            elif msg.type == "set_tempo":
                score.tempos.append(Tempo(beat, round(mido.tempo2bpm(msg.tempo), 3)))
            elif msg.type == "time_signature":
                score.time_sigs.append(TimeSig(beat, msg.numerator, msg.denominator))
            elif msg.type == "key_signature":
                score.key_sigs.append(KeySig(beat, msg.key))
            elif msg.type == "marker" and msg.text.startswith(PICKUP_MARKER):
                score.pickup = float(msg.text[len(PICKUP_MARKER):])
            elif msg.type == "program_change":
                programs.setdefault(msg.channel, msg.program)
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)].append((tick, msg.velocity))
            elif msg.type in ("note_off", "note_on"):
                stack = open_notes.get((msg.channel, msg.note))
                if stack:
                    on_tick, vel = stack.pop(0)
                    if tick > on_tick:
                        notes.append(Note(_r(on_tick / ppq), msg.note, _r((tick - on_tick) / ppq), vel, "", msg.channel))
        if not notes:
            continue
        base = name or f"Track {ti}"
        by_channel: dict[int, list[Note]] = defaultdict(list)
        for n in notes:
            by_channel[n.channel].append(n)
        # Type-0 style tracks that mix channels are split so each part is addressable.
        split = len(by_channel) > 1
        for ch, ch_notes in sorted(by_channel.items()):
            tname = f"{base} ch{ch + 1}" if split else base
            names_seen[tname] += 1
            if names_seen[tname] > 1:
                tname = f"{tname} ({names_seen[tname]})"
            score.tracks[tname] = sorted(Note(n.start, n.pitch, n.dur, n.vel, tname, ch) for n in ch_notes)
            score.channels[tname] = ch
            if ch in programs:
                score.programs[tname] = programs[ch]

    if not score.tempos:
        score.tempos = [Tempo(0.0, 120.0)]
    if not score.time_sigs:
        score.time_sigs = [TimeSig(0.0, 4, 4)]
    score.tempos = _dedupe(sorted(score.tempos, key=lambda t: t.beat))
    score.time_sigs = _dedupe(sorted(score.time_sigs, key=lambda t: t.beat))
    score.key_sigs = _dedupe(sorted(score.key_sigs, key=lambda t: t.beat))
    return score


def _dedupe(events: list) -> list:
    """Keep the last event at any given beat (later events override earlier ones)."""
    out: dict[float, object] = {}
    for e in events:
        out[e.beat] = e
    return list(out.values())


def save(score: Score, path: str | Path) -> None:
    to_midifile(score).save(str(path))


def to_midifile(score: Score) -> mido.MidiFile:
    ppq = score.ppq
    mid = mido.MidiFile(type=1, ticks_per_beat=ppq)

    def tick(beat: float) -> int:
        return max(0, round(beat * ppq))

    conductor: list[tuple[int, int, mido.MetaMessage]] = []
    for ts in score.time_sigs:
        conductor.append((tick(ts.beat), 0, mido.MetaMessage("time_signature", numerator=ts.num, denominator=ts.den, time=0)))
    for ks in score.key_sigs:
        conductor.append((tick(ks.beat), 1, mido.MetaMessage("key_signature", key=ks.key, time=0)))
    for t in score.tempos:
        conductor.append((tick(t.beat), 2, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(t.bpm), time=0)))
    if score.pickup > EPS:
        conductor.append((0, 3, mido.MetaMessage("marker", text=f"{PICKUP_MARKER}{score.pickup:g}", time=0)))
    mid.tracks.append(_to_track("Conductor", conductor))

    for name, notes in score.tracks.items():
        events: list[tuple[int, int, mido.Message]] = []
        ch = score.channels.get(name, notes[0].channel if notes else 0)
        if name in score.programs:
            events.append((0, -1, mido.Message("program_change", channel=ch, program=score.programs[name], time=0)))
        for n in notes:
            on, off = tick(n.start), tick(n.end)
            if off <= on:
                off = on + 1
            # note_off sorts before note_on at the same tick so repeated pitches re-articulate.
            events.append((on, 1, mido.Message("note_on", channel=n.channel, note=n.pitch, velocity=n.vel, time=0)))
            events.append((off, 0, mido.Message("note_off", channel=n.channel, note=n.pitch, velocity=0, time=0)))
        mid.tracks.append(_to_track(name, events))
    return mid


def _to_track(name: str, events: list[tuple[int, int, mido.Message]]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    last = 0
    for t, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        track.append(msg.copy(time=t - last))
        last = t
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track
