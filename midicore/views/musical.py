"""musical view: notes grouped by bar and track, plus derived analysis.

Per bar: one chord label per beat (roman numeral in the estimated key, plus the
chord name). Per track line: notes as NAME(dur) in sequence, where each note
starts where the previous one ended. Gaps are written r(dur); a note that starts
anywhere else is prefixed with @beat. Notes struck together with the same
duration and velocity are joined with '+'. The line's default velocity is
written once (v80); a note that differs carries its own (C5(1,v64)).
Everything after ' | deg' is annotation and is ignored by parse().
"""

from __future__ import annotations

import re
from collections import Counter

from ..analysis import Key, chord_segments, is_drum, key_candidates, names_for_key, roman
from ..model import EPS, Note, Score, fmt_num, parse_pitch, pitch_name
from . import notes as notes_view
from .common import select


def degree(pc: int, key: Key) -> str:
    scale = key.scale
    if pc in scale:
        return str(scale.index(pc) + 1)
    if key.mode == "major" and (pc - key.tonic) % 12 == 6:
        return "#4"
    if (pc + 1) % 12 in scale:
        return "b" + str(scale.index((pc + 1) % 12) + 1)
    return "#" + str(scale.index((pc - 1) % 12) + 1)


def _groups(ns: list[Note]) -> list[list[Note]]:
    groups: list[list[Note]] = []
    for n in sorted(ns, key=lambda n: (n.start, n.pitch)):
        g = groups[-1] if groups else None
        if g and abs(g[0].start - n.start) < EPS and abs(g[0].dur - n.dur) < EPS and g[0].vel == n.vel:
            g.append(n)
        else:
            groups.append([n])
    return groups


def _track_line(ns: list[Note], bar_start: float, key: Key | None, names: list[str]) -> str:
    default_vel = Counter(n.vel for n in ns).most_common(1)[0][0]
    tokens, degs = [], []
    cursor = bar_start
    for g in _groups(ns):
        start = g[0].start
        if start > cursor + EPS:
            tokens.append(f"r({fmt_num(start - cursor)})")
        elif start < cursor - EPS:
            tokens.append(f"@{fmt_num(start - bar_start + 1.0)}")
        vel = "" if g[0].vel == default_vel else f",v{g[0].vel}"
        tokens.append("+".join(pitch_name(n.pitch, names) for n in g) + f"({fmt_num(g[0].dur)}{vel})")
        if key:
            degs.append("+".join(degree(n.pitch % 12, key) for n in g))
        cursor = start + g[0].dur
    line = f"v{default_vel}: " + " ".join(tokens)
    if degs:
        line += "  | deg " + " ".join(degs)
    return line


def render(score: Score, bars=None, tracks=None) -> str:
    sub = select(score, bars, tracks)
    analysis_notes = [n for name, ns in score.tracks.items() if not is_drum(score, name) for n in ns]
    cands = key_candidates(analysis_notes) if analysis_notes else []
    key = cands[0][0] if cands else None
    names = names_for_key(key)
    head = notes_view.header(sub)
    if key:
        alts = ", ".join(f"{k.name} {c:.2f}" for k, c in cands[1:3])
        head.insert(0, f"estimated key: {key.name} (conf {cands[0][1]:.2f}; next: {alts})")
    lines = head

    chords: dict[int, list[str]] = {}
    if key:
        for bar, _, _, chord in chord_segments(score):
            chords.setdefault(bar, []).append(f"{roman(chord, key)} ({chord.label(names)})" if chord else "-")

    bm = score.bars
    lo, hi = bars if bars else (score.first_bar(), score.n_bars())
    for bar in range(lo, hi + 1):
        if bar == 0 and score.pickup <= EPS:
            continue
        b0 = bm.bar_start(bar)
        b1 = bm.bar_end(bar)
        lines.append(f"bar {bar} | " + " | ".join(chords.get(bar, ["-"])))
        for name, ns in sub.tracks.items():
            in_bar = [n for n in ns if b0 - EPS <= n.start < b1 - EPS]
            if in_bar:
                lines.append(f"  {name} " + _track_line(in_bar, b0, key, names))
    return "\n".join(lines) + "\n"


_BAR = re.compile(r"^bar (-?\d+) \|")
_LINE = re.compile(r"^  (.+?) v(\d+): (.*)$")
_TOKEN = re.compile(r"^([^()]+)\(([\d.]+)(?:,v(\d+))?\)$")


def parse(text: str) -> Score:
    lines = text.splitlines()
    if lines and lines[0].startswith("estimated key:"):
        lines = lines[1:]
    header = [ln for ln in lines if not _BAR.match(ln) and not ln.startswith("  ")]
    score = notes_view.parse("\n".join(header))
    bm = score.bars
    bar_start = 0.0
    for line in lines:
        if m := _BAR.match(line):
            bar_start = bm.bar_start(int(m.group(1)))
        elif m := _LINE.match(line):
            name, default_vel = m.group(1), int(m.group(2))
            body = m.group(3).split("  | deg", 1)[0]
            cursor = bar_start
            pending_at = None
            for tok in body.split():
                if tok.startswith("@"):
                    pending_at = bar_start + float(tok[1:]) - 1.0
                    continue
                tm = _TOKEN.match(tok)
                if not tm:
                    raise ValueError(f"cannot parse token {tok!r} in line {line!r}")
                dur = float(tm.group(2))
                if tm.group(1) == "r":
                    cursor += dur
                    continue
                start = pending_at if pending_at is not None else cursor
                pending_at = None
                vel = int(tm.group(3)) if tm.group(3) else default_vel
                for p in tm.group(1).split("+"):
                    score.tracks[name].append(Note(round(start, 6), parse_pitch(p), dur, vel, name, score.channels[name]))
                cursor = start + dur
    score.sort()
    return score
