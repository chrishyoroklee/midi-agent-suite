"""Constraint checks shared by the agent's midi_verify tool and the graders.

Each check takes the edited score, the original score (None if unavailable)
and keyword arguments, and returns a CheckResult with a hard pass/fail plus a
0-1 partial-credit score. Scope arguments, shared by most checks:

    tracks:          list of track names, or "all" (default)
    exclude_tracks:  list of track names to leave out
    bars:            "all" (default), [first, last], or [[a, b], [c, d], ...];
                     last may be null to mean "to the end". A note is in scope
                     when its onset is in one of the ranges.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable

from .analysis import Key, estimate_key, onset_deviation
from .model import EPS, Note, Score, coerce_bar_range, fmt_num, parse_pitch, pitch_name


@dataclass
class CheckResult:
    passed: bool
    score: float
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(d["score"], 4)
        if not d["data"]:
            d.pop("data")
        return d


class CheckError(ValueError):
    pass


@dataclass
class _Check:
    fn: Callable[..., CheckResult]
    needs_input: bool
    doc: str


CHECKS: dict[str, _Check] = {}


def check(name: str, needs_input: bool = False):
    def deco(fn):
        CHECKS[name] = _Check(fn, needs_input, (fn.__doc__ or "").strip().splitlines()[0])
        return fn

    return deco


# ---------------------------------------------------------------- scope


def _track_list(score: Score, value) -> list[str] | None:
    if value in (None, "all"):
        return None
    if isinstance(value, str):
        value = [value]
    lower = {t.lower(): t for t in score.tracks}
    return [lower.get(str(v).lower(), str(v)) for v in value]


def _bar_ranges(value) -> list[tuple[int, int | None]] | None:
    value = coerce_bar_range(value)
    if value in (None, "all"):
        return None
    if isinstance(value, int):
        return [(value, value)]
    if len(value) == 2 and all(v is None or isinstance(v, int) for v in value):
        return [(value[0], value[1])]
    return [(r[0], r[1]) for r in value]


def in_scope(score: Score, tracks=None, exclude_tracks=None, bars=None) -> list[Note]:
    names = _track_list(score, tracks)
    excluded = set(_track_list(score, exclude_tracks) or [])
    ranges = _bar_ranges(bars)
    bm = score.bars
    out = []
    for name, ns in score.tracks.items():
        if (names is not None and name not in names) or name in excluded:
            continue
        for n in ns:
            if ranges is not None:
                bar = bm.bar_of(n.start)
                if not any(lo <= bar and (hi is None or bar <= hi) for lo, hi in ranges):
                    continue
            out.append(n)
    return out


def _scope_kwargs(args: dict) -> dict:
    """Pop scope arguments out of a check's kwargs; a single `track` is accepted too."""
    if "track" in args:
        args["tracks"] = [args.pop("track")]
    return {k: args.pop(k) for k in ("tracks", "exclude_tracks", "bars") if k in args}


def _require_input(inp: Score | None, name: str) -> Score:
    if inp is None:
        raise CheckError(f"{name} needs the original file to compare against")
    return inp


def _describe(notes: list[Note], score: Score, limit: int = 6) -> str:
    bm = score.bars
    parts = []
    for n in notes[:limit]:
        bar, beat = bm.position(n.start)
        parts.append(f"[{n.track}] bar {bar} beat {fmt_num(beat)} {pitch_name(n.pitch)}")
    more = f" (+{len(notes) - limit} more)" if len(notes) > limit else ""
    return "; ".join(parts) + more


# ---------------------------------------------------------------- checks


@check("preserve", needs_input=True)
def preserve(out: Score, inp: Score | None, velocity: bool = False, **args) -> CheckResult:
    """Notes in scope are identical to the original (pitch, start, dur; velocity optional)."""
    inp = _require_input(inp, "preserve")
    scope = _scope_kwargs(args)
    a = Counter(n.key(velocity) for n in in_scope(inp, **scope))
    b = Counter(n.key(velocity) for n in in_scope(out, **scope))
    if a == b:
        return CheckResult(True, 1.0, f"{sum(a.values())} notes unchanged")
    inter, union = sum((a & b).values()), sum((a | b).values())
    missing = [n for n in in_scope(inp, **scope) if (a - b)[n.key(velocity)] > 0]
    extra = [n for n in in_scope(out, **scope) if (b - a)[n.key(velocity)] > 0]
    detail = f"{sum((a - b).values())} original notes changed or removed, {sum((b - a).values())} notes added"
    if missing:
        detail += f"; e.g. missing {_describe(missing, inp, 3)}"
    if extra:
        detail += f"; e.g. added {_describe(extra, out, 3)}"
    return CheckResult(False, inter / union if union else 1.0, detail)


@check("transpose_exact", needs_input=True)
def transpose_exact(out: Score, inp: Score | None, semitones: int, allow_octave: bool = False, **args) -> CheckResult:
    """Notes in scope equal the original shifted by exactly N semitones (any octave if allow_octave)."""
    inp = _require_input(inp, "transpose_exact")
    scope = _scope_kwargs(args)
    src = in_scope(inp, **scope)
    dst = Counter(n.key() for n in in_scope(out, **scope))
    shifts = [semitones + 12 * k for k in range(-4, 5)] if allow_octave else [semitones]
    best = (-1.0, semitones)
    for s in shifts:
        want = Counter((n.track, n.pitch + s, round(n.start, 4), round(n.dur, 4)) for n in src)
        inter = sum((want & dst).values())
        denom = max(sum(want.values()), sum(dst.values()), 1)
        best = max(best, (inter / denom, s))
    score, s = best
    if score >= 1 - EPS:
        return CheckResult(True, 1.0, f"{len(src)} notes shifted by {s:+d} semitones", {"semitones": s})
    return CheckResult(False, max(score, 0.0), f"best match is a {s:+d} shift covering {score:.0%} of notes in scope", {"semitones": s})


@check("range")
def range_check(out: Score, inp: Score | None, low: int | str = 0, high: int | str = 127, **args) -> CheckResult:
    """All notes in scope lie within [low, high]."""
    lo, hi = parse_pitch(low), parse_pitch(high)
    notes = in_scope(out, **_scope_kwargs(args))
    bad = [n for n in notes if not lo <= n.pitch <= hi]
    if not notes:
        return CheckResult(True, 1.0, "no notes in scope")
    if not bad:
        span = f"{pitch_name(min(n.pitch for n in notes))}-{pitch_name(max(n.pitch for n in notes))}"
        return CheckResult(True, 1.0, f"all {len(notes)} notes within {pitch_name(lo)}-{pitch_name(hi)} (actual {span})")
    return CheckResult(False, 1 - len(bad) / len(notes), f"{len(bad)} notes outside {pitch_name(lo)}-{pitch_name(hi)}: {_describe(bad, out)}")


@check("tempo")
def tempo(out: Score, inp: Score | None, bpm: float, bars=None, tolerance: float = 0.5) -> CheckResult:
    """The tempo in effect throughout the given bars (default: whole piece) is bpm."""
    bm = out.bars
    ranges = _bar_ranges(bars) or [(out.first_bar(), None)]
    wrong = []
    for lo, hi in ranges:
        start = bm.bar_start(lo) if lo > 0 else 0.0
        end = bm.bar_end(hi) if hi is not None else float("inf")
        active = [t for t in out.tempos if t.beat <= start + EPS]
        in_effect = [active[-1]] if active else []
        in_effect += [t for t in out.tempos if start + EPS < t.beat < end - EPS]
        wrong += [t for t in in_effect if abs(t.bpm - bpm) > tolerance]
        if not in_effect:
            wrong.append(None)
    if not wrong:
        return CheckResult(True, 1.0, f"tempo is {fmt_num(bpm)} bpm in scope")
    found = ", ".join(f"{fmt_num(t.bpm)} at bar {bm.bar_of(t.beat)}" for t in wrong if t)
    return CheckResult(False, 0.0, f"expected {fmt_num(bpm)} bpm; found {found or 'no tempo'}")


@check("quantize")
def quantize(out: Score, inp: Score | None, grid: float, tolerance: float = 0.01, **args) -> CheckResult:
    """Every onset in scope is within tolerance (beats) of the grid."""
    notes = in_scope(out, **_scope_kwargs(args))
    if not notes:
        return CheckResult(False, 0.0, "no notes in scope")
    devs = [onset_deviation(n.start, grid) for n in notes]
    off = [n for n, d in zip(notes, devs) if d > tolerance]
    mean_dev = sum(devs) / len(devs)
    data = {"max_deviation": round(max(devs), 4), "mean_deviation": round(mean_dev, 4)}
    if not off:
        return CheckResult(True, 1.0, f"all {len(notes)} onsets on the {fmt_num(grid)}-beat grid", data)
    return CheckResult(False, 1 - len(off) / len(notes), f"{len(off)} of {len(notes)} onsets off the {fmt_num(grid)}-beat grid: {_describe(off, out)}", data)


@check("key")
def key_check(out: Score, inp: Score | None, key: str, diatonic: bool = False, **args) -> CheckResult:
    """Estimated key of the scope is the target; with diatonic, every pitch is in the scale too."""
    target = Key.parse(key)
    notes = in_scope(out, **_scope_kwargs(args))
    if not notes:
        return CheckResult(False, 0.0, "no notes in scope")
    est, conf = estimate_key(notes)
    key_ok = est == target
    score = 1.0 if key_ok else 0.0
    detail = f"estimated {est.name} (conf {conf:.2f}), target {target.name}"
    passed = key_ok
    if diatonic:
        outside = [n for n in notes if n.pitch % 12 not in target.scale_pcs()]
        frac = 1 - len(outside) / len(notes)
        score = 0.5 * score + 0.5 * frac
        passed = passed and not outside
        if outside:
            detail += f"; {len(outside)} notes outside the scale: {_describe(outside, out)}"
    return CheckResult(passed, score, detail)


@check("empty")
def empty(out: Score, inp: Score | None, **args) -> CheckResult:
    """There are no notes in scope."""
    scope = _scope_kwargs(args)
    notes = in_scope(out, **scope)
    if not notes:
        return CheckResult(True, 1.0, "no notes in scope")
    before = len(in_scope(inp, **scope)) if inp is not None else len(notes)
    return CheckResult(False, max(0.0, 1 - len(notes) / max(before, 1)), f"{len(notes)} notes remain: {_describe(notes, out)}")


@check("velocity")
def velocity(out: Score, inp: Score | None, value: int, tolerance: int = 0, **args) -> CheckResult:
    """Every note in scope has the given velocity."""
    notes = in_scope(out, **_scope_kwargs(args))
    if not notes:
        return CheckResult(False, 0.0, "no notes in scope")
    bad = [n for n in notes if abs(n.vel - value) > tolerance]
    if not bad:
        return CheckResult(True, 1.0, f"all {len(notes)} notes at velocity {value}")
    return CheckResult(False, 1 - len(bad) / len(notes), f"{len(bad)} of {len(notes)} notes not at velocity {value}")


@check("contains")
def contains(out: Score, inp: Score | None, track: str, bar: int, pitch: int | str, beat: float = 1.0, dur: float | None = None) -> CheckResult:
    """A specific note exists (track, bar, beat, pitch, optionally duration)."""
    p = parse_pitch(pitch)
    names = _track_list(out, [track])
    try:
        start = out.bars.beat_of(bar, beat)
    except ValueError as e:
        raise CheckError(str(e)) from None
    hits = [n for n in out.tracks.get(names[0], []) if n.pitch == p and abs(n.start - start) < 1e-3]
    if dur is not None:
        exact = [n for n in hits if abs(n.dur - dur) < 1e-3]
        if exact:
            return CheckResult(True, 1.0, "note found")
        if hits:
            return CheckResult(False, 0.5, f"note found but duration is {fmt_num(hits[0].dur)}, expected {fmt_num(dur)}")
    elif hits:
        return CheckResult(True, 1.0, "note found")
    return CheckResult(False, 0.0, f"no {pitch_name(p)} in {track} at bar {bar} beat {fmt_num(beat)}")


@check("preserve_pitches", needs_input=True)
def preserve_pitches(out: Score, inp: Score | None, **args) -> CheckResult:
    """Same number of notes with the same pitches in scope; timing may change."""
    inp = _require_input(inp, "preserve_pitches")
    scope = _scope_kwargs(args)
    a = Counter((n.track, n.pitch) for n in in_scope(inp, **scope))
    b = Counter((n.track, n.pitch) for n in in_scope(out, **scope))
    if a == b:
        return CheckResult(True, 1.0, f"{sum(a.values())} notes, same pitches")
    inter, union = sum((a & b).values()), sum((a | b).values())
    return CheckResult(False, inter / union if union else 1.0, f"{sum((a - b).values())} notes lost, {sum((b - a).values())} notes gained")


# ---------------------------------------------------------------- driver


def run_check(name: str, out: Score, inp: Score | None, args: dict | None = None) -> CheckResult:
    if name not in CHECKS:
        raise CheckError(f"unknown check {name!r}; available: {', '.join(sorted(CHECKS))}")
    try:
        return CHECKS[name].fn(out, inp, **dict(args or {}))
    except TypeError as e:
        raise CheckError(f"bad arguments for {name}: {e}") from None


def run_checks(constraints: list[dict], out: Score, inp: Score | None) -> list[dict]:
    """constraints: [{"name": "range", "args": {...}}, ...] -> per-constraint results."""
    results = []
    for c in constraints:
        name = c.get("name") or c.get("check")
        try:
            r = run_check(name, out, inp, c.get("args", {})).to_dict()
        except (CheckError, ValueError) as e:
            r = {"passed": False, "score": 0.0, "detail": f"error: {e}", "error": True}
        results.append({"name": name, "args": c.get("args", {}), **r})
    return results
