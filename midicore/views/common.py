"""Selection helpers shared by all views."""

from __future__ import annotations

from ..model import EPS, Score


def parse_bars(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    spec = spec.strip()
    if "-" in spec[1:]:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
    else:
        lo = hi = int(spec)
    if lo > hi:
        raise ValueError(f"bar range {spec!r} is reversed")
    return lo, hi


def resolve_tracks(score: Score, names: list[str] | None) -> list[str] | None:
    """Exact match first, then case-insensitive. Unknown names are an error."""
    if not names:
        return None
    out = []
    lower = {t.lower(): t for t in score.tracks}
    for name in names:
        name = name.strip()
        if name in score.tracks:
            out.append(name)
        elif name.lower() in lower:
            out.append(lower[name.lower()])
        else:
            raise ValueError(f"unknown track {name!r}; tracks are: {', '.join(score.tracks)}")
    return out


def select(score: Score, bars: tuple[int, int] | None = None, tracks: list[str] | None = None) -> Score:
    """Keep notes whose onset lies in the bar range and whose track is listed."""
    out = score.copy()
    if tracks is not None:
        out.tracks = {t: ns for t, ns in out.tracks.items() if t in tracks}
    if bars is not None:
        bm = score.bars
        lo = bm.bar_start(bars[0]) if bars[0] > 0 or score.pickup > EPS else 0.0
        hi = bm.bar_end(bars[1])
        out.tracks = {t: [n for n in ns if lo - EPS <= n.start < hi - EPS] for t, ns in out.tracks.items()}
    return out
