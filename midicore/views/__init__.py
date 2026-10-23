"""Text views of a Score. The view is the main experimental variable (RQ1).

Every view carries the same information about notes, tempo, meter and key
signatures; only the format differs. Each module exposes render() and parse(),
and tests assert parse(render(score)) == score for all of them.
"""

from __future__ import annotations

from ..model import Score
from . import musical, notes, raw
from .common import parse_bars, resolve_tracks, select

__all__ = ["VIEWS", "parse_bars", "render", "resolve_tracks", "select"]

VIEWS = {"raw": raw, "notes": notes, "musical": musical}


def render(score: Score, view: str, bars: tuple[int, int] | None = None, tracks: list[str] | None = None) -> str:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; use one of {', '.join(VIEWS)}")
    return VIEWS[view].render(score, bars=bars, tracks=tracks)
