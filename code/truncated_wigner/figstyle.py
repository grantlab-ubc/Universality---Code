r"""Enlarged typography for the figures, and nothing else.
"""
from __future__ import annotations

import os

_MPL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".matplotlib_cache")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

import matplotlib as mpl


ROOM = 1.167

BASE = 10.0
TITLE = 12.0
LABEL = 12.0
TICK = 10.0
LEGEND = 8.0
PANEL = 13.0

ANNOT_SCALE = 1.30

_WEIGHTS = ["lines.linewidth", "lines.markersize", "patch.linewidth",
            "axes.linewidth", "xtick.major.width", "ytick.major.width",
            "xtick.minor.width", "ytick.minor.width", "xtick.major.size",
            "ytick.major.size", "xtick.minor.size", "ytick.minor.size"]


def apply() -> None:
    """Raise the type sizes.  Call after the script's own rcParams block."""
    mpl.rcParams.update({
        "font.size": BASE,
        "axes.titlesize": TITLE,
        "axes.labelsize": LABEL,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND,
        "legend.title_fontsize": LEGEND,
    })
    for key in _WEIGHTS:
        try:
            mpl.rcParams[key] = mpl.rcParams[key] * ROOM
        except (KeyError, TypeError):
            pass


def size(w: float, h: float) -> tuple:
    """The original figsize, enlarged so the bigger type has room."""
    return (w * ROOM, h * ROOM)


def a(orig_pt: float) -> float:
    """Scale an explicitly written annotation/legend point size."""
    return orig_pt * ANNOT_SCALE


def panel_label(ax, letter: str, dx: float = -16.0, dy: float = 6.0) -> None:
    """The paper's bold top-left panel letter, at the enlarged size."""
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx * ROOM, dy * ROOM), textcoords="offset points",
                fontsize=PANEL, fontweight="bold", va="bottom", ha="left",
                annotation_clip=False)
