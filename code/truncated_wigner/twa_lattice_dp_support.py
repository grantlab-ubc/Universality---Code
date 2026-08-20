"""Lattice helpers shared by the truncated-Wigner drivers
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from twa_facilitation_dp_support import evolve_single_site, extract_rate

Array = np.ndarray


def nn_field(active: Array) -> Array:
    """Sum of the six nearest-neighbour occupations on the last three axes."""
    out = np.zeros_like(active)
    for axis in (-3, -2, -1):
        out += np.roll(active, 1, axis=axis) + np.roll(active, -1, axis=axis)
    return out


def make_seed(shape_traj: Tuple[int, ...], L: int, kind: str, half: int = 1) -> Array:
    """Initial sigma^z: ground (-1) everywhere except an active seed cube (+1)."""
    sz = -np.ones(shape_traj, dtype=np.float64)
    c = L // 2
    if kind == "single":
        sz[..., c, c, c] = 1.0
    elif kind == "dense":
        rng = np.random.default_rng(20260619)
        sz = np.where(rng.random(shape_traj) < 0.6, 1.0, -1.0)
    else:
        lo, hi = c - half, c + half + 1
        sz[..., lo:hi, lo:hi, lo:hi] = 1.0
    return sz


def kfac_table(
    gamma_phi: float, omega: float, V: float, Delta: float, m_max: int = 6,
    n_traj: int = 8000, dt: float = 0.002,
) -> Dict[int, float]:
    """DTWA single-site facilitation rate k_fac(m) for m active neighbours.
    """
    table: Dict[int, float] = {}
    for m in range(m_max + 1):
        hz = Delta + m * V
        k_guess = (omega ** 2) * gamma_phi / (gamma_phi ** 2 + hz ** 2)
        t_max = float(np.clip(1.5 / max(k_guess, 1e-4), 30.0, 150.0))
        sample = max(1, int(round((t_max / 300.0) / dt)))
        times, n_up, _ = evolve_single_site(
            omega, gamma_phi, hz, t_max, dt, n_traj, sample, 4242 + 17 * m
        )
        k, _ = extract_rate(times, n_up)
        table[m] = float(k) if (math.isfinite(k) and k > 0) else float(k_guess)
        tag = "DTWA" if (math.isfinite(k) and k > 0) else "Lorentz"
        print(f"    k_fac(m={m})  h_z={hz:+.1f}  k={table[m]:.4e} ({tag})")
    return table


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def replot_from_csv(src: Path, fig_path: Path) -> None:
    transition: Dict[float, List[Tuple[float, float]]] = {}
    compare_raw: Dict[str, List[Tuple[float, float]]] = {}
    gamma_strong = 16.0
    kappa_c_stored = float("nan")
    with (src / "twa_lattice_timeseries.csv").open() as f:
        for rec in csv.DictReader(f):
            t, a = float(rec["t"]), float(rec["A"])
            if rec["panel"] == "transition":
                gamma_strong = float(rec["gamma_phi"])
                transition.setdefault(float(rec["kappa"]), []).append((t, a))
            elif rec["panel"] == "compare":
                kappa_c_stored = float(rec["kappa"])
                compare_raw.setdefault(rec["which"], []).append((t, a))

    def _arrays(d):
        out = {}
        for key, vals in d.items():
            vals.sort(key=lambda r: r[0])
            arr = np.array(vals, float)
            out[key] = (arr[:, 0], arr[:, 1])
        return out

    transition_arr = _arrays(transition)
    compare = _arrays(compare_raw)
    kappa_super = 0.6 * kappa_c_stored if math.isfinite(kappa_c_stored) else kappa_c_stored
    make_figure(fig_path, transition_arr, compare, kappa_super, gamma_strong)


if __name__ == "__main__":
    main()
