"""Coherence-slaving collapse for the truncated-Wigner reduction.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from twa_facilitation_dp_support import evolve_single_site

OUT = Path("twa_dp_support_results")
OMEGA = 1.0
GAMMAS = [8.0, 12.0, 16.0, 20.0, 24.0, 32.0]


def main():
    data = {}
    for g in GAMMAS:
        k = OMEGA ** 2 / g
        t_max = 1.6 / k
        dt = min(0.008, 0.05 / g)
        sample = max(1, int(round((t_max / 240) / dt)))
        t, n, c = evolve_single_site(OMEGA, g, 0.0, t_max, dt, 30000, sample,
                                     seed=20250 + int(g))
        data[g] = (t, n, c)
        print(f"  gamma={g:5.1f}  k={k:.4f}  n_end={n[-1]:.3f}  "
              f"coh_peak*g/Omega={np.max(c) * g / OMEGA:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    flat = {"gammas": np.array(GAMMAS), "omega": OMEGA}
    for g, (t, n, c) in data.items():
        flat[f"t_{g:g}"] = t
        flat[f"n_{g:g}"] = n
        flat[f"c_{g:g}"] = c
    np.savez(OUT / "twa_coherence_collapse.npz", **flat)
    print(f"Wrote {OUT / 'twa_coherence_collapse.npz'}")


if __name__ == "__main__":
    main()
