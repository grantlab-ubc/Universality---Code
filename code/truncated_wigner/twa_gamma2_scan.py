"""Width of the Lorentzian activation window against dephasing
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from twa_facilitation_dp_support import RunConfig, detuning_sweep

OLD = Path("twa_dp_support_results")
OUT = Path("twa_dp_support_results_revised")
CACHE = OUT / "gamma2_scan.npz"

GAMMAS_NEW = [3.0, 12.0, 24.0]


def lorentz(x, a, G):
    return a * G ** 2 / (G ** 2 + x ** 2)


def fit_gamma2(hz, k):
    p0 = [float(np.nanmax(k)), 2.0 * float(np.median(np.abs(hz))) / 3.0]
    popt, pcov = curve_fit(lorentz, hz, k, p0=p0, maxfev=20000)
    return float(popt[0]), abs(float(popt[1])), float(np.sqrt(pcov[1, 1]))


def main() -> None:
    cfg = RunConfig(n_traj=4000)
    out = {}
    gammas, g2s, g2errs, k0s = [], [], [], []

    with (OLD / "twa_detuning_sweep.csv").open() as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    hz6 = np.array([r["hz"] for r in rows]); k6 = np.array([r["k_fac"] for r in rows])
    a, G, Ge = fit_gamma2(hz6, k6)
    print(f"gamma_phi= 6 (cached): Gamma_2 = {G:.2f} +- {Ge:.2f}")
    gammas.append(6.0); g2s.append(G); g2errs.append(Ge); k0s.append(a)
    out["hz_6"] = hz6; out["k_6"] = k6

    for g in GAMMAS_NEW:
        width = 2.0 * g
        rel = np.array([-2.5, -2.0, -1.5, -1.0, -0.6, -0.3, 0.0,
                        0.3, 0.6, 1.0, 1.5, 2.0, 2.5])
        hz_values = rel * width
        print(f"[sweep] gamma_phi={g:g}, hz in +-{2.5*width:g}")
        rows = detuning_sweep(cfg, g, hz_values)
        hz = np.array([r["hz"] for r in rows])
        k = np.array([r["k_fac"] for r in rows])
        a, G, Ge = fit_gamma2(hz, k)
        print(f"gamma_phi={g:3g}: Gamma_2 = {G:.2f} +- {Ge:.2f}", flush=True)
        gammas.append(g); g2s.append(G); g2errs.append(Ge); k0s.append(a)
        out[f"hz_{g:g}"] = hz; out[f"k_{g:g}"] = k

    order = np.argsort(gammas)
    np.savez(CACHE,
             gammas=np.array(gammas)[order], Gamma2=np.array(g2s)[order],
             Gamma2_err=np.array(g2errs)[order], k0=np.array(k0s)[order], **out)
    print(f"[cache] wrote {CACHE}")


if __name__ == "__main__":
    main()
