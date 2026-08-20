"""Conjugate-field equation of state (Widom--Griffiths) simulation.
A spontaneous-creation field h is switched on.  
This is the order parameter as a function of its conjugate field and the distance
to criticality -- the ingredients of the directed-percolation equation of state
        rho = |Delta|^beta  H( h / |Delta|^sigma ).
"""
from __future__ import annotations

import sys
import time
import numpy as np
import concurrent.futures

from new_sims_core import PARAMS, build_shifts, update_grid

LAMBDA_C = 2.008

LAMBDA_VALUES = [1.95, 1.97, 1.99, 2.008, 2.03, 2.05, 2.07]
H_VALUES = [3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1]

L_EOS = 30
MAX_T = 160.0
REPLICATES = 4
BASE_SEED = 20260606
WORKERS = 9


def _worker(args):
    (lam, h, rep, seed, L, max_steps, dt, p) = args
    rng = np.random.default_rng(seed)
    shifts = build_shifts(p["r_cut"], p["d_0"])
    grid = np.zeros((L, L, L), dtype=np.int8)
    grid[rng.random((L, L, L), dtype=np.float32) < 0.5] = 1
    rho = np.empty(max_steps)
    for step in range(max_steps):
        rho[step] = np.count_nonzero(grid == 1) / grid.size
        grid, _spread, _loss = update_grid(
            grid, shifts, dt, lam, p["alpha"], p["gamma_base"], p["beta"],
            p["gamma_loss"], rng, h_field=h)
    lo = max_steps // 2
    rho_stat = float(rho[lo:].mean())
    return (lam, h, rep, rho_stat)


def main():
    quick = "--quick" in sys.argv
    lam_values = [1.99, 2.008, 2.03] if quick else LAMBDA_VALUES
    h_values = [1.0e-3, 1.0e-2, 1.0e-1] if quick else H_VALUES
    reps = 2 if quick else REPLICATES
    max_T = 60.0 if quick else MAX_T

    p = PARAMS
    L = L_EOS
    dt = p["dt"]
    max_steps = int(max_T / dt) + 1

    tasks = []
    k = 0
    for lam in lam_values:
        for h in h_values:
            for rep in range(reps):
                seed = int(BASE_SEED + 1_000_003 * k + 7919 * rep)
                tasks.append((lam, h, rep, seed, L, max_steps, dt, p))
                k += 1

    print(f"EOS: {len(lam_values)} lambda x {len(h_values)} h x {reps} rep "
          f"= {len(tasks)} runs on L={L}^3, max_T={max_T}")
    t0 = time.time()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(_worker, tasks))
    except (PermissionError, OSError, RuntimeError) as e:
        print("serial fallback:", e)
        results = [_worker(a) for a in tasks]
    print(f"  done in {time.time() - t0:.1f}s")

    lam_arr = np.array(lam_values, float)
    h_arr = np.array(h_values, float)
    rho_mean = np.full((len(lam_arr), len(h_arr)), np.nan)
    rho_sem = np.full((len(lam_arr), len(h_arr)), np.nan)
    for i, lam in enumerate(lam_values):
        for j, h in enumerate(h_values):
            vals = np.array([r[3] for r in results
                             if r[0] == lam and r[1] == h], float)
            rho_mean[i, j] = vals.mean()
            rho_sem[i, j] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

    out = "new_results_eos_quick.npz" if quick else "new_results_eos.npz"
    np.savez(out, lambda_values=lam_arr, h_values=h_arr,
             lambda_c=LAMBDA_C, rho_mean=rho_mean, rho_sem=rho_sem,
             L=L, max_T=max_T, replicates=reps,
             gamma_loss=p["gamma_loss"])
    print(f"saved {out}")
    if LAMBDA_C in lam_values:
        ic = lam_values.index(LAMBDA_C)
        x = np.log(h_arr)
        y = np.log(rho_mean[ic])
        sl = np.polyfit(x, y, 1)[0]
        print(f"  critical-isotherm exponent rho~h^x at lambda_c: x = {sl:.3f} "
              f"(mean-field 0.5; (3+1)D DP ~ 0.40)")


if __name__ == "__main__":
    main()
