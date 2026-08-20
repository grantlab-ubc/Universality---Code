"""Avalanche cluster geometry
"""
from __future__ import annotations

import sys
import time
import numpy as np
import concurrent.futures

from new_sims_core import PARAMS, build_shifts, update_grid

LAMBDA_C = 2.008
L_CLUSTER = 40
MAX_T = 250.0
BASE_SEED = 20260607
WORKERS = 9
KEEP_RG_MIN = 3.0


def _worker(args):
    (n_runs, seed, L, max_steps, dt, p) = args
    rng = np.random.default_rng(seed)
    shifts = build_shifts(p["r_cut"], p["d_0"])
    c = L // 2
    S_list, T_list, M_list, Rg_list, bnd_list = [], [], [], [], []
    best_coords = None
    best_Rg = -1.0
    for _ in range(n_runs):
        grid = np.zeros((L, L, L), dtype=np.int8)
        visited = np.zeros((L, L, L), dtype=bool)
        grid[c, c, c] = 1
        visited[c, c, c] = True
        S = 1
        steps = 0
        while steps < max_steps:
            if not np.any(grid == 1):
                break
            grid, spread_mask, _loss = update_grid(
                grid, shifts, dt, LAMBDA_C, p["alpha"], p["gamma_base"],
                p["beta"], p["gamma_loss"], rng)
            newly = spread_mask & ~visited
            visited |= newly
            S += int(np.count_nonzero(spread_mask))
            steps += 1
        coords = np.argwhere(visited)
        M = coords.shape[0]
        ctr = coords.mean(0)
        Rg = float(np.sqrt(np.mean(np.sum((coords - ctr) ** 2, axis=1))))
        touched = bool((coords.min() == 0) or (coords.max() == L - 1))
        S_list.append(S)
        T_list.append(steps * dt)
        M_list.append(M)
        Rg_list.append(Rg)
        bnd_list.append(touched)
        if (not touched) and Rg > best_Rg and M >= 50:
            best_Rg = Rg
            best_coords = coords.astype(np.int16)
    return (np.array(S_list), np.array(T_list), np.array(M_list),
            np.array(Rg_list), np.array(bnd_list, bool), best_coords, best_Rg)


def main():
    quick = "--quick" in sys.argv
    n_runs = 1200 if quick else 16000
    max_T = 120.0 if quick else MAX_T
    p = PARAMS
    L = L_CLUSTER
    dt = p["dt"]
    max_steps = int(max_T / dt) + 1

    per = [n_runs // WORKERS] * WORKERS
    per[0] += n_runs % WORKERS
    ss = np.random.SeedSequence(BASE_SEED)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(WORKERS)]
    tasks = [(per[i], seeds[i], L, max_steps, dt, p)
             for i in range(WORKERS) if per[i] > 0]

    print(f"cluster geometry: {n_runs} avalanches on L={L}^3 at lambda_c={LAMBDA_C}")
    t0 = time.time()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(_worker, tasks))
    except (PermissionError, OSError, RuntimeError) as e:
        print("serial fallback:", e)
        results = [_worker(a) for a in tasks]
    print(f"  done in {time.time() - t0:.1f}s")

    S = np.concatenate([r[0] for r in results])
    T = np.concatenate([r[1] for r in results])
    M = np.concatenate([r[2] for r in results])
    Rg = np.concatenate([r[3] for r in results])
    bnd = np.concatenate([r[4] for r in results])
    best = max(results, key=lambda r: r[6])
    best_coords = best[5]

    out = "new_results_cluster_quick.npz" if quick else "new_results_cluster.npz"
    np.savez(out, S=S, T=T, M=M, Rg=Rg, boundary=bnd,
             best_coords=best_coords if best_coords is not None else np.zeros((0, 3)),
             L=L, lambda_c=LAMBDA_C, gamma_loss=p["gamma_loss"])
    print(f"saved {out}  ({S.size} avalanches, "
          f"{int((~bnd).sum())} interior, {int(bnd.sum())} boundary-touching)")

    sel = (~bnd) & (Rg >= KEEP_RG_MIN)
    if sel.sum() > 50:
        Df = np.polyfit(np.log(Rg[sel]), np.log(S[sel]), 1)[0]
        df = np.polyfit(np.log(Rg[sel]), np.log(M[sel]), 1)[0]
        print(f"  interior fractal fit ({int(sel.sum())} clusters, "
              f"R_g in [{Rg[sel].min():.1f},{Rg[sel].max():.1f}]):")
        print(f"    S ~ R_g^Df : D_f = {Df:.2f}  (cf. FSS size exponent D = 3.69)")
        print(f"    M ~ R_g^df : d_f = {df:.2f}")


if __name__ == "__main__":
    main()
