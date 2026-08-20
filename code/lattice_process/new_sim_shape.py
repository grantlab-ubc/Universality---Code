"""Avalanche temporal-profile (shape-collapse) simulation at the production point.
Records the instantaneous active count n(t) for every single-seed avalanche. Then
builds the mean profile <s(t|T)> in duration bands and tests the universal-shape
collapse  <s(t|T)> = T^{gamma_ST-1} F(t/T). """
import os, sys, time, math
import numpy as np
import concurrent.futures
from new_sims_core import PARAMS, build_shifts, update_grid

N_U = 60
U_GRID = (np.arange(N_U) + 0.5) / N_U

def _worker(args):
    (n_runs, seed, L, max_steps, dt, p, shifts, band_edges) = args
    rng = np.random.default_rng(seed)
    c = L // 2
    nb = len(band_edges) - 1
    band_sum = [np.zeros(N_U) for _ in range(nb)]
    band_cnt = np.zeros(nb, dtype=np.int64)
    Tlist = []; Slist = []
    for _ in range(n_runs):
        grid = np.zeros((L, L, L), dtype=np.int8)
        grid[c, c, c] = 1
        act = []
        S = 1
        for step in range(max_steps):
            n_act = int(np.count_nonzero(grid == 1))
            if n_act == 0:
                break
            act.append(n_act)
            grid, spread_mask, _ = update_grid(grid, shifts, dt, p['lambda_fac'],
                p['alpha'], p['gamma_base'], p['beta'], p['gamma_loss'], rng)
            S += int(np.count_nonzero(spread_mask))
        D = len(act)
        if D == 0:
            continue
        T = D * dt
        Tlist.append(T); Slist.append(S)
        b = int(np.searchsorted(band_edges, T, side='right') - 1)
        if 0 <= b < nb and D >= 2:
            a = np.asarray(act, float)
            u_src = (np.arange(D) + 0.5) / D
            band_sum[b] += np.interp(U_GRID, u_src, a)
            band_cnt[b] += 1
    return band_sum, band_cnt, np.array(Tlist), np.array(Slist)

def run(n_runs, band_edges, max_T=250.0, base_seed=20260606, workers=9):
    p = PARAMS; L = p['L']; dt = p['dt']
    max_steps = int(max_T / dt) + 2
    shifts = build_shifts(p['r_cut'], p['d_0'])
    per = [n_runs // workers] * workers
    per[0] += n_runs % workers
    ss = np.random.SeedSequence(base_seed)
    seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(workers)]
    args = [(per[i], seeds[i], L, max_steps, dt, p, shifts, band_edges)
            for i in range(workers) if per[i] > 0]
    nb = len(band_edges) - 1
    tot_sum = [np.zeros(N_U) for _ in range(nb)]
    tot_cnt = np.zeros(nb, dtype=np.int64)
    Ts = []; Ss = []
    t0 = time.time()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_worker, args))
    except (PermissionError, OSError, RuntimeError) as e:
        print("serial fallback:", e); results = [_worker(a) for a in args]
    for bs, bc, Tl, Sl in results:
        for b in range(nb):
            tot_sum[b] += bs[b]
        tot_cnt += bc; Ts.append(Tl); Ss.append(Sl)
    Ts = np.concatenate(Ts); Ss = np.concatenate(Ss)
    print(f"  ran {len(Ts)} avalanches in {time.time()-t0:.1f}s")
    profiles = [tot_sum[b] / max(1, tot_cnt[b]) for b in range(nb)]
    return profiles, tot_cnt, Ts, Ss

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    n = 1500 if quick else int(sys.argv[sys.argv.index("-n")+1]) if "-n" in sys.argv else 60000
    band_edges = np.array([3, 6, 12, 24, 48, 96, 192], float)
    profiles, cnt, Ts, Ss = run(n, band_edges)
    print("band counts:", dict(zip([f"{band_edges[i]:.0f}-{band_edges[i+1]:.0f}" for i in range(len(cnt))], cnt.tolist())))
    mask = Ss > 2
    from numpy.polynomial import polynomial as P
    tb = np.logspace(np.log10(max(Ts[mask].min(),0.5)), np.log10(Ts[mask].max()), 16)
    xc, yc = [], []
    for lo, hi in zip(tb[:-1], tb[1:]):
        m = (Ts >= lo) & (Ts < hi) & mask
        if m.sum() >= 20:
            xc.append(math.sqrt(lo*hi)); yc.append(Ss[m].mean())
    xc, yc = np.log(xc), np.log(yc)
    sl = np.polyfit(xc, yc, 1)[0]
    print(f"  validation gamma_ST (<S> vs T) = {sl:.3f}  (production 1.812)")
    np.savez('new_results_shape.npz', U_GRID=U_GRID, profiles=np.array(profiles),
             counts=cnt, band_edges=band_edges, Ts=Ts, Ss=Ss)
    print("saved new_results_shape.npz")
