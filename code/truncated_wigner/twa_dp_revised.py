"""Quantum lattice dynamics of the facilitation and its reduction to the classical contact process.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from twa_facilitation_dp_support import evolve_single_site, extract_rate, _movavg
from twa_lattice_dp_support import nn_field, make_seed, write_csv

Array = np.ndarray

OLD = Path("twa_dp_support_results")
OUT = Path("twa_dp_support_results_revised")
CACHE = OUT / "revised_cache.npz"

OMEGA = 1.0
GAMMA = 16.0
GAMMA_WEAK = 2.0
L = 12
N_SITES = L ** 3


def lattice_dtwa_cond(
    n_traj: int, gamma_phi: float, kappa: float, omega: float,
    t_max: float, dt: float, sample_dt: float, seed_kind: str, seed: int,
):
    rng = np.random.default_rng(seed)
    shape = (n_traj, L, L, L)
    sz = make_seed(shape, L, seed_kind)
    sx = rng.choice(np.array([-1.0, 1.0]), size=shape)
    sy = rng.choice(np.array([-1.0, 1.0]), size=shape)

    n_steps = int(round(t_max / dt))
    every = max(1, int(round(sample_dt / dt)))
    half = 0.5 * dt
    sqrt_g = math.sqrt(gamma_phi) if gamma_phi > 0 else 0.0
    damp_z = math.exp(-kappa * dt)
    damp_t = math.exp(-0.5 * kappa * dt)

    def half_rotation(sx, sy, sz):
        n = 0.5 * (1.0 + sz)
        m = np.clip(nn_field(n), 0.0, None)
        theta = 2.0 * omega * np.sqrt(m) * half
        c, s = np.cos(theta), np.sin(theta)
        return sx, sy * c - sz * s, sy * s + sz * c

    times: List[float] = []
    counts: List[float] = []
    varis: List[float] = []
    for step in range(n_steps + 1):
        if step % every == 0 or step == n_steps:
            A_traj = np.sum(0.5 * (1.0 + sz), axis=(-3, -2, -1))
            times.append(step * dt)
            counts.append(float(np.mean(A_traj)))
            varis.append(float(np.var(A_traj)))
        if step == n_steps:
            break
        sx, sy, sz = half_rotation(sx, sy, sz)
        if sqrt_g > 0:
            theta = -2.0 * sqrt_g * (math.sqrt(dt) * rng.standard_normal(shape))
            ct, st = np.cos(theta), np.sin(theta)
            sx, sy = sx * ct - sy * st, sx * st + sy * ct
        sz = -1.0 + (sz + 1.0) * damp_z
        sx *= damp_t
        sy *= damp_t
        sx, sy, sz = half_rotation(sx, sy, sz)

    return np.asarray(times), np.asarray(counts), np.asarray(varis)


def classical_cp_var(
    n_real: int, k_of_m: Array, kappa: float,
    t_max: float, dt: float, sample_dt: float, seed_kind: str, seed: int,
):
    rng = np.random.default_rng(seed)
    shape = (n_real, L, L, L)
    active = (make_seed(shape, L, seed_kind) > 0).astype(np.float64)

    n_steps = int(round(t_max / dt))
    every = max(1, int(round(sample_dt / dt)))
    times: List[float] = []
    counts: List[float] = []
    varis: List[float] = []
    for step in range(n_steps + 1):
        if step % every == 0 or step == n_steps:
            A = np.sum(active, axis=(-3, -2, -1))
            times.append(step * dt)
            counts.append(float(np.mean(A)))
            varis.append(float(np.var(A)))
        if step == n_steps:
            break
        m = nn_field(active).astype(np.int64)
        k = k_of_m[np.clip(m, 0, len(k_of_m) - 1)]
        p_act = 1.0 - np.exp(-k * dt)
        p_dec = 1.0 - np.exp(-(kappa + k) * dt)
        r = rng.random(shape)
        births = (active == 0.0) & (r < p_act)
        deaths = (active == 1.0) & (r < p_dec)
        active = active + births.astype(np.float64) - deaths.astype(np.float64)
    return np.asarray(times), np.asarray(counts), np.asarray(varis)


def _sat(a: Array, frac: float = 0.5) -> float:
    n = len(a)
    return float(np.mean(a[int(n * (1.0 - frac)):]))


def classical_cp_increment_stats(
    n_real: int, k_of_m: Array, kappa: float,
    t_max: float, dt: float, seed_kind: str, seed: int, n_bins: int = 14,
):
    rng = np.random.default_rng(seed)
    shape = (n_real, L, L, L)
    active = (make_seed(shape, L, seed_kind) > 0).astype(np.float64)
    n_steps = int(round(t_max / dt))
    edges = np.linspace(0.0, 0.45 * N_SITES, n_bins + 1)
    cnt = np.zeros(n_bins)
    s1 = np.zeros(n_bins)
    s2 = np.zeros(n_bins)
    for _ in range(n_steps):
        A0 = np.sum(active, axis=(-3, -2, -1))
        m = nn_field(active).astype(np.int64)
        k = k_of_m[np.clip(m, 0, len(k_of_m) - 1)]
        p_act = 1.0 - np.exp(-k * dt)
        p_dec = 1.0 - np.exp(-(kappa + k) * dt)
        r = rng.random(shape)
        births = (active == 0.0) & (r < p_act)
        deaths = (active == 1.0) & (r < p_dec)
        active = active + births.astype(np.float64) - deaths.astype(np.float64)
        dA = np.sum(active, axis=(-3, -2, -1)) - A0
        idx = np.clip(np.digitize(A0, edges) - 1, 0, n_bins - 1)
        np.add.at(cnt, idx, 1.0)
        np.add.at(s1, idx, dA)
        np.add.at(s2, idx, dA * dA)
    good = cnt > 200
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = s1 / np.maximum(cnt, 1.0)
    var = s2 / np.maximum(cnt, 1.0) - mean ** 2
    return centers[good], (var / dt)[good], cnt[good]


def lattice_rate_equations(
    k1: float, kappa: float, t_max: float, dt: float, sample_dt: float,
    seed_kind: str,
):
    n = 0.5 * (1.0 + make_seed((L, L, L), L, seed_kind))
    n_steps = int(round(t_max / dt))
    every = max(1, int(round(sample_dt / dt)))
    times: List[float] = []
    counts: List[float] = []
    for step in range(n_steps + 1):
        if step % every == 0 or step == n_steps:
            times.append(step * dt)
            counts.append(float(np.sum(n)))
        if step == n_steps:
            break
        m = nn_field(n)
        n = n + dt * (k1 * m * (1.0 - 2.0 * n) - kappa * n)
        np.clip(n, 0.0, 1.0, out=n)
    return np.asarray(times), np.asarray(counts)


def run_additivity(n_traj: int, dt: float = 0.002) -> Tuple[Array, Array, Array]:
    """Single-site DTWA rate for coupling sqrt(m)*Omega on the energy shell."""
    print(f"[rates] single-site k(m) for m independent channels, gamma_phi={GAMMA}")
    ms, ks, r2s = [0], [0.0], [1.0]
    print("    k(m=0) = 0 exactly (no electron, no coupling)")
    for m in range(1, 7):
        om = math.sqrt(m) * OMEGA
        k_guess = m * OMEGA ** 2 / GAMMA
        t_max = max(30.0, 1.5 / k_guess)
        sample = max(1, int(round((t_max / 400.0) / dt)))
        t, n_up, _ = evolve_single_site(om, GAMMA, 0.0, t_max, dt, n_traj,
                                        sample, 6100 + 13 * m)
        k, r2 = extract_rate(t, n_up)
        ms.append(m); ks.append(float(k)); r2s.append(float(r2))
        print(f"    k(m={m}) = {k:.4e}   k/(m*k1guess)={k / k_guess:.3f}   R2={r2:.4f}")
    return np.array(ms), np.array(ks), np.array(r2s)


def run_all(traj_q: int, real_c: int, t_max: float, dt_q: float, dt_c: float,
            sample_dt: float, n_traj_site: int) -> None:
    table_m, table_k, table_r2 = run_additivity(n_traj_site)
    k1 = table_k[1]
    k_of_m = np.arange(0, 7, dtype=float) * k1

    kappas = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]) * k1
    print(f"[sweep] k1={k1:.4f}  kappa/k1 = {np.round(kappas / k1, 2)}")

    q_t = None
    q_mean, q_var, c_mean, c_var, r_mean = [], [], [], [], []
    for i, kap in enumerate(kappas):
        tq, aq, vq = lattice_dtwa_cond(traj_q, GAMMA, kap, OMEGA,
                                       t_max, dt_q, sample_dt, "cube", 7100 + i)
        tc, ac, vc = classical_cp_var(real_c, k_of_m, kap,
                                      t_max, dt_c, sample_dt, "cube", 8100 + i)
        tr, ar = lattice_rate_equations(k1, kap, t_max, dt_c, sample_dt, "cube")
        q_t = tq
        q_mean.append(aq); q_var.append(vq)
        c_mean.append(ac); c_var.append(vc)
        r_mean.append(ar)
        print(f"    kappa={kap:.3f} ({kap/k1:4.2f} k1)  "
              f"Q rho_sat={_sat(aq)/N_SITES:.4f}  "
              f"RATE rho_sat={_sat(ar)/N_SITES:.4f}  "
              f"CP rho_sat={_sat(ac)/N_SITES:.4f}")

    kap_cmp = 1.5 * k1
    print(f"[compare] kappa={kap_cmp:.3f} (2 k1): strong/weak dephasing vs classical")
    t_s, a_s, _ = lattice_dtwa_cond(traj_q, GAMMA, kap_cmp, OMEGA,
                                    60.0, dt_q, 1.0, "cube", 9100)
    t_w, a_w, _ = lattice_dtwa_cond(traj_q, GAMMA_WEAK, kap_cmp, OMEGA,
                                    60.0, dt_q, 1.0, "cube", 9200)
    t_c, a_c, _ = classical_cp_var(real_c, k_of_m, kap_cmp,
                                   60.0, dt_c, 1.0, "cube", 9300)
    t_r, a_r = lattice_rate_equations(k1, kap_cmp, 60.0, dt_c, 1.0, "cube")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE,
             table_m=table_m, table_k=table_k, table_r2=table_r2, k1=k1,
             kappas=kappas, t=q_t,
             q_mean=np.array(q_mean), q_var=np.array(q_var),
             c_mean=np.array(c_mean), c_var=np.array(c_var),
             r_mean=np.array(r_mean),
             kap_cmp=kap_cmp,
             cmp_t_strong=t_s, cmp_a_strong=a_s,
             cmp_t_weak=t_w, cmp_a_weak=a_w,
             cmp_t_classical=t_c, cmp_a_classical=a_c,
             cmp_t_rate=t_r, cmp_a_rate=a_r,
             N_sites=N_SITES, gamma=GAMMA, gamma_weak=GAMMA_WEAK, omega=OMEGA)
    print(f"[cache] wrote {CACHE}")

    write_csv(OUT / "twa_rate_additivity.csv",
              [{"m": int(m), "k": float(k), "k_over_m_k1": float(k / (m * k1)) if m else 1.0,
                "r2": float(r)} for m, k, r in zip(table_m, table_k, table_r2)])
    rows = []
    for i, kap in enumerate(kappas):
        for t, a, v in zip(q_t, q_mean[i], q_var[i]):
            rows.append({"panel": "sweep", "model": "dtwa", "kappa": kap,
                         "t": t, "A": a, "varA": v})
        for t, a, v in zip(q_t, c_mean[i], c_var[i]):
            rows.append({"panel": "sweep", "model": "classical", "kappa": kap,
                         "t": t, "A": a, "varA": v})
    for tag, (tt, aa) in [("dtwa_strong", (t_s, a_s)), ("dtwa_weak", (t_w, a_w)),
                          ("classical", (t_c, a_c))]:
        for t, a in zip(tt, aa):
            rows.append({"panel": "compare", "model": tag, "kappa": kap_cmp,
                         "t": t, "A": a, "varA": ""})
    write_csv(OUT / "twa_lattice_timeseries_revised.csv", rows)


def _kc_balance(kappas: Array, A_sat: Array, A0: float) -> float:
    """Seed-balance proxy: kappa where the stationary count crosses the seed."""
    y = A_sat - A0
    for i in range(len(kappas) - 1):
        if y[i] >= 0 >= y[i + 1]:
            return float(kappas[i] + (kappas[i + 1] - kappas[i]) * y[i] / (y[i] - y[i + 1]))
    return float("nan")


def rate_error_bars(n_seeds: int = 4, n_traj: int = 8000,
                    dt: float = 0.002) -> Array:
    """Std of the single-site k(m) over independent DTWA seeds (m = 0..6).
    Cached; m=0 has no coupling, so its rate (and error) is identically 0."""
    path = OUT / "twa_rate_errors.npz"
    if path.exists():
        return np.load(path)["err"]
    print(f"[errors] k(m) spread over {n_seeds} independent seeds")
    errs = [0.0]
    for m in range(1, 7):
        om = math.sqrt(m) * OMEGA
        k_guess = m * OMEGA ** 2 / GAMMA
        t_max = max(30.0, 1.5 / k_guess)
        sample = max(1, int(round((t_max / 400.0) / dt)))
        ks = []
        for s in range(n_seeds):
            t, n_up, _ = evolve_single_site(om, GAMMA, 0.0, t_max, dt, n_traj,
                                            sample, 6100 + 13 * m + 1000 * s)
            k, _ = extract_rate(t, n_up)
            ks.append(float(k))
        errs.append(float(np.std(ks, ddof=1)))
        print(f"    m={m}: k = {np.mean(ks):.4e} +/- {errs[-1]:.1e}")
    err = np.array(errs)
    np.savez(path, err=err)
    print(f"[errors] wrote {path}")
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true",
                    help="rebuild figures from the cache, no simulation")
    ap.add_argument("--quick", action="store_true",
                    help="reduced budgets (smoke test)")
    ap.add_argument("--traj", type=int, default=240)
    ap.add_argument("--real", type=int, default=400)
    ap.add_argument("--tmax", type=float, default=120.0)
    ap.add_argument("--dt-q", type=float, default=0.01)
    ap.add_argument("--dt-c", type=float, default=0.02)
    ap.add_argument("--sample-dt", type=float, default=2.0)
    ap.add_argument("--traj-site", type=int, default=8000)
    args = ap.parse_args()

    if not args.replot:
        if args.quick:
            run_all(40, 60, 40.0, args.dt_q, args.dt_c, args.sample_dt, 2000)
        else:
            run_all(args.traj, args.real, args.tmax, args.dt_q, args.dt_c,
                    args.sample_dt, args.traj_site)

    stats = make_lattice_figure(OUT / "twa_lattice_dp_support_figure_revised.png")
    make_single_site_figure(OUT / "twa_dp_support_figure_revised.png")
    write_report(stats)


if __name__ == "__main__":
    main()
