"""Statistics measurement of the fluctuation observable of the jump contact process.
"""
from __future__ import annotations

import math
import time

import numpy as np

import twa_dp_revised as tw
from twa_lattice_dp_support import nn_field, make_seed

OUT = tw.OUT
CACHE = OUT / "fano_run.npz"

L = tw.L
N_SITES = tw.N_SITES
N_REAL = 1200
N_TRAJ_Q = 240
T_MAX = 120.0
DT_C = 0.02
DT_Q = 0.01
SAMPLE_DT = 2.0
KAPPA_GRID = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.4, 3.55, 3.7,
                       3.85, 4.1, 4.3])
KAPPA_HIST = 3.55


def cp_samples(n_real, k_of_m, kappa, t_max, dt, sample_dt, seed):
    """Jump CP; returns per-realisation A at every sample time (n_t, n_real)."""
    rng = np.random.default_rng(seed)
    shape = (n_real, L, L, L)
    active = (make_seed(shape, L, "cube") > 0).astype(np.float64)
    p_act_t = 1.0 - np.exp(-k_of_m * dt)
    p_dec_t = 1.0 - np.exp(-(kappa + k_of_m) * dt)
    n_steps = int(round(t_max / dt))
    every = max(1, int(round(sample_dt / dt)))
    times, samples = [], []
    for step in range(n_steps + 1):
        if step % every == 0 or step == n_steps:
            times.append(step * dt)
            samples.append(np.sum(active, axis=(-3, -2, -1)).astype(np.int16))
        if step == n_steps:
            break
        m = np.clip(nn_field(active).astype(np.int64), 0, 6)
        r = rng.random(shape)
        births = (active == 0.0) & (r < p_act_t[m])
        deaths = (active == 1.0) & (r < p_dec_t[m])
        active = active + births.astype(np.float64) - deaths.astype(np.float64)
    return np.asarray(times), np.asarray(samples)


def dtwa_samples(n_traj, gamma_phi, kappa, omega, t_max, dt, sample_dt, seed):
    """lattice_dtwa_cond with per-trajectory A samples returned."""
    rng = np.random.default_rng(seed)
    shape = (n_traj, L, L, L)
    sz = make_seed(shape, L, "cube")
    sx = rng.choice(np.array([-1.0, 1.0]), size=shape)
    sy = rng.choice(np.array([-1.0, 1.0]), size=shape)
    n_steps = int(round(t_max / dt))
    every = max(1, int(round(sample_dt / dt)))
    half = 0.5 * dt
    sqrt_g = math.sqrt(gamma_phi)
    damp_z = math.exp(-kappa * dt)
    damp_t = math.exp(-0.5 * kappa * dt)

    def half_rotation(sx, sy, sz):
        n = 0.5 * (1.0 + sz)
        m = np.clip(nn_field(n), 0.0, None)
        theta = 2.0 * omega * np.sqrt(m) * half
        c, s = np.cos(theta), np.sin(theta)
        return sx, sy * c - sz * s, sy * s + sz * c

    times, samples = [], []
    for step in range(n_steps + 1):
        if step % every == 0 or step == n_steps:
            times.append(step * dt)
            samples.append(np.sum(0.5 * (1.0 + sz), axis=(-3, -2, -1)))
        if step == n_steps:
            break
        sx, sy, sz = half_rotation(sx, sy, sz)
        theta = -2.0 * sqrt_g * (math.sqrt(dt) * rng.standard_normal(shape))
        ct, st = np.cos(theta), np.sin(theta)
        sx, sy = sx * ct - sy * st, sx * st + sy * ct
        sz = -1.0 + (sz + 1.0) * damp_z
        sx *= damp_t
        sy *= damp_t
        sx, sy, sz = half_rotation(sx, sy, sz)
    return np.asarray(times), np.asarray(samples)


def main() -> None:
    d = np.load(tw.CACHE)
    k1 = float(d["k1"])
    k_of_m = np.arange(0, 7, dtype=float) * k1

    out = {"kappas": KAPPA_GRID, "k1": k1, "n_real": N_REAL,
           "n_traj_q": N_TRAJ_Q, "t_max": T_MAX, "sample_dt": SAMPLE_DT}
    for i, x in enumerate(KAPPA_GRID):
        t0 = time.time()
        times, s = cp_samples(N_REAL, k_of_m, x * k1, T_MAX, DT_C,
                              SAMPLE_DT, 8600 + i)
        late = times > 0.5 * T_MAX
        out[f"cp_{x:g}"] = s[late]
        A_late = s[late].astype(np.float64)
        fano = float(np.mean(np.var(A_late, axis=1) / np.maximum(
            np.mean(A_late, axis=1), 1e-9)))
        print(f"kappa={x:4.2f} k1  <A>={A_late.mean():8.2f}  Fano={fano:7.2f}"
              f"   [{time.time()-t0:5.1f}s]", flush=True)

    t0 = time.time()
    times, sq = dtwa_samples(N_TRAJ_Q, tw.GAMMA, KAPPA_HIST * k1, tw.OMEGA,
                             T_MAX, DT_Q, SAMPLE_DT, 9700)
    late = times > 0.5 * T_MAX
    out["q_hist_kappa"] = KAPPA_HIST
    out["q_hist"] = sq[late]
    A_late = sq[late]
    print(f"quantum kappa={KAPPA_HIST} k1  <A>={A_late.mean():8.2f}  "
          f"Fano={np.mean(np.var(A_late,axis=1)/np.mean(A_late,axis=1)):7.3f}"
          f"   [{time.time()-t0:5.1f}s]", flush=True)

    np.savez_compressed(CACHE, **out)
    print(f"[cache] wrote {CACHE}", flush=True)


if __name__ == "__main__":
    main()
