"""Dephasing crossover of the quantum lattice: where the classical reduction
holds and where it fails.
"""
from __future__ import annotations

import numpy as np

import twa_dp_revised as tw

OUT = tw.OUT
CACHE = OUT / "crossover_scan.npz"

GAMMAS = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0]
N_TRAJ = 200
K1_16 = 0.0594


def main() -> None:
    out = {}
    eps_sat, eps_tr, rho_q_sat, rho_r_sat, k1_list = [], [], [], [], []
    for i, g in enumerate(GAMMAS):
        k1 = K1_16 * 16.0 / g
        kappa = 1.5 * k1
        t_max = min(240.0, 4.0 / k1)
        dt_q = 0.005 if g >= 20.0 else 0.01
        sample = t_max / 60.0
        tq, aq, _ = tw.lattice_dtwa_cond(N_TRAJ, g, kappa, 1.0,
                                         t_max, dt_q, sample, "cube", 9500 + i)
        tr, ar = tw.lattice_rate_equations(k1, kappa, t_max, 0.01, sample, "cube")
        a_rate_sat = tw._sat(ar)
        a_q_sat = tw._sat(aq)
        n = min(len(aq), len(ar))
        dev = np.max(np.abs(aq[:n] - ar[:n])) / a_rate_sat
        es = abs(a_q_sat - a_rate_sat) / a_rate_sat
        eps_sat.append(es); eps_tr.append(float(dev))
        rho_q_sat.append(a_q_sat / tw.N_SITES)
        rho_r_sat.append(a_rate_sat / tw.N_SITES)
        k1_list.append(k1)
        out[f"t_{g:g}"] = tq * k1
        out[f"aq_{g:g}"] = aq
        out[f"ar_{g:g}"] = ar
        print(f"gamma_phi={g:5.1f}  k1={k1:.4f}  rho_Q={a_q_sat/tw.N_SITES:.4f}  "
              f"rho_rate={a_rate_sat/tw.N_SITES:.4f}  eps_sat={es:.3f}  "
              f"eps_transient={dev:.3f}", flush=True)

    np.savez(CACHE, gammas=np.array(GAMMAS), k1=np.array(k1_list),
             eps_sat=np.array(eps_sat), eps_tr=np.array(eps_tr),
             rho_q_sat=np.array(rho_q_sat), rho_r_sat=np.array(rho_r_sat),
             n_traj=N_TRAJ, **out)
    print(f"[cache] wrote {CACHE}")


if __name__ == "__main__":
    main()
