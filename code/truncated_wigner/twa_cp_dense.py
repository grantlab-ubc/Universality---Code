"""Dense sampling of the jump-contact-process recombination sweep across its
absorbing transition.
"""
from __future__ import annotations

import numpy as np

import twa_dp_revised as tw

OUT = tw.OUT
CACHE_DENSE = OUT / "cp_dense.npz"

N_REAL = 400
T_MAX = 120.0
DT = 0.02
SAMPLE_DT = 2.0


def main() -> None:
    d = np.load(tw.CACHE)
    k1 = float(d["k1"])
    N = float(d["N_sites"])
    k_of_m = np.arange(0, 7, dtype=float) * k1

    dense = np.array([3.2, 3.4, 3.55, 3.7, 3.85, 4.25, 4.5, 4.75])
    kappas = dense * k1
    rho, sem = [], []
    for i, kap in enumerate(kappas):
        t, a, v = tw.classical_cp_var(N_REAL, k_of_m, kap, T_MAX, DT,
                                      SAMPLE_DT, "cube", 8200 + i)
        r = tw._sat(a) / N
        s = np.sqrt(tw._sat(v) / N_REAL) / N
        rho.append(r)
        sem.append(s)
        print(f"    kappa={kap:.4f} ({dense[i]:.2f} k1)  rho_CP={r:.5f} "
              f"+- {s:.5f}", flush=True)

    kr = np.linspace(1.0, 7.0, 49) * k1
    rho_r = []
    for kap in kr:
        _, ar = tw.lattice_rate_equations(k1, kap, T_MAX, DT, SAMPLE_DT, "cube")
        rho_r.append(tw._sat(ar) / N)
    rho_r = np.asarray(rho_r)

    np.savez(CACHE_DENSE,
             kappas_cp=kappas, rho_cp=np.asarray(rho), sem_cp=np.asarray(sem),
             kappas_rate=kr, rho_rate=rho_r, k1=k1, n_real=N_REAL)
    print(f"[cache] wrote {CACHE_DENSE}")


if __name__ == "__main__":
    main()
