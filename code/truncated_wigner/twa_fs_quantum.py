"""Denser quantum finite-size scan for panel (f) of the lattice-TWA figure.
"""
from __future__ import annotations

import time
import numpy as np

import twa_dp_revised as M
from twa_dp_revised import lattice_dtwa_cond, lattice_rate_equations, _sat

OUT = M.OUT


def main():
    k1 = float(np.load(M.CACHE)["k1"])
    kap = 1.5 * k1
    Ls = [8, 10, 12, 14, 16, 20, 24, 28]
    ntraj, tmax, dtq = 64, 40.0, 0.01
    rho_q, sem_q, rho_r = [], [], []
    for L in Ls:
        M.L = L; M.N_SITES = L ** 3
        t0 = time.time()
        _, a, v = lattice_dtwa_cond(ntraj, M.GAMMA, kap, M.OMEGA, tmax, dtq, 2.0, "dense", 7700 + L)
        rq = _sat(a) / L ** 3
        sq = float(np.sqrt(_sat(v) / ntraj) / L ** 3)
        _, ar = lattice_rate_equations(k1, kap, tmax, 0.02, 2.0, "dense")
        rr = _sat(ar) / L ** 3
        rho_q.append(rq); sem_q.append(sq); rho_r.append(rr)
        print(f"L={L:2d}  rho_Q={rq:.4f} +/- {sq:.4f}   rho_rate={rr:.4f}   "
              f"({time.time()-t0:.0f} s)", flush=True)
    np.savez(OUT / "finite_size_q.npz", L=np.array(Ls), rho_q=np.array(rho_q),
             sem_q=np.array(sem_q), rho_r=np.array(rho_r), kappa=kap, k1=k1)
    print("[cache] wrote", OUT / "finite_size_q.npz", flush=True)


if __name__ == "__main__":
    main()
