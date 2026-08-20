"""Extension of the high-statistics contact-process sweep
"""
from __future__ import annotations

import time

import numpy as np

import twa_dp_revised as tw
from twa_fano_run import cp_samples, T_MAX, DT_C, SAMPLE_DT, N_REAL

OUT = tw.OUT
CACHE = OUT / "fano_ext.npz"
KAPPA_EXT = np.array([4.6, 5.0, 6.0, 7.0])


def main() -> None:
    d = np.load(tw.CACHE)
    k1 = float(d["k1"])
    k_of_m = np.arange(0, 7, dtype=float) * k1
    out = {"kappas": KAPPA_EXT, "k1": k1, "n_real": N_REAL}
    for i, x in enumerate(KAPPA_EXT):
        t0 = time.time()
        times, s = cp_samples(N_REAL, k_of_m, x * k1, T_MAX, DT_C,
                              SAMPLE_DT, 8800 + i)
        late = times > 0.5 * T_MAX
        out[f"cp_{x:g}"] = s[late]
        dead = float(np.mean(s[late][-1] == 0))
        print(f"kappa={x:4.2f} k1  dead fraction={dead:.3f}"
              f"   [{time.time()-t0:5.1f}s]", flush=True)
    np.savez_compressed(CACHE, **out)
    print(f"[cache] wrote {CACHE}", flush=True)


if __name__ == "__main__":
    main()
