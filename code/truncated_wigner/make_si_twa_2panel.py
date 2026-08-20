"""Generator for the supporting truncated-Wigner panels.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nature_style as ns
ns.apply()

sys.path.insert(0, os.path.dirname(HERE))
import figstyle as fs
fs.apply()

import matplotlib.pyplot as plt
import twa_dp_revised as tw

OUT, CACHE = tw.OUT, tw.CACHE
SI_DIR = os.path.join(os.path.dirname(HERE), "SI_figures")
os.makedirs(SI_DIR, exist_ok=True)

BLUE = ns._C["blue"]

d = np.load(CACHE)
k1 = float(d["k1"])
N = float(d["N_sites"])
kappas = d["kappas"]
q_mean, q_var = d["q_mean"], d["q_var"]
r_mean = d["r_mean"]
xk = kappas / k1
rho_q = np.array([tw._sat(a) for a in q_mean]) / N
rho_r = np.array([tw._sat(a) for a in r_mean]) / N
sem_q = np.sqrt(np.array([tw._sat(v) for v in q_var]) / 240.0) / N

fig, axes = plt.subplots(1, 2, figsize=fs.size(5.6, 2.6))

ax = axes[0]
order = np.argsort(xk)
ax.plot(xk[order], rho_r[order], "--", color="black", lw=1.3, zorder=2,
        label=r"rate kinetics $k(m)=m\,k_1$")
ax.errorbar(xk, rho_q, yerr=sem_q, fmt="o", color=BLUE, lw=0, elinewidth=0.8,
            capsize=1.6, zorder=3, label="quantum lattice (DTWA)",
            **ns.DATA_MARKER)
ax.set_xlabel(r"recombination $\kappa/k_1$")
ax.set_ylabel(r"stationary active density $\rho_\mathrm{stat}$")
ax.set_title(r"Quantum $=$ rate kinetics")
ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.legend(loc="upper right", fontsize=fs.a(6))
fs.panel_label(ax, "a")

ax = axes[1]
fsz = np.load(OUT / "finite_size_q.npz")
ax.axhline(float(np.mean(fsz["rho_r"])), ls="--", color="black", lw=1.3,
           zorder=2, label="classical rate kinetics")
ax.errorbar(fsz["L"], fsz["rho_q"], yerr=fsz["sem_q"], fmt="o", color=BLUE,
            lw=0, elinewidth=0.8, capsize=1.6, zorder=3,
            label="quantum lattice (DTWA)", **ns.DATA_MARKER)
ax.set_xlabel(fr"lattice size $L$  (at $\kappa={fsz['kappa']/k1:.1f}\,k_1$)")
ax.set_ylabel(r"stationary active density $\rho_\mathrm{stat}$")
ax.set_title("Finite-size convergence")
ax.set_xticks(list(fsz["L"]))
ax.set_ylim(0.3555, 0.4005)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
ax.legend(loc="upper right", fontsize=fs.a(6))
fs.panel_label(ax, "b")

fig.tight_layout(pad=0.5, w_pad=1.8)
for ext in ("pdf", "png"):
    p = os.path.join(SI_DIR, f"figure_11_twa_si_2panel.{ext}")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", p)
