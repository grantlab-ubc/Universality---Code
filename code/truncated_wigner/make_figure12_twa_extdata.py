"""Generator for the extended truncated-Wigner figure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nature_style as ns
ns.apply()

sys.path.insert(0, os.path.dirname(HERE))
import figstyle as fs
fs.apply()
ns.panel_label = fs.panel_label

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import twa_dp_revised as tw

C = ns._C
BLUE, RED, GREY, GREEN, TEAL = C["blue"], C["red"], C["grey"], C["green"], C["teal"]
ORANGE = C["orange"]
OLD, OUT, CACHE = tw.OLD, tw.OUT, tw.CACHE
MAIN_GFX = Path(HERE).parent / "DP_universality_Nature_revised" / "dp_loss_window_results"
DRAFT_GFX = Path(HERE).parent / "draft_figures"

d = np.load(CACHE)
k1 = float(d["k1"]); N = float(d["N_sites"])
kappas = d["kappas"]; q_mean, q_var = d["q_mean"], d["q_var"]
c_mean = d["c_mean"]
A0 = float(q_mean[0][0])
rho_c = np.array([tw._sat(a) for a in c_mean]) / N
kc_c = tw._kc_balance(kappas, rho_c * N, A0)
xk = kappas / k1
rho_q = np.array([tw._sat(a) for a in q_mean]) / N
A_q = rho_q * N
varA_q = np.array([tw._sat(v) for v in q_var])
q_ok = A_q >= 1.0
fano_q_x, fano_q = xk[q_ok], varA_q[q_ok] / A_q[q_ok]


fig, axes4 = plt.subplots(2, 2, figsize=fs.size(ns.COL_WIDTH_DOUBLE, 4.9))
axes = axes4.flat

ax = axes4[0, 0]
ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
box = FancyBboxPatch((0.10, 0.26), 0.56, 0.68,
                     boxstyle="round,pad=0.015,rounding_size=0.03",
                     fc="#F0F4F9", ec="#C3CEDA", lw=0.6, zorder=0)
ax.add_patch(box)

def _level(y, x0, x1, color, lw=1.8):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="round",
            zorder=3)

_level(0.76, 0.16, 0.52, C["red"])
ax.text(0.34, 0.80, r"$|P\rangle\ \mathrm{NO}^{+}\!+e^{-}$", color=C["red"],
        ha="center", va="bottom", fontsize=fs.a(5.2), zorder=3)
_level(0.40, 0.16, 0.52, BLUE)
ax.text(0.34, 0.36, r"$|R\rangle$ Rydberg $\mathrm{NO}^{*}$", color=BLUE,
        ha="center", va="top", fontsize=fs.a(5.2), zorder=3)
ax.annotate("", xy=(0.24, 0.745), xytext=(0.24, 0.415),
            arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.0,
                            mutation_scale=8), zorder=3)
ax.text(0.265, 0.58, r"$\Omega\sqrt{m}$", ha="left", va="center",
        fontsize=fs.a(5.4), zorder=3)
ax.annotate("", xy=(0.46, 0.415), xytext=(0.46, 0.745),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9,
                            mutation_scale=7), zorder=3)
ax.text(0.485, 0.58, r"$\kappa$", ha="left", va="center", color=BLUE,
        fontsize=fs.a(5.4), zorder=3)
_xs = np.linspace(0.545, 0.65, 80)
ax.plot(_xs, 0.76 + 0.045 * (_xs - 0.545) / 0.105
        + 0.013 * np.sin(2 * np.pi * (_xs - 0.545) / 0.021),
        color=C["purple"], lw=0.9, zorder=3)
ax.text(0.60, 0.845, r"$\gamma_\phi$", color=C["purple"], fontsize=fs.a(5.4),
        ha="center", va="bottom", zorder=3)
ax.plot([0.045], [0.52], "o", color=ORANGE, ms=4.2,
        markeredgecolor="#B26B3F", markeredgewidth=0.5, zorder=4)
ax.annotate("", xy=(0.215, 0.56), xytext=(0.07, 0.525),
            arrowprops=dict(arrowstyle="-|>", color="#B26B3F", lw=0.8,
                            ls="--", mutation_scale=6), zorder=4)
ax.text(0.015, 0.565, "$e^-$ of active\nneighbour", fontsize=fs.a(4.4),
        color="#8A5A34", ha="left", va="bottom", linespacing=1.25, zorder=4)
ax.annotate("", xy=(0.69, 0.285), xytext=(0.50, 0.74),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=0.9,
                            mutation_scale=7), zorder=3)
ax.text(0.635, 0.545, r"$\gamma_{\rm loss}$", color=GREEN, fontsize=fs.a(5.4),
        ha="left", va="center", zorder=3)
_level(0.24, 0.66, 0.96, GREY, lw=1.6)
ax.text(0.81, 0.205, r"$|V\rangle$ $\mathrm{N}+\mathrm{O}$ (inert)",
        color=GREY, ha="center", va="top", fontsize=fs.a(5.0))
ax.text(0.02, 0.10, r"$m=0\ \Rightarrow\ \Omega=0$: exact absorbing state",
        fontsize=fs.a(4.8), color=GREY, ha="left", va="center")
ax.text(0.02, 0.02,
        r"$\gamma_\phi\gg\Omega$: rate $k=C\,\Omega^{2}/\gamma_\phi$"
        " per electron",
        fontsize=fs.a(4.8), color="black", ha="left", va="center")
ax.set_title("Open three-state quantum site (NO)")
ns.panel_label(ax, "a")

ax = axes4[0, 1]
t, a = d["cmp_t_rate"], d["cmp_a_rate"]
ax.plot(t, a / a[0], "--", color="black", lw=1.4, zorder=2,
        label="classical rate kinetics")
t, a = d["cmp_t_strong"], d["cmp_a_strong"]
ax.plot(t, a / a[0], "o", color=BLUE, ms=3.0, mfc="none", markevery=2, zorder=3,
        label=fr"quantum, strong deph. ($\gamma_\phi={float(d['gamma']):g}$)")
t, a = d["cmp_t_weak"], d["cmp_a_weak"]
ax.plot(t, a / a[0], ":", color=GREY, lw=1.2, zorder=2,
        label=fr"quantum, weak deph. ($\gamma_\phi={float(d['gamma_weak']):g}$)")
ax.set_xlabel(r"time $t$ (units of $\Omega^{-1}$)")
ax.set_ylabel(r"active count $A(t)/A(0)$")
ax.set_title("Same stationary state at strong dephasing")
ax.set_ylim(-1.2, 47.0)
ax.set_yticks([0, 10, 20, 30])
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=fs.a(5),
          labelspacing=0.25, handlelength=1.2, handletextpad=0.4, borderpad=0.2)
ns.panel_label(ax, "b")

ax = axes4[1, 0]
nz = np.load(OUT / "twa_noise_increments.npz")
a_bin, v_bin, c_bin = nz["a"], nz["v"], nz["c"]
slope_n = float(np.sum(v_bin * a_bin * c_bin) / np.sum(a_bin ** 2 * c_bin))
ax.plot(a_bin, v_bin, "o", color=TEAL, ms=3.6, markeredgewidth=0.4,
        markeredgecolor="white", ls="none",
        label=r"jump CP: $\mathrm{Var}(\Delta A|A)/\Delta t$")
xx = np.linspace(0, float(np.max(a_bin)) * 1.05, 50)
ax.plot(xx, slope_n * xx, "--", color="black", lw=1.1,
        label=r"multiplicative $\propto A$")
ax.axhline(slope_n * 0.5 * float(np.max(a_bin)), color=GREY, ls="--", lw=0.9,
           label="additive noise")
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.annotate("noise vanishes at $A=0$\n(absorbing state)", xy=(4, 4),
            xytext=(0.24, 0.04), textcoords="axes fraction",
            fontsize=fs.a(5.5), ha="left", va="bottom", linespacing=1.3,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                            mutation_scale=7))
ax.set_xlabel(r"active count $A$")
ax.set_ylabel(r"noise power $\mathrm{Var}(\Delta A|A)/\Delta t$")
ax.set_title("Multiplicative demographic noise")
ax.legend(loc="upper left", fontsize=fs.a(5), handlelength=1.2, borderpad=0.2,
          labelspacing=0.3, handletextpad=0.4)
ns.panel_label(ax, "c")

ax = axes4[1, 1]
rngb = np.random.default_rng(4242)
FR = OUT / "fano_run.npz"
fr = np.load(FR) if FR.exists() else None
if fr is not None:
    fx, fv, fe = [], [], []
    for x in fr["kappas"]:
        A = fr[f"cp_{x:g}"].astype(np.float64)
        if np.mean(A) < 1.0:
            continue
        fano = float(np.mean(np.var(A, axis=1) /
                             np.maximum(np.mean(A, axis=1), 1e-9)))
        boots = []
        for _ in range(300):
            idx = rngb.integers(0, A.shape[1], A.shape[1])
            Ab = A[:, idx]
            boots.append(np.mean(np.var(Ab, axis=1) /
                                 np.maximum(np.mean(Ab, axis=1), 1e-9)))
        fx.append(float(x)); fv.append(fano); fe.append(float(np.std(boots)))
    ax.errorbar(fx, fv, yerr=fe, fmt="D", color=TEAL, ms=3.4,
                markeredgewidth=0.4, markeredgecolor="white", lw=0.8,
                capsize=1.6, ls="none", zorder=3)
fano_q_err = fano_q * np.sqrt(2.0 / 239.0)
ax.errorbar(fano_q_x, fano_q, yerr=fano_q_err, fmt="o", color=BLUE, ms=3.4,
            markeredgewidth=0.4, markeredgecolor="white", lw=0.8,
            capsize=1.6, ls="none", zorder=3)
ax.set_yscale("log")
ax.axhline(1.0, color=GREY, ls="--", lw=0.9)
_bt2 = plt.matplotlib.transforms.blended_transform_factory(ax.transData,
                                                           ax.transAxes)
ax.axvline(6.0, ymax=0.56, color="black", ls=":", lw=0.8)
ax.text(6.0, 0.03, r" $6k_1$ (MF)", color="black", fontsize=fs.a(5.5),
        ha="left", va="bottom", transform=_bt2)
if np.isfinite(kc_c):
    ax.axvline(kc_c / k1, color=GREY, ls=":", lw=0.8)
    ax.text(kc_c / k1, 0.03, fr"${kc_c/k1:.1f}k_1$ (DP) ", color=GREY,
            fontsize=fs.a(5.5), ha="right", va="bottom", transform=_bt2)
ax.text(0.03, 0.97, "discrete events\n(jump CP)", transform=ax.transAxes,
        color=TEAL, fontsize=fs.a(5.5), ha="left", va="top", linespacing=1.25)
ax.text(0.40, 0.415, "Poissonian", transform=ax.transAxes, color=GREY,
        fontsize=fs.a(5.5), ha="right", va="bottom")
ax.text(0.775, 0.10, "smooth quantum ensemble (DTWA)",
        transform=ax.transAxes, color=BLUE, fontsize=fs.a(5.5), ha="right",
        va="bottom")
ax.set_ylim(0.03, 40.0)
ax.set_xlim(0.7, 7.3)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax.set_xlabel(r"recombination $\kappa/k_1$")
ax.set_ylabel(r"Fano factor $\mathrm{Var}(A)/\langle A\rangle$")
ax.set_title("Critical bunching of the fluctuations")
if fr is not None and "q_hist" in fr:
    xh = float(fr["q_hist_kappa"])
    Ac = fr[f"cp_{xh:g}"].astype(np.float64).ravel()
    Aq = np.asarray(fr["q_hist"]).ravel()
    axi = ax.inset_axes([0.615, 0.60, 0.37, 0.33])
    bins = np.linspace(0.0, 3.0, 31)
    hc, _ = np.histogram(Ac / Ac.mean(), bins=bins, density=True)
    hq, _ = np.histogram(Aq / Aq.mean(), bins=bins, density=True)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    axi.fill_between(ctr, hc / hc.max(), step="mid", color=TEAL, alpha=0.55,
                     lw=0)
    axi.fill_between(ctr, hq / hq.max(), step="mid", color=BLUE, alpha=0.75,
                     lw=0)
    axi.text(1.35, 0.30, "discrete", color=TEAL, fontsize=fs.a(4.8),
             ha="left", va="bottom")
    axi.text(1.18, 0.82, "quantum", color=BLUE, fontsize=fs.a(4.8),
             ha="left", va="bottom")
    axi.set_xlim(0, 3)
    axi.set_ylim(0, 1.15)
    axi.set_yticks([])
    axi.set_xticks([0, 1, 2, 3])
    axi.set_xlabel(r"$A/\langle A\rangle$", fontsize=fs.a(5), labelpad=1)
    axi.set_title(fr"all runs at $\kappa={xh:g}\,k_1$", fontsize=fs.a(5), pad=1.5)
    axi.set_facecolor("white")
    axi.tick_params(labelsize=fs.a(4.5), width=0.4, length=1.5, pad=1)
    for s in axi.spines.values():
        s.set_linewidth(0.4)
ns.panel_label(ax, "d")

fig.tight_layout(pad=0.6, w_pad=1.6)
for base in (Path(HERE), MAIN_GFX, DRAFT_GFX):
    base.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = base / f"supp_fig_twa_reduction.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
