"""Generator for the truncated-Wigner reduction figure.
"""
from __future__ import annotations

import csv
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import twa_dp_revised as tw

C = ns._C
BLUE, RED, GREY, GREEN, TEAL = C["blue"], C["red"], C["grey"], C["green"], C["teal"]
PURPLE, ORANGE, BROWN = C["purple"], C["orange"], C["brown"]
OLD, OUT, CACHE = tw.OLD, tw.OUT, tw.CACHE
MAIN_GFX = Path(HERE).parent / "DP_universality_Nature_revised" / "dp_loss_window_results"
DRAFT_GFX = Path(HERE).parent / "draft_figures"

d = np.load(CACHE)
k1 = float(d["k1"]); N = float(d["N_sites"])
table_m, table_k = d["table_m"], d["table_k"]
kappas = d["kappas"]; q_mean, q_var = d["q_mean"], d["q_var"]
c_mean, c_var = d["c_mean"], d["c_var"]; r_mean = d["r_mean"]
A0 = float(q_mean[0][0])
rho_q = np.array([tw._sat(a) for a in q_mean]) / N
rho_c = np.array([tw._sat(a) for a in c_mean]) / N
var_c = np.array([tw._sat(v) for v in c_var])
kc_c = tw._kc_balance(kappas, rho_c * N, A0)
slope_m = float(np.sum(table_k * table_m) / np.sum(table_m.astype(float) ** 2))
sem_q = np.sqrt(np.array([tw._sat(v) for v in q_var]) / 240.0) / N
sem_c = np.sqrt(var_c / 400.0) / N
k_err = tw.rate_error_bars()
xk = kappas / k1

DENSE = OUT / "cp_dense.npz"
if DENSE.exists():
    dd = np.load(DENSE)
    xk_cp = np.concatenate([xk, dd["kappas_cp"] / k1])
    rho_cp = np.concatenate([rho_c, dd["rho_cp"]])
    var_dense = (dd["sem_cp"] * N) ** 2 * float(dd["n_real"])
    varA_cp = np.concatenate([var_c, var_dense])
    sem_cp = np.concatenate([sem_c, dd["sem_cp"]])
    order = np.argsort(xk_cp)
    xk_cp, rho_cp, sem_cp = xk_cp[order], rho_cp[order], sem_cp[order]
    varA_cp = varA_cp[order]
    xk_rate, rho_rate = dd["kappas_rate"] / k1, dd["rho_rate"]
else:
    print("warning: cp_dense.npz not found -- panels e,f use the sparse grid")
    xk_cp, rho_cp, sem_cp, varA_cp = xk, rho_c, sem_c, var_c
    rho_r = np.array([tw._sat(a) for a in r_mean]) / N
    xk_rate, rho_rate = xk, rho_r

A_cp = rho_cp * N
fano_ok = A_cp >= 1.0
fano_cp_x, fano_cp = xk_cp[fano_ok], varA_cp[fano_ok] / A_cp[fano_ok]
A_q = rho_q * N
varA_q = np.array([tw._sat(v) for v in q_var])
q_ok = A_q >= 1.0
fano_q_x, fano_q = xk[q_ok], varA_q[q_ok] / A_q[q_ok]


def read_rows(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


gamma_rows = read_rows(OLD / "twa_gamma_sweep.csv")
omega_rows = read_rows(OLD / "twa_omega_sweep.csv")
det_rows = read_rows(OLD / "twa_detuning_sweep.csv")


def draw_coherence(ax, fig):
    dc = np.load(OLD / "twa_coherence_collapse.npz")
    gammas = dc["gammas"]; omega_c = float(dc["omega"])
    gnorm = mpl.colors.Normalize(vmin=float(np.min(gammas)), vmax=float(np.max(gammas)))
    cmapv = mpl.colormaps["viridis"]
    alln, ally = [], []
    for g in gammas:
        n = dc[f"n_{g:g}"]; cc = tw._movavg(dc[f"c_{g:g}"], 17)
        m = (n > 0.012) & (n < 0.45)
        ax.plot(n[m], cc[m] * g / omega_c, color=cmapv(gnorm(g)), lw=0.8, alpha=0.65)
        alln.append(n[m]); ally.append(cc[m] * g / omega_c)
    alln_c = np.concatenate(alln); ally_c = np.concatenate(ally)
    bins = np.linspace(0.02, 0.45, 16)
    idx = np.digitize(alln_c, bins); bx = 0.5 * (bins[1:] + bins[:-1])
    by = np.array([ally_c[idx == i].mean() if np.any(idx == i) else np.nan
                   for i in range(1, len(bins))])
    bs = np.array([ally_c[idx == i].std() if np.any(idx == i) else np.nan
                   for i in range(1, len(bins))])
    ax.fill_between(bx, by - bs, by + bs, color="black", alpha=0.12, lw=0,
                    zorder=5)
    hmean, = ax.plot(bx, by, color="black", lw=1.6,
                     label=r"mean $f(n_P)\,\pm$ s.d.", zorder=6)
    ax.set_xlim(0, 0.45); ax.set_ylim(0, 1.10)
    ax.set_xlabel(r"population $n_P$")
    ax.set_ylabel(r"$|\langle S_\perp\rangle|\,\gamma_\phi/\Omega$")
    ax.set_title("Coherence locked to population")
    axi = ax.inset_axes([0.50, 0.63, 0.33, 0.34])
    for g in gammas:
        n = dc[f"n_{g:g}"]; cc = tw._movavg(dc[f"c_{g:g}"], 17)
        m = (n > 0.012) & (n < 0.45)
        axi.plot(n[m], cc[m], color=cmapv(gnorm(g)), lw=0.7, alpha=0.9)
    axi.set_xlim(0, 0.45)
    axi.set_xticks([0, 0.2, 0.4])
    axi.set_xticklabels(["0", "0.2", "0.4"])
    axi.set_title(r"unscaled $|\langle S_\perp\rangle|$", fontsize=fs.a(5.5), pad=1.5)
    axi.set_xlabel(r"$n_P$", fontsize=fs.a(5.5), labelpad=1)
    axi.set_facecolor("white")
    axi.tick_params(labelsize=fs.a(5), width=0.4, length=1.5, pad=1)
    for s in axi.spines.values():
        s.set_linewidth(0.4)
    gsort = np.sort(np.asarray(gammas, dtype=float))
    cmap_d = mpl.colors.ListedColormap([cmapv(gnorm(g)) for g in gsort])
    norm_d = mpl.colors.BoundaryNorm(np.arange(len(gsort) + 1), len(gsort))
    sm = mpl.cm.ScalarMappable(norm=norm_d, cmap=cmap_d); sm.set_array([])
    cax = ax.inset_axes([0.86, 0.58, 0.045, 0.38])
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(len(gsort)) + 0.5)
    cb.ax.set_yticklabels([f"{g:g}" for g in gsort])
    cb.set_label(r"$\gamma_\phi$", fontsize=fs.a(6), labelpad=1)
    cb.ax.tick_params(labelsize=fs.a(5), width=0.5, length=0)
    cb.outline.set_linewidth(0.5)
    ax.legend(handles=[hmean], loc="lower left", fontsize=fs.a(6), handlelength=1.3,
              borderpad=0.2)


def draw_ratelaw_valid(ax, axr):
    """Rate law as the log-log straight line (both dephasing sweeps on one
    line over ~2.5 decades of the scaling variable Omega^2/gamma_phi), with
    the validity domain shown ON the same plot: the low-gamma_phi sweep
    (brown) peels away from the line by up to 45% in the shaded coherent
    regime.  `axr` is a residual strip UNDER the main axes sharing the
    x-axis: the 12^3 lattice deviation from the reduced kinetics, aligned so
    the reader sees WHERE on the rate law the lattice deviates."""
    g = np.array([r["gamma_phi"] for r in gamma_rows])
    kg = np.array([r["k_fac"] for r in gamma_rows])
    ok = np.isfinite(kg) & (kg > 0)
    g, kg = g[ok], kg[ok]
    om = np.array([r["omega"] for r in omega_rows])
    ko = np.array([r["k_fac"] for r in omega_rows])
    gphi_o = float(omega_rows[0]["gamma_phi"])
    x_g = 1.0 / g
    x_o = om ** 2 / gphi_o
    ratios = np.concatenate([kg * g, ko * gphi_o / om ** 2])
    Cfit = float(np.median(ratios))
    Cerr = float(np.std(ratios))
    dec = 2 if Cerr >= 0.005 else 3
    ax.axvspan(1.0 / 3.0, 3.0, color=GREY, alpha=0.15, zorder=0)
    xx = np.geomspace(0.006, 2.6, 60)
    ax.loglog(xx, Cfit * xx, "-", color="black", lw=1.0, zorder=1,
              label=(fr"$k=C\,\Omega^2/\gamma_\phi$,"
                     "\n"
                     fr"$C={Cfit:.{dec}f}\pm{Cerr:.{dec}f}$"))
    ax.loglog(x_g, kg, "o", color=BLUE, **ns.DATA_MARKER, zorder=3,
              label=r"sweep $\gamma_\phi$ ($\Omega=1$)")
    ax.loglog(x_o, ko, "s", color=RED, **ns.DATA_MARKER, zorder=3,
              label=fr"sweep $\Omega$ ($\gamma_\phi={gphi_o:g}$)")
    lowg = OUT / "gamma_sweep_lowg.csv"
    if lowg.exists():
        rows = read_rows(lowg)
        g_lo = np.array([r["gamma_phi"] for r in rows])
        k_lo = np.array([r["k_fac"] for r in rows])
        ax.loglog(1.0 / g_lo, k_lo, "s", markerfacecolor="none",
                  markeredgecolor=BROWN, ms=3.6, markeredgewidth=0.8,
                  ls="none", zorder=3)
    _bt = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(1.1, 0.42, "coherent regime:\nrate law fails",
            transform=_bt, color=BROWN, fontsize=fs.a(5.5), ha="center", va="top",
            linespacing=1.25)
    ax.set_xlim(0.006, 3.0)
    ax.set_ylabel(r"conversion rate $k$")
    ax.set_title("One rate law and its limits")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", fontsize=fs.a(5.5), handlelength=1.2, borderpad=0.2,
              labelspacing=0.3, handletextpad=0.4)
    ns.style_loglog(ax)
    XO = OUT / "crossover_scan.npz"
    if XO.exists():
        dx = np.load(XO)
        axr.axvspan(1.0 / 3.0, 3.0, color=GREY, alpha=0.15, zorder=0)
        axr.loglog(1.0 / dx["gammas"], 100 * dx["eps_sat"], "o", color=BLUE,
                   ms=2.6, markeredgewidth=0.3, markeredgecolor="white",
                   ls="-", lw=0.7)
        axr.set_ylim(0.5, 60)
        axr.set_yticks([1, 10])
        axr.set_yticklabels(["1", "10"])
        axr.minorticks_off()
        axr.set_ylabel("lattice\ndev. (%)", fontsize=fs.a(6), labelpad=2)
        axr.text(0.03, 0.83, r"$12^3$ lattice vs kinetics", fontsize=fs.a(5.5),
                 transform=axr.transAxes, ha="left", va="top")
    axr.set_xlim(0.006, 3.0)
    axr.set_xlabel(r"$\Omega^2/\gamma_\phi$")


def draw_gating(ax):
    """Lorentzian energy-shell gating: activation confined to the resonant
    window Gamma_2 << U, so off-shell configurations are suppressed and the
    process stays short ranged.  Inset: the window width scales linearly with
    the dephasing rate (gamma2_scan.npz), so one fit is not a coincidence."""
    hz = np.array([r["hz"] for r in det_rows])
    kd = np.array([r["k_fac"] for r in det_rows])
    gamma_det = float(det_rows[0]["gamma_phi"])
    order = np.argsort(hz)
    k0 = float(np.nanmax(kd))
    V_window = 20.0
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(lambda x, a, G: a * G ** 2 / (G ** 2 + x ** 2),
                               hz, kd, p0=[k0, gamma_det], maxfev=10000)
        G2err = float(np.sqrt(pcov[1, 1]))
        xx = np.linspace(hz.min(), hz.max(), 200)
        ax.plot(xx, popt[0] * popt[1] ** 2 / (popt[1] ** 2 + xx ** 2), "-",
                color="black", lw=1.0,
                label=(fr"Lorentzian, $\Gamma_2="
                       fr"({popt[1]:.1f}\pm{G2err:.1f})\,\Omega$"))
        half = popt[0] / 2.0
        ax.annotate("", xy=(-popt[1], half), xytext=(popt[1], half),
                    arrowprops=dict(arrowstyle="<|-|>", color="black", lw=0.8,
                                    mutation_scale=6))
        ax.text(0.0, half * 1.06, r"$2\Gamma_2$", ha="center", va="bottom",
                fontsize=fs.a(6))
    except Exception:
        ax.plot(hz[order], kd[order], "-", color="black", lw=1.0)
    ax.plot(hz[order], kd[order], "o", color=BLUE, markersize=3.2,
            markeredgewidth=0.4, markeredgecolor="white", ls="none",
            label=fr"DTWA ($\gamma_\phi={gamma_det:g}$)")
    ax.axvspan(-1.5, 1.5, color=ORANGE, alpha=0.30)
    for m_, lab in [(-V_window, r"$-U$"), (V_window, r"$+U$")]:
        ax.axvline(m_, color=GREY, ls=":", lw=0.7)
        ax.text(m_, k0 * 0.02, lab, ha="center", va="bottom", color=GREY,
                fontsize=fs.a(6))
    ax.set_ylim(0, k0 * 1.22)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    ax.set_xlabel(r"energy defect $h_z/\Omega$")
    ax.set_ylabel(r"conversion rate $k/\Omega$")
    ax.set_title("Energy-shell gating")
    ax.legend(loc="upper left", fontsize=fs.a(5.5), handlelength=1.2, borderpad=0.2,
              labelspacing=0.3, handletextpad=0.4)
    G2 = OUT / "gamma2_scan.npz"
    if G2.exists():
        dg = np.load(G2)
        axi = ax.inset_axes([0.70, 0.58, 0.28, 0.34])
        axi.errorbar(dg["gammas"], dg["Gamma2"], yerr=dg["Gamma2_err"], fmt="o",
                     color=BLUE, ms=2.2, lw=0.6, capsize=1.2,
                     markeredgewidth=0.3, markeredgecolor="white")
        slope = float(np.sum(dg["Gamma2"] * dg["gammas"]) /
                      np.sum(dg["gammas"] ** 2))
        s_err = float(np.sqrt(np.sum((dg["gammas"] * dg["Gamma2_err"]) ** 2))
                      / np.sum(dg["gammas"] ** 2))
        gg = np.linspace(0, float(np.max(dg["gammas"])) * 1.1, 20)
        axi.plot(gg, slope * gg, "--", color="black", lw=0.6)
        axi.set_xlim(0, None); axi.set_ylim(0, None)
        axi.set_xlabel(r"$\gamma_\phi/\Omega$", fontsize=fs.a(5.5), labelpad=1)
        axi.set_ylabel(r"$\Gamma_2/\Omega$", fontsize=fs.a(5.5), labelpad=1)
        last = max(1, int(round(s_err * 100)))
        axi.set_title(fr"$\Gamma_2={slope:.2f}({last})\,\gamma_\phi$",
                      fontsize=fs.a(5.5), pad=1.5)
        axi.set_facecolor("white")
        axi.tick_params(labelsize=fs.a(5), width=0.4, length=1.5, pad=1)
        for s in axi.spines.values():
            s.set_linewidth(0.4)


def draw_additivity(ax):
    mm = np.linspace(0, 6, 50)
    m_f = table_m.astype(float)
    slope_err = float(np.sqrt(np.sum((m_f * np.asarray(k_err)) ** 2))
                      / np.sum(m_f ** 2))
    ax.plot(mm, mm * slope_m, "--", color="black", lw=1.2,
            label=(fr"$k=m\,\bar k_1$" "\n"
                   fr"($\bar k_1={slope_m:.4f}\pm{slope_err:.4f}\,\Omega$)"))
    ax.errorbar(table_m, table_k, yerr=k_err, fmt="o", color=BLUE, ms=3.6,
                markeredgewidth=0.4, markeredgecolor="white", lw=0.8, capsize=1.8,
                label="single-site DTWA")
    nz = table_m >= 1
    dev_pct = float(np.max(np.abs(table_k[nz] - slope_m * m_f[nz])
                           / (slope_m * m_f[nz])) * 100)
    ax.text(0.97, 0.06, fr"additive to $\leq{np.ceil(dev_pct):.0f}\%$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.a(5.5),
            color="black")
    ax.plot([0], [0.0], "o", color=BLUE, ms=3.6, markeredgewidth=0.4,
            markeredgecolor="white")
    ax.annotate("$m{=}0$: $k=0$ exactly\n(absorbing state)",
                xy=(0.08, 0.004), xytext=(1.0, 0.035), fontsize=fs.a(5.5),
                ha="left", va="center", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6,
                                shrinkA=1, shrinkB=2))
    ax.set_xlabel(r"active neighbours $m$")
    ax.set_ylabel(r"conversion rate $k(m)/\Omega$")
    ax.set_title("Additive activation rate")
    ax.legend(loc="upper left", fontsize=fs.a(5.5), handlelength=1.2, borderpad=0.2,
              labelspacing=0.3, handletextpad=0.4)


def draw_kappac(ax):
    FLOOR = 1.0 / N
    live = rho_rate >= FLOOR
    ax.semilogy(xk_rate[live], rho_rate[live], "--", color="black",
                lw=1.3, zorder=2)
    ax.errorbar(xk, np.maximum(rho_q, FLOOR * 0.55), yerr=sem_q, fmt="o",
                color=BLUE, ms=3.6, markeredgewidth=0.4, markeredgecolor="white",
                lw=0.8, capsize=1.6, zorder=4)
    det = rho_cp >= FLOOR
    ax.errorbar(xk_cp[det], rho_cp[det], yerr=sem_cp[det], fmt="D", color=TEAL,
                ms=3.2, markeredgewidth=0.4, markeredgecolor="white", lw=0.8,
                capsize=1.6, zorder=3)
    if np.any(~det):
        ax.errorbar(xk_cp[~det], np.full(np.sum(~det), FLOOR), yerr=FLOOR * 0.35,
                    uplims=True, fmt="_", color=TEAL, ms=4.0, elinewidth=0.8,
                    capsize=0, zorder=3)
    _bt = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(6.0, color="black", ls=":", lw=0.8)
    ax.text(5.9, 0.99, r"$6k_1$ (MF)", color="black", fontsize=fs.a(5.5),
            ha="right", va="top", transform=_bt)
    if np.isfinite(kc_c):
        ax.axvline(kc_c / k1, color=GREY, ls=":", lw=0.8)
        ax.text(kc_c / k1, 0.99, fr"${kc_c/k1:.1f}k_1$ (DP) ", color=GREY,
                fontsize=fs.a(5.5), ha="right", va="top", transform=_bt)
        ax.annotate("", xy=(kc_c / k1, 0.88), xytext=(6.0, 0.88),
                    xycoords=_bt, textcoords=_bt,
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.8,
                                    mutation_scale=6))
        ax.text(0.5 * (kc_c / k1 + 6.0), 0.895, "fluctuation shift",
                color=GREY, fontsize=fs.a(5.5), ha="center", va="bottom",
                transform=_bt)
    ax.text(0.04, 0.52, "jump CP", transform=ax.transAxes,
            color=TEAL, fontsize=fs.a(6), ha="left", va="top")
    ax.text(0.04, 0.32, "DTWA", transform=ax.transAxes,
            color=BLUE, fontsize=fs.a(6), ha="left", va="top")
    ax.text(0.04, 0.14, "rate kinetics", transform=ax.transAxes,
            color="black", fontsize=fs.a(6), ha="left", va="top")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_ylim(FLOOR * 0.30, 0.75)
    ax.set_xlabel(r"recombination $\kappa/k_1$")
    ax.set_ylabel(r"stationary density $\rho_\mathrm{stat}$")
    ax.set_title("Fluctuation-shifted transition")


def draw_switchoff(ax, xlim_from=None):
    """Panel f: the absorbing event itself, run by run.  Fraction of jump-CP
    realisations that END in the empty state (A = 0 at t = 120/Omega; base
    sweep twa_fano_run.py + deep-absorbing extension twa_fano_ext.py, 1200
    realisations per coupling, binomial error bars).  The quantum ensemble is
    plotted at zero at every coupling: a smooth Wigner trajectory can decay
    but can never reach and lock into the exactly empty state -- switching
    off is a discreteness-only event."""
    xs, ps, es = [], [], []
    for cache in (OUT / "fano_run.npz", OUT / "fano_ext.npz"):
        if not cache.exists():
            print(f"warning: {cache.name} not found -- switch-off panel partial")
            continue
        fz = np.load(cache)
        for x in fz["kappas"]:
            A = np.asarray(fz[f"cp_{x:g}"])
            dead = float(np.mean(A[-1] == 0))
            n = A.shape[1]
            xs.append(float(x)); ps.append(dead)
            es.append(float(np.sqrt(dead * (1.0 - dead) / n)))
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]; ps = np.asarray(ps)[order]
    es = np.asarray(es)[order]
    ax.errorbar(xs, ps, yerr=es, fmt="D", color=TEAL, ms=3.4,
                markeredgewidth=0.4, markeredgecolor="white", lw=0.9, ls="-",
                capsize=1.6, zorder=3)
    ax.plot(xk, np.zeros_like(xk), "o", color=BLUE, ms=3.4,
            markeredgewidth=0.4, markeredgecolor="white", ls="-", lw=0.9,
            zorder=4)
    _bt = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if np.isfinite(kc_c):
        ax.axvline(kc_c / k1, color=GREY, ls=":", lw=0.8)
        ax.text(kc_c / k1, 0.99, fr"${kc_c/k1:.1f}k_1$ (DP) ", color=GREY,
                fontsize=fs.a(5.5), ha="right", va="top", transform=_bt)
    ax.axvline(6.0, color="black", ls=":", lw=0.8)
    ax.text(6.1, 0.52, r"$6k_1$ (MF)", color="black", fontsize=fs.a(5.5),
            ha="left", va="center", transform=_bt)
    ax.text(0.05, 0.92, "jump CP", transform=ax.transAxes, color=TEAL,
            fontsize=fs.a(6), ha="left", va="top")
    ax.text(5.85, 0.10, "DTWA: never absorbs", transform=_bt,
            color=BLUE, fontsize=fs.a(6), ha="right", va="bottom")
    ax.set_ylim(-0.045, 1.045)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    if xlim_from is not None:
        ax.set_xlim(xlim_from.get_xlim())
    ax.set_xlabel(r"recombination $\kappa/k_1$")
    ax.set_ylabel("fraction of runs switched off")
    ax.set_title("Switch-off requires discreteness")


_w, _h = fs.size(ns.COL_WIDTH_DOUBLE, 5.0)
fig = plt.figure(figsize=(_w * 1.15, _h * 1.06))
gs = fig.add_gridspec(2, 3)
ax_a = fig.add_subplot(gs[0, 0])
gs_b = gs[0, 1].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.10)
ax_b = fig.add_subplot(gs_b[0])
ax_b2 = fig.add_subplot(gs_b[1], sharex=ax_b)
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

draw_coherence(ax_a, fig);            ns.panel_label(ax_a, "a")
draw_ratelaw_valid(ax_b, ax_b2);      ns.panel_label(ax_b, "b")
draw_gating(ax_c);                    ns.panel_label(ax_c, "c")
draw_additivity(ax_d);                ns.panel_label(ax_d, "d")
draw_kappac(ax_e);                    ns.panel_label(ax_e, "e")
draw_switchoff(ax_f, xlim_from=ax_e); ns.panel_label(ax_f, "f")

fig.tight_layout(pad=0.6, w_pad=1.6, h_pad=2.0)
for base in (Path(HERE), MAIN_GFX, DRAFT_GFX):
    base.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = base / f"figure_11.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out)
