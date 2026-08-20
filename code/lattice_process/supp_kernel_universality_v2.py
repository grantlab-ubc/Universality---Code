"""Supplementary test: kernel-independence of the DP exponents.
  The four structurally different short-range kernels collapse onto the DP
  values. The long-range kernel is drifted toward the mean-field / Levy point.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "supplementary_results")
CACHE = os.path.join(OUTPUT_DIR, "supp_kernel_universality_v2.npz")
MAIN_SCRIPT = os.path.join(HERE, "dp_universality_simulation.py")

_spec = importlib.util.spec_from_file_location("uc_main", MAIN_SCRIPT)
uc = importlib.util.module_from_spec(_spec)
sys.modules["uc_main"] = uc
_spec.loader.exec_module(uc)

L = 30
ALPHA = 0.13
GAMMA_BASE = 1.85
BETA = 0.1
GAMMA_LOSS = 0.0002
MAX_T = 250.0
DT = 0.09
PERIODIC = True
DP_REF = uc.DP_AVALANCHE["(3+1)D DP"]
MF_REF = uc.DP_AVALANCHE["Mean-field"]

F_STAR = 0.013


def _raw_shifts(r_cut, form, d_0=1.0, sigma=1.0):
    out = []
    R = int(math.floor(r_cut))
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                if dx == dy == dz == 0:
                    continue
                d2 = dx * dx + dy * dy + dz * dz
                if d2 <= r_cut * r_cut:
                    r = math.sqrt(d2)
                    if form == "exp":
                        w = math.exp(-r / d_0)
                    elif form == "gauss":
                        w = math.exp(-(r * r) / (2.0 * d_0 * d_0))
                    elif form == "tophat":
                        w = 1.0
                    elif form == "nn":
                        w = 1.0 if d2 == 1 else 0.0
                    elif form == "ramp":
                        w = max(0.0, r_cut - r)
                    elif form == "longrange":
                        w = r ** (-(3.0 + sigma))
                    else:
                        raise ValueError(form)
                    if w > 0:
                        out.append((dx, dy, dz, r, w))
    return out


def make_shifts(form, r_cut, target_weight, d_0=1.0, sigma=1.0):
    raw = _raw_shifts(r_cut, form, d_0, sigma)
    s = sum(w for *_xyz, _r, w in raw)
    scale = (target_weight / s) if (target_weight is not None and s > 0) else 1.0
    shifts = [(dx, dy, dz, np.float32(w * scale)) for dx, dy, dz, _r, w in raw]
    sec_mom = sum((r * r) * (w * scale) for *_xyz, r, w in raw)
    return shifts, s * scale, sec_mom


KERNELS = [
    dict(key="exp",    label=r"exponential $e^{-r}$",   form="exp",    r_cut=2.0,
         sigma=1.0, longrange=False, grid=[1.998, 2.002, 2.005, 2.009, 2.013, 2.016, 2.020]),
    dict(key="gauss",  label=r"Gaussian $e^{-r^2/2}$",  form="gauss",  r_cut=2.0,
         sigma=1.0, longrange=False, grid=[2.014, 2.019, 2.024, 2.029, 2.034, 2.039, 2.044]),
    dict(key="tophat", label=r"uniform (top-hat)",      form="tophat", r_cut=2.0,
         sigma=1.0, longrange=False, grid=[1.965, 1.970, 1.975, 1.980, 1.985, 1.990, 1.995]),
    dict(key="ramp",   label=r"linear ramp $(r_c{-}r)$", form="ramp",  r_cut=2.0,
         sigma=1.0, longrange=False, grid=[2.052, 2.058, 2.064, 2.070, 2.076, 2.082, 2.088]),
    dict(key="lr1", label=r"long-range $r^{-1}$ ($r_c{=}3$)", form="longrange", r_cut=3.0,
         sigma=-2.0, longrange=True,
         grid=[1.852, 1.860, 1.868, 1.878, 1.890]),
    dict(key="lr2", label=r"long-range $r^{-1}$ ($r_c{=}4$)", form="longrange", r_cut=4.0,
         sigma=-2.0, longrange=True,
         grid=[1.826, 1.832, 1.839, 1.847, 1.856]),
    dict(key="lr3", label=r"long-range $r^{-1}$ ($r_c{=}5$)", form="longrange", r_cut=5.0,
         sigma=-2.0, longrange=True, L=48, n_runs=3000,
         grid=[1.806, 1.812, 1.818, 1.824, 1.830, 1.836]),
    dict(key="lrmf", label=r"long-range $r^{-1}$ ($r_c{=}9$, near MF)", form="longrange",
         r_cut=9.0, sigma=-2.0, longrange=True, L=40, n_runs=8000, match_fspan=0.006,
         grid=[1.790, 1.795, 1.800, 1.805, 1.810]),
]


def run_avalanches(shifts, lam, num_runs, base_seed, ncores=None, L_box=L):
    ncores = ncores or max(1, min(os.cpu_count() or 1, 8))
    per = [num_runs // ncores] * ncores
    per[0] += num_runs % ncores
    ss = np.random.SeedSequence(base_seed)
    seeds = [int(x.generate_state(1)[0]) for x in ss.spawn(ncores)]
    args = [(n, sd, L_box, MAX_T, DT, lam, ALPHA, GAMMA_BASE, BETA, GAMMA_LOSS, shifts, PERIODIC)
            for n, sd in zip(per, seeds) if n > 0]
    sizes, durs, areas = [], [], []
    for s, d, a, _vf in uc._run_batches(uc._run_simulation_batch, args, ncores):
        sizes += s
        durs += d
        areas += a
    return np.asarray(sizes, float), np.asarray(durs, float), np.asarray(areas, float)


def susceptibility(sizes, areas, L_box=L):
    """Finite-size susceptibility from non-spanning avalanches, chi=<S^2>/<S>."""
    span = areas >= 0.5 * L_box ** 3
    ns = sizes[~span]
    chi = float((ns ** 2).mean() / ns.mean()) if ns.size and ns.mean() > 0 else 0.0
    return chi, float(np.mean(span))


def pdf_curvature(sizes):
    x, y = uc.log_binned_pdf(np.asarray(sizes, float))
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0)
    lx, ly = np.log10(x[m]), np.log10(y[m])
    if len(lx) < 6:
        return np.inf
    lo, hi = np.percentile(lx, [8, 96])
    sel = (lx >= lo) & (lx <= hi)
    if sel.sum() < 6:
        sel = np.ones_like(lx, dtype=bool)
    return abs(float(np.polyfit(lx[sel], ly[sel], 2)[0]))


def fit_exponents(sizes, durs):
    
    fs = uc.choose_distribution_fit(sizes, uc.FIT_CONFIG["S"])
    ft = uc.choose_distribution_fit(durs, uc.FIT_CONFIG["T"])
    return fs.exponent, fs.error, ft.exponent, ft.error


def interp_at_fstar(fspan, y, target=F_STAR):
    """Linear interpolation of y(f_span) to `target` (f_span is monotonic in lambda)."""
    o = np.argsort(fspan)
    return float(np.interp(target, np.asarray(fspan)[o], np.asarray(y)[o]))


def match_with_error(fspan, y, yerr, window=0.013, target=F_STAR):
    
    ff = np.asarray(fspan, float); yy = np.asarray(y, float); ye = np.asarray(yerr, float)
    sel = np.abs(ff - target) <= window
    if sel.sum() < 5:
        order = np.argsort(np.abs(ff - target))
        sel = np.zeros_like(ff, bool); sel[order[:min(5, len(ff))]] = True
    x = ff[sel] - target
    yv = yy[sel]
    w = 1.0 / np.clip(ye[sel], 1e-3, None) ** 2
    deg = 2 if len(x) >= 5 else (1 if len(x) >= 3 else 0)
    cols = [np.ones_like(x)] + [x ** p for p in range(1, deg + 1)]
    X = np.vstack(cols).T
    cov = np.linalg.inv(X.T @ (w[:, None] * X))
    beta = cov @ (X.T @ (w * yv))
    resid = yv - X @ beta
    dof = max(len(x) - (deg + 1), 1)
    chi2 = float(np.sum(w * resid ** 2))
    cov_scaled = cov * (chi2 / dof)
    val = float(beta[0])
    se = float(math.sqrt(max(cov_scaled[0, 0], 0.0)))
    return val, se


def parabolic_peak(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmax(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i])
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return float(x1)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if A >= 0:
        return float(x1)
    return float(min(max(-B / (2 * A), x0), x2))


def scan_kernel(kspec, base_w, num_scan):
    shifts, tot_w, sec_mom = make_shifts(kspec["form"], kspec["r_cut"],
                                         target_weight=base_w, sigma=kspec["sigma"],
                                         d_0=kspec.get("d_0", 1.0))
    L_box = int(kspec.get("L", L))
    target = float(kspec.get("match_fspan", F_STAR))
    lams = np.array(kspec["grid"], float)
    chi, fspan, tauS, tauSe, alpT, alpTe, curv = ([] for _ in range(7))
    best = dict(df=np.inf, S=np.array([]), T=np.array([]), lam=np.nan, f=np.nan)
    rng = np.random.default_rng(20260620)
    print(f"\n=== scan '{kspec['key']}' ({kspec['label']}): {len(shifts)} nbrs, "
          f"L={L_box}, 2nd_mom={sec_mom:.2f}, {len(lams)} lambda ===")
    for i, lam in enumerate(lams):
        s, d, a = run_avalanches(shifts, lam, num_scan,
                                 base_seed=7000 + 137 * i + hash(kspec["key"]) % 991,
                                 L_box=L_box)
        c, f = susceptibility(s, a, L_box=L_box)
        ts, tse, at, ate = fit_exponents(s, d)
        cv = pdf_curvature(s)
        chi.append(c); fspan.append(f)
        tauS.append(ts); tauSe.append(tse); alpT.append(at); alpTe.append(ate)
        curv.append(cv)
        if abs(f - target) < best["df"]:
            keep = 60000
            si = s if s.size <= keep else s[rng.choice(s.size, keep, replace=False)]
            di = d if d.size <= keep else d[rng.choice(d.size, keep, replace=False)]
            best = dict(df=abs(f - target), S=np.asarray(si, np.float32),
                        T=np.asarray(di, np.float32), lam=lam, f=f)
        print(f"  lam={lam:.3f}  chi={c:9.0f}  f_span={f:.3f}  "
              f"tau_S={ts:.3f}+-{tse:.3f}  alpha_T={at:.3f}+-{ate:.3f}  curv={cv:.3f}")
    chi = np.array(chi); fspan = np.array(fspan)
    tauS = np.array(tauS); tauSe = np.array(tauSe)
    alpT = np.array(alpT); alpTe = np.array(alpTe)
    lam_star = interp_at_fstar(fspan, lams, target)
    tS, tSe = match_with_error(fspan, tauS, tauSe, target=target)
    aT, aTe = match_with_error(fspan, alpT, alpTe, target=target)
    print(f"  -> f*={target:.3f}: lambda*={lam_star:.3f}  "
          f"tau_S*={tS:.3f}+-{tSe:.3f}  alpha_T*={aT:.3f}+-{aTe:.3f}  "
          f"[chi peak {parabolic_peak(lams, chi):.3f}]  dist@lam={best['lam']:.3f} f={best['f']:.3f}")
    return dict(label=kspec["label"], longrange=kspec["longrange"],
                n_neighbours=len(shifts), second_moment=sec_mom, L_box=L_box,
                match_fspan=target,
                lams=lams, chi=chi, fspan=fspan,
                tauS=tauS, tauSe=tauSe, alpT=alpT, alpTe=alpTe, curv=np.array(curv),
                lam_star=lam_star, tauS_star=tS, tauS_star_e=tSe,
                alpT_star=aT, alpT_star_e=aTe,
                dist_S=best["S"], dist_T=best["T"],
                dist_lam=best["lam"], dist_fspan=best["f"])


def save_cache(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    flat = {"__keys__": np.array([k["key"] for k in KERNELS])}
    for key, r in results.items():
        for name, val in r.items():
            flat[f"{key}/{name}"] = np.array(val) if name == "label" else np.asarray(val)
    np.savez(CACHE, **flat)
    print(f"\nSaved cache: {CACHE}")


def load_cache():
    z = np.load(CACHE, allow_pickle=True)
    keys = [str(k) for k in z["__keys__"]]
    results = {}
    for key in keys:
        results[key] = {f.split("/", 1)[1]: z[f] for f in z.files if f.startswith(f"{key}/")}
    return results


def _scalar(v):
    return float(np.asarray(v).reshape(-1)[0])


def make_figure(results):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    rc = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.6,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4.0,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    mpl.rcParams.update(rc)
    palette = {"exp": "#0072B2", "gauss": "#009E73", "tophat": "#E69F00",
               "ramp": "#CC79A7", "nn": "#CC79A7", "lr1": "#D55E00", "lr2": "#882255",
               "lr3": "#332288", "lrmf": "#332288"}
    markers = {"exp": "o", "gauss": "s", "tophat": "^", "ramp": "D", "nn": "v",
               "lr1": "v", "lr2": "v", "lr3": "h", "lrmf": "h"}
    short = [k for k in ["exp", "gauss", "tophat", "ramp", "nn"] if k in results]
    lr_keys = [k for k in ["lr2", "lrmf"] if k in results]
    ordered = short + lr_keys

    kernel_label = {
        "exp": "exponential", "gauss": "Gaussian",
        "tophat": "uniform", "ramp": "linear ramp",
        "nn": "nearest-neighbour",
        "lr2": r"long-range, $\langle r^2\rangle_w{\approx}59$",
        "lrmf": r"long-range, $\langle r^2\rangle_w{\approx}300$",
    }

    fig = plt.figure(figsize=(7.0, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.04], wspace=0.30)
    axD = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])

    tauD = DP_REF["tau"]
    XLO, XHI = 6.0, 400.0
    lr_curve = {}
    for key in ordered:
        r = results[key]
        S = np.sort(np.asarray(r.get("dist_S", []), float))
        S = S[S > 0]
        if S.size < 500:
            continue
        xg = np.geomspace(XLO, XHI, 50)
        cc = 1.0 - np.searchsorted(S, xg, side="left") / S.size
        m = cc > 0
        xg, cc = xg[m], cc[m]
        yc = cc * xg ** (tauD - 1.0)
        sel = (xg >= 8) & (xg <= 90)
        norm = np.median(yc[sel]) if sel.any() else np.median(yc)
        y = yc / norm
        lr = (key in lr_keys)
        axD.plot(xg, y, ls="--" if lr else "-", color=palette[key],
                 lw=2.0 if lr else 1.2, alpha=0.95, zorder=6 if lr else 4,
                 label=kernel_label.get(key, key))
    axD.axhline(1.0, color="black", ls="-.", lw=0.9, zorder=3)
    axD.set_xscale("log"); axD.set_yscale("log")
    axD.set_ylim(0.78, 1.40); axD.set_xlim(XLO, XHI)
    axD.set_xlabel(r"avalanche size $S$")
    axD.set_ylabel(r"compensated survival $C(S)\,S^{\,\tau_\mathrm{DP}-1}$ (norm.)")
    axD.set_title(r"Critical size distributions ($\tau_\mathrm{DP}{=}1.40$)")
    axD.text(0.035, 0.06, r"flat $=$ DP", transform=axD.transAxes, fontsize=6.2, va="bottom")
    axD.legend(loc="upper right", fontsize=4.3, handletextpad=0.5, labelspacing=0.22,
               borderpad=0.4, handlelength=1.6, frameon=True, framealpha=0.92,
               edgecolor="0.8", facecolor="white")
    axD.text(-0.26, 1.03, "a", transform=axD.transAxes, fontsize=9, fontweight="bold")

    ax = axC
    print("\nMatched-criticality exponents at f* = %.3f:" % F_STAR)
    pts = {}
    for key in ordered:
        r = results[key]
        tgt = float(_scalar(r["match_fspan"])) if "match_fspan" in r else F_STAR
        ff = np.asarray(r["fspan"], float)
        if "point_estimate" in r and int(_scalar(r["point_estimate"])):
            ts, tse = _scalar(r["tauS_star"]), _scalar(r["tauS_star_e"])
            at, ate = _scalar(r["alpT_star"]), _scalar(r["alpT_star_e"])
        else:
            ts, tse = match_with_error(ff, r["tauS"], r["tauSe"], target=tgt)
            at, ate = match_with_error(ff, r["alpT"], r["alpTe"], target=tgt)
        sm = _scalar(r["second_moment"]); lc = interp_at_fstar(ff, r["lams"], tgt)
        pts[key] = (ts, tse, at, ate, sm)
        print(f"  {key:9s}: tau_S={ts:.3f}+-{tse:.3f}  alpha_T={at:.3f}+-{ate:.3f}  "
              f"lambda*={lc:.3f}  2nd_mom={sm:.2f}")

    def _pt(axx, key, ms, ew, cs, zo, label=None, alpha=1.0):
        ts, tse, at, ate, _sm = pts[key]
        axx.errorbar(ts, at, xerr=tse, yerr=ate, fmt=markers[key], color=palette[key],
                     ms=ms, markeredgewidth=0.4, markeredgecolor="white",
                     elinewidth=ew, capsize=cs, zorder=zo, label=label, alpha=alpha)

    ax.plot(DP_REF["tau"], DP_REF["alpha_T"], marker="*", mfc="white", mec="black",
            mew=0.9, ms=10, ls="none", zorder=4, label=r"$(3{+}1)$D DP")
    ax.plot(MF_REF["tau"], MF_REF["alpha_T"], marker="o", mfc="white", mec="0.35",
            mew=0.9, ms=6, ls="none", zorder=4, label="mean-field / Lévy")
    for key in short:
        _pt(ax, key, ms=3.8, ew=0.8, cs=1.7, zo=6, alpha=0.75,
            label=kernel_label.get(key, key))
    for key in lr_keys:
        _pt(ax, key, ms=4.6, ew=0.9, cs=2.0, zo=7, alpha=0.95,
            label=kernel_label.get(key, key))

    ax.set_xlim(1.37, 1.53); ax.set_ylim(1.60, 2.04)
    ax.set_xlabel(r"size exponent $\tau_S$")
    ax.set_ylabel(r"duration exponent $\alpha_T$")
    ax.set_title("Matched-criticality exponents")
    ax.legend(loc="lower right", fontsize=4.1, handletextpad=0.3, labelspacing=0.2,
              borderpad=0.4, handlelength=1.0, markerscale=0.55, frameon=True,
              framealpha=0.92, edgecolor="0.8", facecolor="white")
    ax.text(-0.20, 1.03, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.tight_layout(pad=0.5, w_pad=1.5)
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(OUTPUT_DIR, "supp_fig10_kernel_universality_v2" + ext))
    print("Wrote supp_fig10_kernel_universality_v2.{pdf,png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--only", type=str, default=None,
                    help="run only this kernel key, merge into the existing cache, replot")
    ap.add_argument("--num-scan", type=int, default=3500)
    args = ap.parse_args()

    if args.replot:
        make_figure(load_cache())
        return

    _, base_w, _ = make_shifts("exp", 2.0, target_weight=None)
    print(f"baseline integrated weight = {base_w:.4f}")

    todo = [k for k in KERNELS if (args.only is None or k["key"] == args.only)]
    results = load_cache() if args.only else {}
    for kspec in todo:
        n = (700 if args.quick else
             kspec.get("n_runs") or
             (max(5000, args.num_scan // 2) if kspec["longrange"] else args.num_scan))
        results[kspec["key"]] = scan_kernel(kspec, base_w, n)

    save_cache(results)
    make_figure(results)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
