"""Single site truncated Wigner method for the reduction of the quantum
facilitation to a classical rate.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


Array = np.ndarray


def _rotate_about_axis(
    sx: Array, sy: Array, sz: Array, wx: float, wy: float, wz: float, h: float
) -> Tuple[Array, Array, Array]:
    """Rodrigues rotation of spin vectors about the fixed axis w by angle |w|*h."""
    norm = math.sqrt(wx * wx + wy * wy + wz * wz)
    if norm < 1e-14:
        return sx, sy, sz
    ux, uy, uz = wx / norm, wy / norm, wz / norm
    angle = norm * h
    c, s = math.cos(angle), math.sin(angle)
    dot = ux * sx + uy * sy + uz * sz
    rx = sx * c + (uy * sz - uz * sy) * s + ux * dot * (1.0 - c)
    ry = sy * c + (uz * sx - ux * sz) * s + uy * dot * (1.0 - c)
    rz = sz * c + (ux * sy - uy * sx) * s + uz * dot * (1.0 - c)
    return rx, ry, rz


def evolve_single_site(
    omega: float,
    gamma_phi: float,
    hz: float,
    t_max: float,
    dt: float,
    n_traj: int,
    sample_every: int,
    seed: int,
) -> Tuple[Array, Array, Array]:
    """Evolve a driven, dephased two-level site under a fixed local field h_z.
    """
    rng = np.random.default_rng(seed)
    sx = rng.choice(np.array([-1.0, 1.0]), size=n_traj)
    sy = rng.choice(np.array([-1.0, 1.0]), size=n_traj)
    sz = -np.ones(n_traj)

    n_steps = int(round(t_max / dt))
    half = 0.5 * dt
    times: List[float] = []
    n_up: List[float] = []
    coh: List[float] = []
    sqrt_gamma = math.sqrt(gamma_phi) if gamma_phi > 0 else 0.0

    for step in range(n_steps + 1):
        if step % sample_every == 0 or step == n_steps:
            times.append(step * dt)
            mz = float(np.mean(sz))
            n_up.append(0.5 * (1.0 + mz))
            coh.append(math.hypot(float(np.mean(sx)), float(np.mean(sy))))
        if step == n_steps:
            break
        sx, sy, sz = _rotate_about_axis(sx, sy, sz, 2.0 * omega, 0.0, hz, half)
        if sqrt_gamma > 0.0:
            theta = -2.0 * sqrt_gamma * (math.sqrt(dt) * rng.standard_normal(n_traj))
            ct, st = np.cos(theta), np.sin(theta)
            sx, sy = sx * ct - sy * st, sx * st + sy * ct
        sx, sy, sz = _rotate_about_axis(sx, sy, sz, 2.0 * omega, 0.0, hz, half)

    return np.asarray(times), np.asarray(n_up), np.asarray(coh)


def extract_rate(times: Array, n_up: Array) -> Tuple[float, float]:

    mask = (n_up > 0.02) & (n_up < 0.40) & np.isfinite(n_up)
    if int(np.sum(mask)) < 5:
        mask = (n_up > 0.01) & (n_up < 0.45) & np.isfinite(n_up)
    if int(np.sum(mask)) < 5:
        return math.nan, math.nan
    t = times[mask]
    y = np.log(np.clip(1.0 - 2.0 * n_up[mask], 1e-6, None))
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = math.nan if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot
    return float(-0.5 * slope), r2


@dataclass
class RunConfig:
    omega: float = 1.0
    V_nn: float = 20.0
    n_traj: int = 6000
    dt: float = 0.002
    seed: int = 20260604


def gamma_sweep(cfg: RunConfig, gammas: List[float]) -> List[Dict[str, float]]:
    """k_fac vs dephasing at the facilitation resonance (h_z = 0)."""
    rows: List[Dict[str, float]] = []
    for g in gammas:
        t_max = max(30.0, 1.2 * g / (cfg.omega ** 2))
        sample = max(1, int(round((t_max / 400.0) / cfg.dt)))
        times, n_up, _ = evolve_single_site(
            cfg.omega, g, 0.0, t_max, cfg.dt, cfg.n_traj, sample, cfg.seed + int(1000 * g)
        )
        k, r2 = extract_rate(times, n_up)
        rows.append({"gamma_phi": g, "omega": cfg.omega, "k_fac": k, "r2": r2,
                     "k_gamma_product": k * g})
        print(f"  gamma={g:7.3f}  k_fac={k:.5e}  k*gamma={k*g:.4f}  R2={r2:.4f}")
    return rows


def omega_sweep(cfg: RunConfig, omegas: List[float], gamma_phi: float) -> List[Dict[str, float]]:
    """k_fac vs drive at fixed strong dephasing, resonance (h_z = 0)."""
    rows: List[Dict[str, float]] = []
    for om in omegas:
        t_max = max(30.0, 1.2 * gamma_phi / (om ** 2))
        sample = max(1, int(round((t_max / 400.0) / cfg.dt)))
        times, n_up, _ = evolve_single_site(
            om, gamma_phi, 0.0, t_max, cfg.dt, cfg.n_traj, sample, cfg.seed + int(7919 * om)
        )
        k, r2 = extract_rate(times, n_up)
        rows.append({"omega": om, "gamma_phi": gamma_phi, "k_fac": k, "r2": r2,
                     "k_over_omega2": k / (om ** 2)})
        print(f"  omega={om:6.3f}  k_fac={k:.5e}  k/omega^2={k/(om**2):.5f}  R2={r2:.4f}")
    return rows


def detuning_sweep(cfg: RunConfig, gamma_phi: float, hz_values: Array) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for hz in hz_values:
        k_guess = (cfg.omega ** 2) * gamma_phi / (gamma_phi ** 2 + hz ** 2)
        t_max = float(np.clip(1.5 / max(k_guess, 1e-4), 40.0, 4000.0))
        sample = max(1, int(round((t_max / 400.0) / cfg.dt)))
        times, n_up, _ = evolve_single_site(
            cfg.omega, gamma_phi, float(hz), t_max, cfg.dt, cfg.n_traj, sample,
            cfg.seed + int(31 * (hz + 50)),
        )
        k, r2 = extract_rate(times, n_up)
        rows.append({"hz": float(hz), "gamma_phi": gamma_phi, "omega": cfg.omega,
                     "k_fac": k if math.isfinite(k) else 0.0, "r2": r2})
        print(f"  hz={hz:7.3f}  k_fac={k if math.isfinite(k) else 0.0:.5e}  R2={r2:.4f}")
    return rows


def coherence_traces(
    cfg: RunConfig, gammas: List[float], n_traj_coh: int = 40000
) -> Dict[float, Tuple[Array, Array, Array]]:

    traces: Dict[float, Tuple[Array, Array, Array]] = {}
    for g in gammas:
        t_max = max(40.0, 1.5 * g / (cfg.omega ** 2))
        sample = max(1, int(round((t_max / 600.0) / cfg.dt)))
        times, n_up, coh = evolve_single_site(
            cfg.omega, g, 0.0, t_max, cfg.dt, n_traj_coh, sample, cfg.seed + 5 * int(g)
        )
        traces[g] = (times, n_up, coh)
    return traces


def _movavg(y: Array, w: int = 7) -> Array:
    if len(y) < w or w < 2:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def write_report(path: Path, cfg: RunConfig, gamma_rows, omega_rows, det_rows) -> None:
    k_gamma = [r["k_gamma_product"] for r in gamma_rows if math.isfinite(r["k_fac"])]
    k_over_o2 = [r["k_over_omega2"] for r in omega_rows if math.isfinite(r["k_fac"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("one-active-neighbour facilitation resonance.\n\n")
        f.write(f"Parameters: Omega={cfg.omega}, V_nn={cfg.V_nn}, trajectories={cfg.n_traj}, dt={cfg.dt}\n\n")
        if k_gamma:
            f.write(f"(b) k_fac * gamma_phi (should be ~const = C*Omega^2):\n")
            f.write(f"    mean={np.mean(k_gamma):.4f}, std={np.std(k_gamma):.4f}, "
                    f"rel.spread={np.std(k_gamma)/max(np.mean(k_gamma),1e-12):.3f}\n")
        if k_over_o2:
            f.write(f"(b) k_fac / Omega^2 (should be ~const = C/gamma_phi):\n")
            f.write(f"    mean={np.mean(k_over_o2):.5f}, std={np.std(k_over_o2):.5f}, "
                    f"rel.spread={np.std(k_over_o2)/max(np.mean(k_over_o2),1e-12):.3f}\n")
        f.write("\n(c) facilitation resonance k_fac(h_z): peak at h_z=0 (one active neighbour),\n")
        f.write("    suppressed at h_z=+/-V_nn (zero/two neighbours).\n")
        on = next((r["k_fac"] for r in det_rows if abs(r["hz"]) < 1e-9), float("nan"))
        off = next((r["k_fac"] for r in det_rows if abs(r["hz"] - cfg.V_nn) < 1e-6), float("nan"))
        if math.isfinite(on) and math.isfinite(off) and off > 0:
            f.write(f"    contrast k(0)/k(V_nn) = {on/off:.1f}\n")
    print(f"Wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="TWA quantum->classical facilitation-rate validation for the DP paper.")
    ap.add_argument("--out", type=Path, default=Path("twa_dp_support_results"))
    ap.add_argument("--traj", type=int, default=6000)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true",
                    help="Rebuild the figure from cached CSVs in --out.")
    ap.add_argument("--fig-path", type=Path, default=None,
                    help="Output path for the figure (default: <out>/twa_dp_support_figure.png).")
    args = ap.parse_args()

    if args.replot:
        fig_path = args.fig_path or (args.out / "twa_dp_support_figure.png")
        replot_from_csv(args.out, fig_path)
        return

    cfg = RunConfig(n_traj=args.traj, dt=args.dt)
    if args.quick:
        cfg.n_traj = 1500
        cfg.dt = 0.004

    gammas = [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
    omegas = [0.5, 0.7, 1.0, 1.4, 2.0]
    gamma_for_omega = 32.0
    gamma_det = 6.0
    hz_values = np.array([-26, -22, -20, -18, -14, -10, -7, -5, -3, -2, -1, 0,
                          1, 2, 3, 5, 7, 10, 14, 18, 20, 22, 26], float)
    coh_gammas = [8.0, 16.0, 32.0]
    if args.quick:
        gammas = [8.0, 16.0, 32.0]
        omegas = [0.5, 1.0, 2.0]
        gamma_det = 6.0
        hz_values = np.array([-20, -14, -8, -4, 0, 4, 8, 14, 20], float)
        coh_gammas = [16.0]

    print("\n[a/b] gamma sweep at facilitation resonance h_z=0:")
    gamma_rows = gamma_sweep(cfg, gammas)
    print(f"\n[b] omega sweep at gamma_phi={gamma_for_omega}:")
    omega_rows = omega_sweep(cfg, omegas, gamma_for_omega)
    print(f"\n[c] detuning (facilitation-resonance) sweep at gamma_phi={gamma_det}, V_nn={cfg.V_nn}:")
    det_rows = detuning_sweep(cfg, gamma_det, hz_values)
    print("\n[a] coherence traces:")
    traces = coherence_traces(cfg, coh_gammas)

    out = args.out
    write_csv(out / "twa_gamma_sweep.csv", gamma_rows)
    write_csv(out / "twa_omega_sweep.csv", omega_rows)
    write_csv(out / "twa_detuning_sweep.csv", det_rows)
    coh_rows = [
        {"gamma_phi": g, "t": float(t), "n_up": float(n), "coherence": float(c)}
        for g, (ts, ns, cs) in sorted(traces.items())
        for t, n, c in zip(ts, ns, cs)
    ]
    write_csv(out / "twa_coherence_traces.csv", coh_rows)
    make_figure(out / "twa_dp_support_figure.png", cfg, gamma_rows, omega_rows, det_rows, traces)
    write_report(out / "twa_dp_support_report.txt", cfg, gamma_rows, omega_rows, det_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
