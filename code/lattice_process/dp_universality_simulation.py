

from __future__ import annotations

import os
import time
import math
import json
import hashlib
import multiprocessing as mp
import concurrent.futures
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np



OUTPUT_DIR = "dp_loss_window_results"
PERIODIC = True
USE_SIMULATION_CACHE = True
CACHE_DIR = os.path.join(OUTPUT_DIR, "simulation_cache")
CACHE_SCHEMA_VERSION = 2
MAX_WORKERS = 9
MAIN_LAMBDA_FAC = 2.008

LOSSLESS_GAMMA_LOSS = 2.0e-4

LOSS_WINDOW_EPS = 0.20

FIT_CONFIG = {
    "S":  {"auto_xmin": True, "x_min": None, "x_max": None},
    "T":  {"auto_xmin": True, "x_min": None, "x_max": None},
    "A":  {"auto_xmin": True, "x_min": None, "x_max": None},
    "ST": {"T_min": 5.0, "T_max": 40.0},
    "spreading": {
        "P":  {"t_min": 2.0, "t_max": 70.0},
        "N":  {"t_min": 0.6, "t_max": 10},
        "R2": {"t_min": 0.6, "t_max": 40.0},
    },
    "critical_scan": {
        "P":  {"t_min": 2.0, "t_max": 70.0},
        "N":  {"t_min": 0.7, "t_max": 10.0},
        "R2": {"t_min": 0.7, "t_max": 40.0},
    },
    "critical_decay": {"t_min": 5.0, "t_max": 70.0},
    "FSS_cutoff": {"L_min": 8, "L_max": None},
    "FSS_exponents": {
        "L_min": 12,
        "L_max": None,
        "loss_safe_fraction": 0.85,
        "weighted": True,
    },
}

FSS_L_VALUES = [
    8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
    28, 30, 32, 34, 36, 38, 40
]
FSS_RUNS_EACH = 20000
FSS_CUTOFF_METHOD = "moment"
FSS_CUTOFF_QUANTILE = 0.99
FSS_BOOTSTRAP_SAMPLES = 600
FSS_MIN_SIZE = 2.0
FSS_MIN_DURATION = None
FSS_GAMMA_LOSS = 2.0e-4
FSS_MAX_T = 1800.0
FSS_LAMBDA_FAC = 2.008

SPREADING_NUM_RUNS = 1200
SPREADING_MAX_T = 300.0

RUN_CRITICAL_SCAN = True
RUN_LOSS_SWEEP = True
RUN_ACTIVE_DENSITY_DECAY = True
RUN_AVALANCHE_LOSS_EXPONENT_SWEEP = True
RUN_FSS_EXPONENT_EXTRAPOLATION = True

CRITICAL_SCAN_LAMBDA_VALUES = [
    
     1.96, 2.00, MAIN_LAMBDA_FAC, 2.04, 2.14,
]
CRITICAL_SCAN_RUNS = 1600
CRITICAL_SCAN_MAX_T = 200.0

LOSS_SWEEP_GAMMA_VALUES = [0.0001,0.001, 0.01, 0.10, 1.00]
LOSS_SWEEP_RUNS = 800
LOSS_SWEEP_MAX_T = 200.0

ACTIVE_DECAY_LAMBDA_VALUES = [
    0.1, 1, 2.00,  5, 10, 0.05,
]
ACTIVE_DECAY_BROAD_L = 30
ACTIVE_DECAY_L_VALUES = [15,20,25,30,35,40]
ACTIVE_DECAY_FSS_LAMBDAS = [2.008]
ACTIVE_DECAY_RUNS = 150
ACTIVE_DECAY_MAX_T = 100.0
ACTIVE_DECAY_GAMMA_LOSS = LOSSLESS_GAMMA_LOSS
ACTIVE_DECAY_INITIAL_STATE = "fully_active"
ACTIVE_DECAY_INITIAL_FRACTION = 0.50

AVALANCHE_LOSS_EXPONENT_GAMMA_VALUES = [
   0.0001 ,0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
    0.05, 0.10, 0.20, 0.40,0.6,0.8, 1.00,
]
AVALANCHE_LOSS_EXPONENT_RUNS = 4000
AVALANCHE_LOSS_EXPONENT_MAX_T = 200.0

AVALANCHE_ST_MIN_T = 1.0

SIM_PARAMS = dict(
    L=30,
    lambda_fac=MAIN_LAMBDA_FAC,
    alpha=0.13,
    gamma_base=1.85,
    beta=0.1,
    gamma_loss=0.0002,
    r_cut=2.0,
    d_0=1.0,
    num_runs=1200,
    max_T=250.0,
    dt=0.09,
)

DP_AVALANCHE = {
    "(1+1)D DP":  dict(tau=1.108, alpha_T=1.159, gamma_ST=1.072, delta=0.159, eta=0.314, z=1.581),
    "(2+1)D DP":  dict(tau=1.268, alpha_T=1.450, gamma_ST=1.477, delta=0.451, eta=0.230, z=1.766),
    "(3+1)D DP":  dict(tau=1.4,  alpha_T=1.73,  gamma_ST=1.85,  delta=0.73,  eta=0.108,  z=1.89),
    "Mean-field": dict(tau=1.50,  alpha_T=2.00,  gamma_ST=2.00,  delta=1.00,  eta=0.00,  z=2.00),
}
REFERENCE_CLASS = "(3+1)D DP"



@dataclass
class FitResult:
    exponent: float = np.nan
    error: float = np.nan
    n_used: int = 0
    xmin: float = np.nan
    xmax: float = np.nan


@dataclass
class SpreadingResults:
    t: np.ndarray
    N_all: np.ndarray
    N_surv: np.ndarray
    P_surv: np.ndarray
    R2: np.ndarray
    void_fraction: np.ndarray
    n_total: int


@dataclass
class DensityDecayResults:
    t: np.ndarray
    rho: np.ndarray
    void_fraction: np.ndarray
    n_total: int
    L: int
    lambda_fac: float
    gamma_loss: float


def _num_workers(num_runs: int) -> int:
    worker_cap = mp.cpu_count() - 1 if MAX_WORKERS is None else int(MAX_WORKERS)
    return max(1, min(worker_cap, int(num_runs)))


def _run_batches(worker_fn, args_list: List[Tuple], num_cores: int):
    if num_cores <= 1 or len(args_list) <= 1:
        return [worker_fn(args) for args in args_list]

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
            return list(ex.map(worker_fn, args_list))
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"  Multiprocessing unavailable ({exc}); falling back to serial execution.")
        return [worker_fn(args) for args in args_list]


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    return value


def _cache_path(kind: str, params: Dict) -> str:
    payload = {
        "kind": kind,
        "schema": CACHE_SCHEMA_VERSION,
        "params": _jsonable(params),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return os.path.join(CACHE_DIR, f"{kind}_{digest}.npz")


def _save_npz_atomic(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _load_npz(path: str):
    if not USE_SIMULATION_CACHE or not os.path.exists(path):
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        print(f"Cache ignored because it could not be read: {path} ({exc})")
        return None



def _build_shifts(r_cut: float, d_0: float) -> List[Tuple[int, int, int, np.float32]]:
    shifts: List[Tuple[int, int, int, np.float32]] = []
    R = int(math.floor(r_cut))
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                d_sq = dx * dx + dy * dy + dz * dz
                if d_sq <= r_cut * r_cut:
                    dist = math.sqrt(d_sq)
                    shifts.append((dx, dy, dz, np.float32(math.exp(-dist / d_0))))
    return shifts


_FFT_NEIGHBOUR_THRESHOLD = 64
_KERNEL_FFT_CACHE: Dict[Tuple, np.ndarray] = {}


def _kernel_rfft(shifts: List[Tuple[int, int, int, np.float32]], L: int) -> np.ndarray:
    """Cached real-FFT of the kernel placed on an L^3 lattice (circular)."""
    key = (L, tuple((int(dx), int(dy), int(dz), float(w)) for dx, dy, dz, w in shifts))
    cached = _KERNEL_FFT_CACHE.get(key)
    if cached is None:
        K = np.zeros((L, L, L), dtype=np.float64)
        for dx, dy, dz, weight in shifts:
            K[dx % L, dy % L, dz % L] += float(weight)
        cached = np.fft.rfftn(K)
        _KERNEL_FFT_CACHE[key] = cached
    return cached


def compute_phi(active_mask: np.ndarray,
                shifts: List[Tuple[int, int, int, np.float32]],
                periodic: bool = True) -> np.ndarray:
    active_f32 = active_mask.astype(np.float32)
    Phi = np.zeros_like(active_f32, dtype=np.float32)
    L = active_mask.shape[0]

    if periodic:
        if len(shifts) >= _FFT_NEIGHBOUR_THRESHOLD:
            Khat = _kernel_rfft(shifts, L)
            Ahat = np.fft.rfftn(active_f32.astype(np.float64))
            return np.fft.irfftn(Ahat * Khat, s=(L, L, L)).astype(np.float32)
        for dx, dy, dz, weight in shifts:
            Phi += np.roll(active_f32, shift=(dx, dy, dz), axis=(0, 1, 2)) * weight
        return Phi

    for dx, dy, dz, weight in shifts:
        src_x = slice(0, L - dx) if dx > 0 else slice(-dx, L) if dx < 0 else slice(None)
        dst_x = slice(dx, L) if dx > 0 else slice(0, L + dx) if dx < 0 else slice(None)
        src_y = slice(0, L - dy) if dy > 0 else slice(-dy, L) if dy < 0 else slice(None)
        dst_y = slice(dy, L) if dy > 0 else slice(0, L + dy) if dy < 0 else slice(None)
        src_z = slice(0, L - dz) if dz > 0 else slice(-dz, L) if dz < 0 else slice(None)
        dst_z = slice(dz, L) if dz > 0 else slice(0, L + dz) if dz < 0 else slice(None)
        Phi[dst_x, dst_y, dst_z] += active_f32[src_x, src_y, src_z] * weight
    return Phi


def update_grid(grid: np.ndarray,
                shifts: List[Tuple[int, int, int, np.float32]],
                dt: float,
                lambda_fac: float,
                alpha: float,
                gamma_base: float,
                beta: float,
                gamma_loss: float,
                rng: np.random.Generator,
                periodic: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    active_mask = grid == 1
    inactive_mask = grid == 0
    Phi = compute_phi(active_mask, shifts, periodic=periodic)

    spread_rate = lambda_fac * np.expm1(alpha * Phi)
    spread_rate = np.clip(spread_rate, 0.0, None)
    p_spread = 1.0 - np.exp(-spread_rate * dt)
    spread_roll = rng.random(grid.shape, dtype=np.float32)
    spread_mask = inactive_mask & (spread_roll < p_spread)

    decay_rate = gamma_base * np.exp(beta * Phi)
    loss_rate = gamma_loss
    total_rate = decay_rate + loss_rate
    p_event = 1.0 - np.exp(-total_rate * dt)

    event_roll = rng.random(grid.shape, dtype=np.float32)
    branch_roll = rng.random(grid.shape, dtype=np.float32)

    event_mask = active_mask & (event_roll < p_event)
    p_loss_given_event = loss_rate / np.maximum(total_rate, 1e-30)
    loss_mask = event_mask & (branch_roll < p_loss_given_event)
    decay_mask = event_mask & ~loss_mask

    new_grid = grid.copy()
    new_grid[spread_mask] = 1
    new_grid[decay_mask] = 0
    new_grid[loss_mask] = 2
    return new_grid, spread_mask, loss_mask



def _run_simulation_batch(args):
    (num_runs_batch, seed, L, max_T, dt, lambda_fac, alpha, gamma_base,
     beta, gamma_loss, shifts, periodic) = args

    rng = np.random.default_rng(seed)
    center = L // 2
    sizes: List[int] = []
    durations: List[float] = []
    areas: List[int] = []
    final_void_fracs: List[float] = []

    for _ in range(num_runs_batch):
        grid = np.zeros((L, L, L), dtype=np.int8)
        visited = np.zeros((L, L, L), dtype=np.bool_)
        grid[center, center, center] = 1
        visited[center, center, center] = True
        S_size = 1
        T = 0.0

        while T < max_T:
            if not np.any(grid == 1):
                break
            grid, spread_mask, _loss_mask = update_grid(
                grid, shifts, dt, lambda_fac, alpha, gamma_base, beta,
                gamma_loss, rng, periodic=periodic
            )
            newly_visited = spread_mask & ~visited
            visited |= newly_visited
            S_size += int(np.sum(spread_mask))
            T += dt

        sizes.append(S_size)
        durations.append(T)
        areas.append(int(np.sum(visited)))
        final_void_fracs.append(float(np.mean(grid == 2)))

    return sizes, durations, areas, final_void_fracs


def _simulate_spatial_dp_multicore_uncached(L: int = 25, lambda_fac: float = 1.5, alpha: float = 0.1,
                                            gamma_base: float = 1.0, beta: float = 0.1,
                                            gamma_loss: float = 0.0, r_cut: float = 4.0,
                                            d_0: float = 1.0, num_runs: int = 1000,
                                            max_T: float = 50.0, dt: float = 0.05,
                                            periodic: bool = PERIODIC,
                                            base_seed: int = 12345):
    shifts = _build_shifts(r_cut, d_0)
    print(f"Grid: {L}^3 | Neighbours: {len(shifts)} | Runs: {num_runs} | periodic={periodic}")

    num_cores = _num_workers(num_runs)
    runs_per_core = [num_runs // num_cores] * num_cores
    runs_per_core[0] += num_runs % num_cores

    ss = np.random.SeedSequence(base_seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(runs_per_core))]

    args_list = []
    for r, seed in zip(runs_per_core, child_seeds):
        if r > 0:
            args_list.append((r, seed, L, max_T, dt, lambda_fac, alpha, gamma_base,
                              beta, gamma_loss, shifts, periodic))

    all_s, all_d, all_a, all_vf = [], [], [], []
    t0 = time.time()
    for s, d, a, vf in _run_batches(_run_simulation_batch, args_list, num_cores):
        all_s.extend(s)
        all_d.extend(d)
        all_a.extend(a)
        all_vf.extend(vf)
    print(f"Done in {time.time() - t0:.2f}s")

    return (np.array(all_s, dtype=float),
            np.array(all_d, dtype=float),
            np.array(all_a, dtype=float),
            np.array(all_vf, dtype=float))


def simulate_spatial_dp_multicore(L: int = 25, lambda_fac: float = 1.5, alpha: float = 0.1,
                                  gamma_base: float = 1.0, beta: float = 0.1,
                                  gamma_loss: float = 0.0, r_cut: float = 4.0,
                                  d_0: float = 1.0, num_runs: int = 1000,
                                  max_T: float = 50.0, dt: float = 0.05,
                                  periodic: bool = PERIODIC,
                                  base_seed: int = 12345):
    params = dict(L=L, lambda_fac=lambda_fac, alpha=alpha, gamma_base=gamma_base,
                  beta=beta, gamma_loss=gamma_loss, r_cut=r_cut, d_0=d_0,
                  max_T=max_T, dt=dt, periodic=periodic, base_seed=base_seed)
    cache_file = _cache_path("avalanche", params)

    if USE_SIMULATION_CACHE:
        cached = _load_npz(cache_file)
        if cached is not None:
            cached_runs = int(cached["num_runs"])
            if cached_runs >= num_runs:
                print(f"Cache hit: avalanche {cached_runs} runs -> using first {num_runs}")
                return (cached["sizes"][:num_runs].astype(float),
                        cached["durations"][:num_runs].astype(float),
                        cached["areas"][:num_runs].astype(float),
                        cached["final_void_fracs"][:num_runs].astype(float))

            print(f"Cache extend: avalanche {cached_runs} -> {num_runs} runs")
            extra_runs = int(num_runs - cached_runs)
            extra_seed = int(base_seed + 1009 * cached_runs + 17)
            s2, d2, a2, vf2 = _simulate_spatial_dp_multicore_uncached(
                L=L, lambda_fac=lambda_fac, alpha=alpha, gamma_base=gamma_base,
                beta=beta, gamma_loss=gamma_loss, r_cut=r_cut, d_0=d_0,
                num_runs=extra_runs, max_T=max_T, dt=dt, periodic=periodic,
                base_seed=extra_seed)
            sizes = np.concatenate([cached["sizes"], s2])
            durations = np.concatenate([cached["durations"], d2])
            areas = np.concatenate([cached["areas"], a2])
            final_void_fracs = np.concatenate([cached["final_void_fracs"], vf2])
            _save_npz_atomic(cache_file, num_runs=len(sizes), sizes=sizes,
                             durations=durations, areas=areas,
                             final_void_fracs=final_void_fracs)
            return sizes, durations, areas, final_void_fracs

    sizes, durations, areas, final_void_fracs = _simulate_spatial_dp_multicore_uncached(
        L=L, lambda_fac=lambda_fac, alpha=alpha, gamma_base=gamma_base,
        beta=beta, gamma_loss=gamma_loss, r_cut=r_cut, d_0=d_0,
        num_runs=num_runs, max_T=max_T, dt=dt, periodic=periodic,
        base_seed=base_seed)
    if USE_SIMULATION_CACHE:
        _save_npz_atomic(cache_file, num_runs=len(sizes), sizes=sizes,
                         durations=durations, areas=areas,
                         final_void_fracs=final_void_fracs)
    return sizes, durations, areas, final_void_fracs



def _run_spreading_batch(args):
    (num_runs_batch, seed, L, max_steps, dt, lambda_fac, alpha, gamma_base,
     beta, gamma_loss, shifts, periodic) = args

    rng = np.random.default_rng(seed)
    center = L // 2

    idx = np.arange(L, dtype=np.float32) - center
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    r2_grid = (xx * xx + yy * yy + zz * zz).astype(np.float32)

    N_all_sum = np.zeros(max_steps, dtype=np.float64)
    N_surv_sum = np.zeros(max_steps, dtype=np.float64)
    P_surv_cnt = np.zeros(max_steps, dtype=np.int64)
    R2_surv_sum = np.zeros(max_steps, dtype=np.float64)
    void_frac_sum = np.zeros(max_steps, dtype=np.float64)

    for _ in range(num_runs_batch):
        grid = np.zeros((L, L, L), dtype=np.int8)
        grid[center, center, center] = 1

        for step in range(max_steps):
            active_mask = grid == 1
            n_act = int(np.sum(active_mask))
            v_frac = float(np.mean(grid == 2))

            N_all_sum[step] += n_act
            void_frac_sum[step] += v_frac

            if n_act > 0:
                P_surv_cnt[step] += 1
                N_surv_sum[step] += n_act
                R2_surv_sum[step] += float(np.sum(r2_grid * active_mask)) / n_act
            else:
                void_frac_sum[step:] += v_frac
                break

            grid, _spread_mask, _loss_mask = update_grid(
                grid, shifts, dt, lambda_fac, alpha, gamma_base, beta,
                gamma_loss, rng, periodic=periodic
            )

    return N_all_sum, N_surv_sum, P_surv_cnt, R2_surv_sum, void_frac_sum, num_runs_batch


def _simulate_spreading_dynamics_uncached(L: int, lambda_fac: float, alpha: float, gamma_base: float,
                                          beta: float, gamma_loss: float, r_cut: float, d_0: float,
                                          num_runs: int = 600, max_T: float = 80.0, dt: float = 0.09,
                                          periodic: bool = PERIODIC,
                                          base_seed: int = 67890) -> SpreadingResults:
    shifts = _build_shifts(r_cut, d_0)
    max_steps = int(max_T / dt) + 1
    num_cores = _num_workers(num_runs)

    runs_per_core = [num_runs // num_cores] * num_cores
    runs_per_core[0] += num_runs % num_cores
    ss = np.random.SeedSequence(base_seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(runs_per_core))]

    args_list = []
    for r, seed in zip(runs_per_core, child_seeds):
        if r > 0:
            args_list.append((r, seed, L, max_steps, dt, lambda_fac, alpha, gamma_base,
                              beta, gamma_loss, shifts, periodic))

    print(f"\nSpreading dynamics: {num_runs} runs on {num_cores} cores | periodic={periodic}")
    t0 = time.time()

    N_all_tot = np.zeros(max_steps)
    N_surv_tot = np.zeros(max_steps)
    P_surv_tot = np.zeros(max_steps, dtype=np.int64)
    R2_tot = np.zeros(max_steps)
    vf_tot = np.zeros(max_steps)
    n_total = 0

    for Na, Ns, Pc, R2, vf, nb in _run_batches(_run_spreading_batch, args_list, num_cores):
        N_all_tot += Na
        N_surv_tot += Ns
        P_surv_tot += Pc
        R2_tot += R2
        vf_tot += vf
        n_total += nb

    t_grid = np.arange(max_steps) * dt
    P_surv = P_surv_tot / n_total
    N_all = N_all_tot / n_total
    safe = P_surv_tot > 5
    N_surv = np.where(safe, N_surv_tot / np.maximum(P_surv_tot, 1), np.nan)
    R2 = np.where(safe, R2_tot / np.maximum(P_surv_tot, 1), np.nan)
    void_fraction = vf_tot / n_total

    print(f"Spreading done in {time.time() - t0:.2f}s")
    return SpreadingResults(t_grid, N_all, N_surv, P_surv, R2, void_fraction, n_total)


def simulate_spreading_dynamics(L: int, lambda_fac: float, alpha: float, gamma_base: float,
                                beta: float, gamma_loss: float, r_cut: float, d_0: float,
                                num_runs: int = 600, max_T: float = 80.0, dt: float = 0.09,
                                periodic: bool = PERIODIC,
                                base_seed: int = 67890) -> SpreadingResults:
    params = dict(L=L, lambda_fac=lambda_fac, alpha=alpha, gamma_base=gamma_base,
                  beta=beta, gamma_loss=gamma_loss, r_cut=r_cut, d_0=d_0,
                  num_runs=num_runs, max_T=max_T, dt=dt, periodic=periodic,
                  base_seed=base_seed)
    cache_file = _cache_path("spreading", params)
    if USE_SIMULATION_CACHE:
        cached = _load_npz(cache_file)
        if cached is not None:
            print(f"Cache hit: spreading L={L}, lambda_fac={lambda_fac:g}, runs={num_runs}")
            return SpreadingResults(
                cached["t"].astype(float),
                cached["N_all"].astype(float),
                cached["N_surv"].astype(float),
                cached["P_surv"].astype(float),
                cached["R2"].astype(float),
                cached["void_fraction"].astype(float),
                int(cached["n_total"]),
            )

    res = _simulate_spreading_dynamics_uncached(
        L=L, lambda_fac=lambda_fac, alpha=alpha, gamma_base=gamma_base,
        beta=beta, gamma_loss=gamma_loss, r_cut=r_cut, d_0=d_0,
        num_runs=num_runs, max_T=max_T, dt=dt, periodic=periodic,
        base_seed=base_seed)
    if USE_SIMULATION_CACHE:
        _save_npz_atomic(cache_file, t=res.t, N_all=res.N_all, N_surv=res.N_surv,
                         P_surv=res.P_surv, R2=res.R2,
                         void_fraction=res.void_fraction, n_total=res.n_total)
    return res



def _initial_grid(L: int, rng: np.random.Generator, initial_state: str,
                  active_fraction: float = 0.50) -> np.ndarray:
    grid = np.zeros((L, L, L), dtype=np.int8)
    if initial_state == "fully_active":
        grid.fill(1)
    elif initial_state == "random_active":
        active_fraction = float(np.clip(active_fraction, 0.0, 1.0))
        grid[rng.random((L, L, L), dtype=np.float32) < active_fraction] = 1
    elif initial_state == "single_seed":
        grid[L // 2, L // 2, L // 2] = 1
    else:
        raise ValueError(f"Unknown initial_state={initial_state!r}")
    return grid


def _run_density_decay_batch(args):
    (num_runs_batch, seed, L, max_steps, dt, lambda_fac, alpha, gamma_base,
     beta, gamma_loss, shifts, periodic, initial_state, initial_fraction) = args

    rng = np.random.default_rng(seed)
    rho_sum = np.zeros(max_steps, dtype=np.float64)
    void_sum = np.zeros(max_steps, dtype=np.float64)

    for _ in range(num_runs_batch):
        grid = _initial_grid(L, rng, initial_state, initial_fraction)
        extinct = False
        final_void = 0.0

        for step in range(max_steps):
            if not extinct:
                active_mask = grid == 1
                n_active = int(np.sum(active_mask))
                final_void = float(np.mean(grid == 2))
                rho_sum[step] += n_active / grid.size
                void_sum[step] += final_void

                if n_active == 0:
                    extinct = True
                    continue

                grid, _spread_mask, _loss_mask = update_grid(
                    grid, shifts, dt, lambda_fac, alpha, gamma_base, beta,
                    gamma_loss, rng, periodic=periodic
                )
            else:
                void_sum[step] += final_void

    return rho_sum, void_sum, num_runs_batch


def _simulate_density_decay_uncached(L: int, lambda_fac: float, alpha: float, gamma_base: float,
                                     beta: float, gamma_loss: float, r_cut: float, d_0: float,
                                     num_runs: int, max_T: float, dt: float,
                                     initial_state: str = "fully_active",
                                     initial_fraction: float = 0.50,
                                     periodic: bool = PERIODIC,
                                     base_seed: int = 76000) -> DensityDecayResults:
    shifts = _build_shifts(r_cut, d_0)
    max_steps = int(max_T / dt) + 1
    num_cores = _num_workers(num_runs)

    runs_per_core = [num_runs // num_cores] * num_cores
    runs_per_core[0] += num_runs % num_cores
    ss = np.random.SeedSequence(base_seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(runs_per_core))]

    args_list = []
    for r, seed in zip(runs_per_core, child_seeds):
        if r > 0:
            args_list.append((r, seed, L, max_steps, dt, lambda_fac, alpha,
                              gamma_base, beta, gamma_loss, shifts, periodic,
                              initial_state, initial_fraction))

    print(f"  Active-density decay: L={L}, lambda_fac={lambda_fac:g}, "
          f"runs={num_runs}, gamma_loss={gamma_loss:g}")
    t0 = time.time()
    rho_tot = np.zeros(max_steps)
    void_tot = np.zeros(max_steps)
    n_total = 0

    for rho, vf, nb in _run_batches(_run_density_decay_batch, args_list, num_cores):
        rho_tot += rho
        void_tot += vf
        n_total += nb

    print(f"      done in {time.time() - t0:.2f}s")
    t_grid = np.arange(max_steps) * dt
    return DensityDecayResults(t_grid, rho_tot / n_total, void_tot / n_total,
                               n_total, L, lambda_fac, gamma_loss)


def simulate_density_decay(L: int, lambda_fac: float, alpha: float, gamma_base: float,
                           beta: float, gamma_loss: float, r_cut: float, d_0: float,
                           num_runs: int, max_T: float, dt: float,
                           initial_state: str = "fully_active",
                           initial_fraction: float = 0.50,
                           periodic: bool = PERIODIC,
                           base_seed: int = 76000) -> DensityDecayResults:
    params = dict(L=L, lambda_fac=lambda_fac, alpha=alpha, gamma_base=gamma_base,
                  beta=beta, gamma_loss=gamma_loss, r_cut=r_cut, d_0=d_0,
                  num_runs=num_runs, max_T=max_T, dt=dt,
                  initial_state=initial_state, initial_fraction=initial_fraction,
                  periodic=periodic, base_seed=base_seed)
    cache_file = _cache_path("density_decay", params)
    if USE_SIMULATION_CACHE:
        cached = _load_npz(cache_file)
        if cached is not None:
            print(f"Cache hit: density decay L={L}, lambda_fac={lambda_fac:g}, runs={num_runs}")
            return DensityDecayResults(
                cached["t"].astype(float),
                cached["rho"].astype(float),
                cached["void_fraction"].astype(float),
                int(cached["n_total"]),
                int(cached["L"]),
                float(cached["lambda_fac"]),
                float(cached["gamma_loss"]),
            )

    res = _simulate_density_decay_uncached(
        L=L, lambda_fac=lambda_fac, alpha=alpha, gamma_base=gamma_base,
        beta=beta, gamma_loss=gamma_loss, r_cut=r_cut, d_0=d_0,
        num_runs=num_runs, max_T=max_T, dt=dt, initial_state=initial_state,
        initial_fraction=initial_fraction, periodic=periodic,
        base_seed=base_seed)
    if USE_SIMULATION_CACHE:
        _save_npz_atomic(cache_file, t=res.t, rho=res.rho,
                         void_fraction=res.void_fraction, n_total=res.n_total,
                         L=res.L, lambda_fac=res.lambda_fac,
                         gamma_loss=res.gamma_loss)
    return res


def run_active_density_decay_scan(lambda_values: Iterable[float],
                                  L_values: Iterable[int],
                                  base_params: Dict,
                                  num_runs: int,
                                  max_T: float,
                                  gamma_loss: float):
    results = {}
    decay_params = {k: base_params[k] for k in
                    ["alpha", "gamma_base", "beta", "r_cut", "d_0", "dt"]}
    broad_L = int(ACTIVE_DECAY_BROAD_L)
    run_pairs = {(broad_L, float(lam)) for lam in lambda_values}
    for L_val in L_values:
        for lam in ACTIVE_DECAY_FSS_LAMBDAS:
            run_pairs.add((int(L_val), float(lam)))

    for i, (L_val, lam) in enumerate(sorted(run_pairs)):
        res = simulate_density_decay(
            L=int(L_val), lambda_fac=float(lam),
            gamma_loss=float(gamma_loss), num_runs=num_runs, max_T=max_T,
            initial_state=ACTIVE_DECAY_INITIAL_STATE,
            initial_fraction=ACTIVE_DECAY_INITIAL_FRACTION,
            periodic=PERIODIC,
            base_seed=76000 + 1000 * int(L_val) + i,
            **decay_params
        )
        results[(int(L_val), float(lam))] = res
    return results



def dp_window_time(gamma_loss: float, eps: float = LOSS_WINDOW_EPS) -> float:
    if gamma_loss <= 0:
        return np.inf
    return eps / gamma_loss


def estimate_dp_finite_size_time(L: float, z_ref: Optional[float] = None) -> float:
    z_val = DP_AVALANCHE[REFERENCE_CLASS]["z"] if z_ref is None else z_ref
    return float(L ** z_val)


def print_fss_feasibility(L_values: Iterable[int], gamma_loss: float,
                          max_T: Optional[float] = None):
    Ls = list(L_values)
    if not Ls:
        return
    Lmax = max(Ls)
    t_loss = dp_window_time(gamma_loss)
    t_fs = estimate_dp_finite_size_time(Lmax)
    print(f"  FSS feasibility at Lmax={Lmax}:")
    print(f"      expected DP finite-size time L^z ≈ {t_fs:.3g} using {REFERENCE_CLASS} z")
    if max_T is not None:
        print(f"      simulation horizon max_T = {max_T:.3g}")
        if max_T < 0.25 * t_fs:
            print("      WARNING: max_T is far below the expected finite-size cutoff time.")
    if np.isfinite(t_loss):
        print(f"      loss-small time t_loss = {t_loss:.3g}")
        if t_loss < 0.25 * t_fs:
            needed = LOSS_WINDOW_EPS / t_fs
            print("      WARNING: loss/crossover occurs far before finite-size cutoff.")
            print(f"      For loss-small FSS at L={Lmax}, gamma_loss should be << {needed:.3g}.")
    else:
        print("      loss-small time is infinite because gamma_loss=0.")


def default_spreading_fit_window(t: np.ndarray, dt: float, gamma_loss: float) -> Tuple[float, float]:
    t_loss = dp_window_time(gamma_loss)
    finite_t = t[np.isfinite(t)]
    finite_t = finite_t[finite_t > 0]
    if len(finite_t) == 0:
        return np.nan, np.nan
    lo = max(5 * dt, 0.15 * min(t_loss, finite_t.max()))
    hi = min(t_loss, 0.85 * finite_t.max())
    if not np.isfinite(hi):
        hi = 0.85 * finite_t.max()
    if hi <= lo:
        lo = finite_t[int(0.15 * len(finite_t))]
        hi = finite_t[int(0.60 * len(finite_t))]
    return float(lo), float(hi)


def configured_time_fit_window(section: str, kind: Optional[str], t: np.ndarray,
                               dt: float, gamma_loss: float) -> Tuple[float, float]:
    auto_lo, auto_hi = default_spreading_fit_window(t, dt, gamma_loss)
    cfg = FIT_CONFIG.get(section, {})
    if not isinstance(cfg, dict):
        cfg = {}
    if not cfg and section != "spreading":
        cfg = FIT_CONFIG.get("spreading", {})
    sub_cfg = cfg.get(kind, {}) if kind is not None and isinstance(cfg.get(kind, {}), dict) else {}
    lo = sub_cfg.get("t_min", cfg.get("t_min", None))
    hi = sub_cfg.get("t_max", cfg.get("t_max", None))
    return float(auto_lo if lo is None else lo), float(auto_hi if hi is None else hi)


def spreading_fit_window(kind: str, t: np.ndarray, dt: float,
                         gamma_loss: float,
                         section: str = "spreading") -> Tuple[float, float]:
    return configured_time_fit_window(section, kind, t, dt, gamma_loss)


def critical_decay_fit_window(t: np.ndarray, dt: float,
                              gamma_loss: float) -> Tuple[float, float]:
    return configured_time_fit_window("critical_decay", None, t, dt, gamma_loss)


def power_law_mle(data: np.ndarray, x_min: float, x_max: Optional[float] = None) -> FitResult:
    data = np.asarray(data, dtype=float)
    if x_max is None:
        sample = data[data >= x_min]
    else:
        sample = data[(data >= x_min) & (data <= x_max)]
    n = len(sample)
    if n < 10:
        return FitResult(n_used=n, xmin=x_min, xmax=np.nan if x_max is None else x_max)
    S = np.sum(np.log(sample / x_min))
    if S <= 0:
        return FitResult(n_used=n, xmin=x_min, xmax=np.nan if x_max is None else x_max)
    tau = 1.0 + n / S
    sigma = (tau - 1.0) / math.sqrt(n)
    return FitResult(tau, sigma, n, x_min, np.nan if x_max is None else x_max)


def find_xmin_KS(data: np.ndarray, max_trials: int = 60) -> float:
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    if len(data) < 100:
        return float(np.min(data)) if len(data) else np.nan
    unique = np.unique(data)
    if len(unique) < 10:
        return float(np.min(data))

    upper = max(10, int(len(unique) * 0.85))
    cands = unique[np.linspace(0, upper - 1, min(max_trials, upper)).astype(int)]
    best_D = np.inf
    best_xm = float(np.min(data))

    for xm in cands:
        sample = data[data >= xm]
        if len(sample) < 50:
            continue
        fr = power_law_mle(sample, xm)
        tau = fr.exponent
        if np.isnan(tau) or tau <= 1:
            continue
        ss = np.sort(sample)
        emp = np.arange(1, len(ss) + 1) / len(ss)
        theory = 1.0 - (xm / ss) ** (tau - 1.0)
        D = np.max(np.abs(emp - theory))
        if D < best_D:
            best_D = D
            best_xm = float(xm)
    return best_xm


def log_binned_pdf(data: np.ndarray, num_bins: int = 35,
                   min_count: int = 1, plot_min: float = np.nan,
                   plot_max: float = np.nan) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(data, dtype=float)
    valid = valid[valid > 0]
    if len(valid) < 2:
        return np.array([]), np.array([])
    bins = np.logspace(np.log10(valid.min()), np.log10(valid.max()), num_bins)
    raw_counts, edges = np.histogram(valid, bins=bins)
    widths = np.diff(edges)
    total = raw_counts.sum()
    if total <= 0:
        return np.array([]), np.array([])
    counts = raw_counts / (total * widths)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = raw_counts >= max(1, int(min_count))
    if np.isfinite(plot_min):
        mask &= centers >= plot_min
    if np.isfinite(plot_max):
        mask &= centers <= plot_max
    return centers[mask], counts[mask]


def log_binned_series(x: np.ndarray, y: np.ndarray, num_bins: int = 45,
                      min_count: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = (x > 0) & np.isfinite(y) & (y > 0)
    x = x[ok]
    y = y[ok]
    if len(x) < max(2, min_count):
        return np.array([]), np.array([])
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), num_bins + 1)
    xb, yb = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if int(mask.sum()) < min_count:
            continue
        xb.append(math.sqrt(lo * hi))
        yb.append(float(np.mean(y[mask])))
    return np.asarray(xb), np.asarray(yb)


def empirical_survival_curve(data: np.ndarray, num_points: int = 90,
                             min_survivors: int = 8, plot_min: float = np.nan,
                             plot_max: float = np.nan) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(data, dtype=float)
    valid = np.sort(valid[valid > 0])
    if len(valid) < max(2, min_survivors):
        return np.array([]), np.array([])
    lo = max(valid[0], float(plot_min)) if np.isfinite(plot_min) else valid[0]
    hi = min(valid[-1], float(plot_max)) if np.isfinite(plot_max) else valid[-1]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), np.array([])
    grid = np.geomspace(lo, hi, max(2, int(num_points)))
    survivor_counts = len(valid) - np.searchsorted(valid, grid, side="left")
    mask = survivor_counts >= max(1, int(min_survivors))
    if not np.any(mask):
        return np.array([]), np.array([])
    return grid[mask], survivor_counts[mask] / len(valid)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray,
                     x_min: Optional[float] = None,
                     x_max: Optional[float] = None) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    if np.sum(mask) < 3:
        return np.nan, np.nan
    lx = np.log10(x[mask])
    ly = np.log10(y[mask])
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    err = math.sqrt(cov[0, 0]) if cov.size else np.nan
    return float(coeffs[0]), float(err)


def effective_exponent(x: np.ndarray, y: np.ndarray, sign: float = 1.0,
                       smooth: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return np.array([]), np.array([])
    lx = np.log(x)
    ly = np.log(y)
    deriv = np.gradient(ly, lx)
    eff = sign * deriv
    if smooth and smooth > 2 and len(eff) > smooth:
        kernel = np.ones(smooth) / smooth
        eff = np.convolve(eff, kernel, mode="same")
    return x, eff


def avalanche_size_vs_duration(durations: np.ndarray, sizes: np.ndarray, num_bins: int = 18,
                               min_count: int = 5):
    durations = np.asarray(durations, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    ok = (durations > 0) & (sizes > 0)
    d = durations[ok]
    s = sizes[ok]
    if len(d) < 20:
        return None
    Tmin, Tmax = d.min(), d.max()
    if Tmin >= Tmax:
        return None
    edges = np.logspace(np.log10(Tmin), np.log10(Tmax), num_bins + 1)
    centers, mean_S, sem_S = [], [], []
    for i in range(num_bins):
        mask = (d >= edges[i]) & (d < edges[i + 1])
        n = int(mask.sum())
        if n < min_count:
            continue
        centers.append(math.sqrt(edges[i] * edges[i + 1]))
        mean_S.append(float(np.mean(s[mask])))
        sem_S.append(float(np.std(s[mask]) / math.sqrt(n)))
    if len(centers) < 4:
        return None
    return np.array(centers), np.array(mean_S), np.array(sem_S)


def choose_distribution_fit(data: np.ndarray, cfg: Dict) -> FitResult:
    valid = np.asarray(data, dtype=float)
    valid = valid[valid > 0]
    if len(valid) == 0:
        return FitResult()
    x_min = cfg.get("x_min")
    x_max = cfg.get("x_max")
    if cfg.get("auto_xmin", True) and x_min is None:
        x_min = find_xmin_KS(valid)
    if x_min is None:
        x_min = float(np.min(valid))
    return power_law_mle(valid, x_min, x_max)


def density_decay_metrics(res: DensityDecayResults, dt: float) -> Dict[str, float]:
    t_min, t_max = critical_decay_fit_window(res.t, dt, res.gamma_loss)
    ok = (res.t > 0) & np.isfinite(res.rho) & (res.rho > 0)
    slope, slope_err = fit_loglog_slope(res.t[ok], res.rho[ok], t_min, t_max)
    decay_exp = -slope if not np.isnan(slope) else np.nan

    curvature_rms = np.nan
    fit_mask = ok & (res.t >= t_min) & (res.t <= t_max)
    if np.sum(fit_mask) >= 5 and not np.isnan(slope):
        lx = np.log(res.t[fit_mask])
        ly = np.log(res.rho[fit_mask])
        coeff = np.polyfit(lx, ly, 1)
        resid = ly - np.polyval(coeff, lx)
        curvature_rms = float(np.sqrt(np.mean(resid * resid)))

    return {
        "t_min": t_min,
        "t_max": t_max,
        "decay_exp": decay_exp,
        "decay_exp_err": slope_err,
        "curvature_rms": curvature_rms,
        "final_rho": float(res.rho[np.isfinite(res.rho)][-1]),
    }


def estimate_size_duration_exponent(durations: np.ndarray, sizes: np.ndarray,
                                    gamma_loss: float) -> Tuple[float, float]:
    t_loss = dp_window_time(gamma_loss)
    lower_T = max(AVALANCHE_ST_MIN_T, 5 * SIM_PARAMS["dt"])
    mask = np.asarray(durations, dtype=float) >= lower_T
    if np.isfinite(t_loss):
        mask &= np.asarray(durations, dtype=float) <= t_loss
    res = avalanche_size_vs_duration(np.asarray(durations)[mask],
                                     np.asarray(sizes)[mask])
    if res is None:
        return np.nan, np.nan
    centers, mean_S, _sem_S = res
    T_min = FIT_CONFIG["ST"].get("T_min") or max(lower_T, float(np.quantile(centers, 0.20)))
    T_max = FIT_CONFIG["ST"].get("T_max")
    if T_max is None:
        T_max = float(np.quantile(centers, 0.90))
        if np.isfinite(t_loss):
            T_max = min(T_max, t_loss)
    return fit_loglog_slope(centers, mean_S, T_min, T_max)


def estimate_size_duration_exponent_fss(durations: np.ndarray, sizes: np.ndarray,
                                        frac_hi: float = 0.6) -> Tuple[float, float]:
 
    d = np.asarray(durations, dtype=float)
    s = np.asarray(sizes, dtype=float)
    ok = (d > 0) & (s > 0)
    d, s = d[ok], s[ok]
    if len(d) < 50:
        return np.nan, np.nan
    lower_T = max(AVALANCHE_ST_MIN_T, 5 * SIM_PARAMS["dt"])
    T_min = FIT_CONFIG["ST"].get("T_min") or lower_T
    T_cut = float(np.mean(d ** 2) / np.mean(d))
    T_max = max(T_min * 3.0, frac_hi * T_cut)
    res = avalanche_size_vs_duration(d, s, num_bins=22)
    if res is None:
        return np.nan, np.nan
    centers, mean_S, _sem_S = res
    return fit_loglog_slope(centers, mean_S, T_min, T_max)


def fss_exponent_records(fss_results: Dict[int, Dict]) -> Tuple[Dict[int, Dict], float]:
  
    pooled_d = []
    for rec in fss_results.values():
        d = np.asarray(rec.get("durations_raw", rec["durations"]), dtype=float)
        gl = rec.get("gamma_loss", 0.0)
        tl = dp_window_time(gl)
        d = d[(d > 0) & np.isfinite(d)]
        if np.isfinite(tl):
            d = d[d <= tl]
        if len(d):
            pooled_d.append(d)
    pooled_d = np.concatenate(pooled_d) if pooled_d else np.array([])
    dur_xmin = find_xmin_KS(pooled_d) if len(pooled_d) else np.nan

    records: Dict[int, Dict] = {}
    for L_val, rec in sorted(fss_results.items()):
        sizes = np.asarray(rec.get("sizes_raw", rec["sizes"]), dtype=float)
        durations = np.asarray(rec.get("durations_raw", rec["durations"]), dtype=float)
        areas = np.asarray(rec.get("areas_raw", rec.get("sizes_raw", rec["sizes"])),
                           dtype=float)
        gl = rec.get("gamma_loss", 0.0)
        tl = dp_window_time(gl)
        mask = durations > 0
        if np.isfinite(tl):
            mask &= durations <= tl
        s, d, a = sizes[mask], durations[mask], areas[mask]

        fit_S = choose_distribution_fit(s, FIT_CONFIG["S"])
        fit_A = choose_distribution_fit(a, FIT_CONFIG["A"])
        d_pos = d[d > 0]
        xmin_T = dur_xmin if np.isfinite(dur_xmin) else (
            float(np.min(d_pos)) if len(d_pos) else np.nan)
        fit_T = power_law_mle(d, xmin_T, None) if len(d_pos) else FitResult()
        gamma_ST, gamma_ST_err = estimate_size_duration_exponent_fss(d, s)
        records[int(L_val)] = {
            "tau_S": fit_S.exponent, "tau_S_err": fit_S.error,
            "alpha_T": fit_T.exponent, "alpha_T_err": fit_T.error,
            "tau_A": fit_A.exponent, "tau_A_err": fit_A.error,
            "gamma_ST": gamma_ST, "gamma_ST_err": gamma_ST_err,
        }
    return records, float(dur_xmin)


def avalanche_exponent_record(sizes: np.ndarray, durations: np.ndarray,
                              areas: np.ndarray, gamma_loss: float) -> Dict[str, float]:
    t_loss = dp_window_time(gamma_loss)
    durations = np.asarray(durations, dtype=float)
    mask = durations > 0
    if np.isfinite(t_loss):
        mask &= durations <= t_loss

    s = np.asarray(sizes, dtype=float)[mask]
    d = durations[mask]
    a = np.asarray(areas, dtype=float)[mask]
    fit_S = choose_distribution_fit(s, FIT_CONFIG["S"])
    fit_T = choose_distribution_fit(d, FIT_CONFIG["T"])
    fit_A = choose_distribution_fit(a, FIT_CONFIG["A"])
    gamma_ST, gamma_ST_err = estimate_size_duration_exponent(d, s, gamma_loss)

    return {
        "tau_S": fit_S.exponent,
        "tau_S_err": fit_S.error,
        "tau_S_n": fit_S.n_used,
        "tau_S_xmin": fit_S.xmin,
        "alpha_T": fit_T.exponent,
        "alpha_T_err": fit_T.error,
        "alpha_T_n": fit_T.n_used,
        "alpha_T_xmin": fit_T.xmin,
        "tau_A": fit_A.exponent,
        "tau_A_err": fit_A.error,
        "tau_A_n": fit_A.n_used,
        "tau_A_xmin": fit_A.xmin,
        "gamma_ST": gamma_ST,
        "gamma_ST_err": gamma_ST_err,
        "n_avalanches": int(len(d)),
        "t_loss": t_loss,
    }


def _replicate_sem(records: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = np.array([rec.get(key, np.nan) for rec in records], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    mean_val = float(np.mean(vals))
    sem_val = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else np.nan
    return mean_val, sem_val


def combined_uncertainty(record: Dict[str, float], key: str, err_key: str) -> float:
    fit_err = float(record.get(err_key, np.nan))
    rep_sem = float(record.get(f"{key}_rep_sem", np.nan))
    if np.isfinite(fit_err) and np.isfinite(rep_sem):
        return float(math.sqrt(fit_err * fit_err + rep_sem * rep_sem))
    if np.isfinite(rep_sem):
        return rep_sem
    return fit_err


def linear_extrapolate_invL(Ls: np.ndarray, ys: np.ndarray,
                            yerr: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
    Ls = np.asarray(Ls, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(Ls) & (Ls > 0) & np.isfinite(ys)
    weights = None
    if yerr is not None:
        err = np.asarray(yerr, dtype=float)
        mask &= np.isfinite(err) & (err > 0)
        weights = 1.0 / err[mask]
    if np.sum(mask) < 2:
        return np.nan, np.nan, np.nan, np.nan
    x = 1.0 / Ls[mask]
    y = ys[mask]
    if len(x) >= 3:
        coeff, cov = np.polyfit(x, y, 1, w=weights, cov=True)
    else:
        coeff = np.polyfit(x, y, 1, w=weights)
        cov = np.full((2, 2), np.nan)
    slope, intercept = float(coeff[0]), float(coeff[1])
    intercept_err = float(math.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan
    return intercept, intercept_err, slope, float(np.nanstd(y - np.polyval(coeff, x)))


def _spreading_observables_for_scan(sp: SpreadingResults, gamma_loss: float,
                                    dt: float) -> Dict[str, float]:
    p_t_min, p_t_max = spreading_fit_window("P", sp.t, dt, gamma_loss,
                                            section="critical_scan")
    n_t_min, n_t_max = spreading_fit_window("N", sp.t, dt, gamma_loss,
                                            section="critical_scan")

    ok = (sp.t > 0) & (sp.P_surv > 0)
    slope, slope_err = fit_loglog_slope(sp.t[ok], sp.P_surv[ok], p_t_min, p_t_max)
    delta = -slope if not np.isnan(slope) else np.nan

    okN = (sp.t > 0) & np.isfinite(sp.N_all) & (sp.N_all > 0)
    eta, eta_err = fit_loglog_slope(sp.t[okN], sp.N_all[okN], n_t_min, n_t_max)

    t_delta, delta_eff = effective_exponent(sp.t, sp.P_surv, sign=-1.0, smooth=11)
    plateau = (t_delta >= p_t_min) & (t_delta <= p_t_max) & np.isfinite(delta_eff)
    delta_eff_mean = float(np.nanmean(delta_eff[plateau])) if np.any(plateau) else np.nan
    delta_eff_std = float(np.nanstd(delta_eff[plateau])) if np.any(plateau) else np.nan

    curvature_rms = np.nan
    fit_mask = ok & (sp.t >= p_t_min) & (sp.t <= p_t_max)
    if np.sum(fit_mask) >= 5 and not np.isnan(slope):
        lx = np.log(sp.t[fit_mask])
        ly = np.log(sp.P_surv[fit_mask])
        coeff = np.polyfit(lx, ly, 1)
        resid = ly - np.polyval(coeff, lx)
        curvature_rms = float(np.sqrt(np.mean(resid * resid)))

    return {
        "t_min": p_t_min, "t_max": p_t_max,
        "p_t_min": p_t_min, "p_t_max": p_t_max,
        "n_t_min": n_t_min, "n_t_max": n_t_max,
        "delta": delta, "delta_err": slope_err,
        "eta": eta, "eta_err": eta_err,
        "delta_eff_mean": delta_eff_mean, "delta_eff_std": delta_eff_std,
        "curvature_rms": curvature_rms,
        "final_P_surv": float(sp.P_surv[np.isfinite(sp.P_surv)][-1]),
    }


def run_critical_scan(lambda_values: Iterable[float], base_params: Dict,
                      num_runs: int, max_T: float):
    results = {}
    scan_params = {k: base_params[k] for k in
                   ["L", "alpha", "gamma_base", "beta", "gamma_loss", "r_cut", "d_0"]}
    for i, lam in enumerate(lambda_values):
        print(f"  Critical scan: lambda_fac={lam:g} ...")
        sp = simulate_spreading_dynamics(**scan_params,
                                         lambda_fac=float(lam),
                                         num_runs=num_runs,
                                         max_T=max_T,
                                         dt=base_params["dt"],
                                         periodic=PERIODIC,
                                         base_seed=22000 + i)
        metrics = _spreading_observables_for_scan(sp, base_params["gamma_loss"],
                                                  base_params["dt"])
        results[float(lam)] = {"spreading": sp, "metrics": metrics}
        print(f"      delta={metrics['delta']:.3f}, eta={metrics['eta']:.3f}, "
              f"curvature={metrics['curvature_rms']:.3g}, "
              f"plateau_std={metrics['delta_eff_std']:.3g}")
    return results


def run_loss_sweep(gamma_values: Iterable[float], base_params: Dict,
                   num_runs: int, max_T: float):
    results = {}
    sweep_params = {k: base_params[k] for k in
                    ["L", "lambda_fac", "alpha", "gamma_base", "beta", "r_cut", "d_0"]}
    for i, gl in enumerate(gamma_values):
        print(f"  Loss sweep: gamma_loss={gl:g} ...")
        sp = simulate_spreading_dynamics(**sweep_params,
                                         gamma_loss=float(gl),
                                         num_runs=num_runs,
                                         max_T=max_T,
                                         dt=base_params["dt"],
                                         periodic=PERIODIC,
                                         base_seed=33000 + i)
        metrics = _spreading_observables_for_scan(sp, float(gl), base_params["dt"])
        results[float(gl)] = {"spreading": sp, "metrics": metrics}
        print(f"      delta={metrics['delta']:.3f}, eta={metrics['eta']:.3f}, "
              f"fit=[{metrics['t_min']:.3g}, {metrics['t_max']:.3g}]")
    return results


def estimate_cutoff_scale(data: np.ndarray, method: str = FSS_CUTOFF_METHOD,
                          quantile: float = FSS_CUTOFF_QUANTILE) -> float:
    sample = np.asarray(data, dtype=float)
    sample = sample[(sample > 0) & np.isfinite(sample)]
    if len(sample) < 10:
        return np.nan
    if method == "moment":
        mean = np.mean(sample)
        return float(np.mean(sample ** 2) / mean) if mean > 0 else np.nan
    if method == "quantile":
        return float(np.quantile(sample, quantile))
    raise ValueError(f"Unknown FSS_CUTOFF_METHOD={method!r}")


def bootstrap_cutoff_error(data: np.ndarray, seed: int,
                           n_boot: int = FSS_BOOTSTRAP_SAMPLES) -> float:
    sample = np.asarray(data, dtype=float)
    sample = sample[(sample > 0) & np.isfinite(sample)]
    if len(sample) < 20 or n_boot <= 1:
        return np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = sample[rng.integers(0, len(sample), len(sample))]
        vals[i] = estimate_cutoff_scale(draw)
    return float(np.nanstd(vals, ddof=1))


def run_finite_size_scaling(L_values: Iterable[int], base_params: Dict, num_runs_each: int):
    results = {}
    for L_val in L_values:
        params = {**base_params, "L": L_val, "num_runs": num_runs_each}
        print(f"  FSS: L={L_val} ...")
        s, d, a, _vf = simulate_spatial_dp_multicore(**params, periodic=PERIODIC,
                                                     base_seed=1111 + int(L_val))
        min_duration = SIM_PARAMS["dt"] if FSS_MIN_DURATION is None else FSS_MIN_DURATION
        s_filt = s[s >= FSS_MIN_SIZE]
        d_filt = d[d >= min_duration]
        S_cut = estimate_cutoff_scale(s_filt)
        T_cut = estimate_cutoff_scale(d_filt)
        S_cut_err = bootstrap_cutoff_error(s_filt, seed=44000 + int(L_val))
        T_cut_err = bootstrap_cutoff_error(d_filt, seed=55000 + int(L_val))
        results[L_val] = {
            "S_cut": S_cut, "T_cut": T_cut,
            "S_cut_err": S_cut_err, "T_cut_err": T_cut_err,
            "gamma_loss": params["gamma_loss"],
            "lambda_fac": params["lambda_fac"],
            "max_T": params["max_T"],
            "nS": len(s_filt), "nT": len(d_filt),
            "sizes": s_filt, "durations": d_filt,
            "sizes_raw": s, "durations_raw": d, "areas_raw": a,
        }
        print(f"      S_cut={S_cut:.3g}±{S_cut_err:.2g}, "
              f"T_cut={T_cut:.3g}±{T_cut_err:.2g}, n={len(s_filt)}")
    return results


def run_avalanche_loss_exponent_sweep(gamma_values: Iterable[float], base_params: Dict,
                                      num_runs: int, max_T: float,
                                      precomputed: Optional[Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None):
    results = {}
    precomputed = precomputed or {}
    sweep_params = {k: base_params[k] for k in
                    ["L", "lambda_fac", "alpha", "gamma_base", "beta",
                     "r_cut", "d_0", "dt"]}
    for i, gl in enumerate(gamma_values):
        gl = float(gl)
        if gl in precomputed and 1 <= 1:
            print(f"  Avalanche loss-exponent sweep: gamma_loss={gl:g} (using main samples)")
            sizes, durations, areas = precomputed[gl]
            record = avalanche_exponent_record(sizes, durations, areas, gl)
            record["replicates"] = 1
            record["runs_per_replicate"] = len(durations)
            record["n_runs_total"] = len(durations)
        else:
            print(f"  Avalanche loss-exponent sweep: gamma_loss={gl:g} "
                  f"({1} x {num_runs} runs) ...")
            size_parts, duration_parts, area_parts = [], [], []
            replicate_records = []
            for rep in range(1):
                sizes, durations, areas, _vf = simulate_spatial_dp_multicore(
                    **sweep_params, gamma_loss=gl, num_runs=num_runs, max_T=max_T,
                    periodic=PERIODIC, base_seed=88000 + 1000 * i + rep)
                size_parts.append(sizes)
                duration_parts.append(durations)
                area_parts.append(areas)
                replicate_records.append(avalanche_exponent_record(sizes, durations, areas, gl))

            sizes_all = np.concatenate(size_parts)
            durations_all = np.concatenate(duration_parts)
            areas_all = np.concatenate(area_parts)
            record = avalanche_exponent_record(sizes_all, durations_all, areas_all, gl)
            record["replicates"] = 1
            record["runs_per_replicate"] = num_runs
            record["n_runs_total"] = int(len(durations_all))
            for key in ["tau_S", "alpha_T", "tau_A", "gamma_ST"]:
                mean_val, sem_val = _replicate_sem(replicate_records, key)
                record[f"{key}_rep_mean"] = mean_val
                record[f"{key}_rep_sem"] = sem_val
        results[gl] = record
    return results



def measure_avalanche_exponents(sizes: np.ndarray, durations: np.ndarray,
                                areas: np.ndarray, gamma_loss: float) -> Dict:
    """Size, duration and area exponents, for the full record and for the
    small-loss window T <= t_loss."""
    t_loss = dp_window_time(gamma_loss)
    dp_mask = durations <= t_loss if np.isfinite(t_loss) else np.ones_like(durations, dtype=bool)
    return {
        "tau_S_all":        choose_distribution_fit(sizes, FIT_CONFIG["S"]),
        "alpha_T_all":      choose_distribution_fit(durations, FIT_CONFIG["T"]),
        "tau_A_all":        choose_distribution_fit(areas, FIT_CONFIG["A"]),
        "tau_S_DPwindow":   choose_distribution_fit(sizes[dp_mask], FIT_CONFIG["S"]),
        "alpha_T_DPwindow": choose_distribution_fit(durations[dp_mask], FIT_CONFIG["T"]),
        "tau_A_DPwindow":   choose_distribution_fit(areas[dp_mask], FIT_CONFIG["A"]),
        "dp_window_mask":   dp_mask,
    }


def measure_size_duration_exponent(durations: np.ndarray, sizes: np.ndarray,
                                   gamma_loss: float) -> Tuple[float, float]:
    """gamma_ST from the binned mean size at fixed duration, <S>(T) ~ T^gamma_ST."""
    t_loss = dp_window_time(gamma_loss)
    lower_T = max(AVALANCHE_ST_MIN_T, 5 * SIM_PARAMS["dt"])
    dp_mask = durations <= t_loss if np.isfinite(t_loss) else np.ones_like(durations, dtype=bool)
    dp_mask &= durations >= lower_T
    res = avalanche_size_vs_duration(durations[dp_mask], sizes[dp_mask],
                                     num_bins=FIT_CONFIG["ST"].get("num_bins", 18),
                                     min_count=FIT_CONFIG["ST"].get("min_count", 5))
    if res is None:
        return np.nan, np.nan
    centers, mean_S, _ = res
    T_min = FIT_CONFIG["ST"].get("T_min") or max(lower_T, float(np.quantile(centers, 0.20)))
    T_max = FIT_CONFIG["ST"].get("T_max") or min(float(np.quantile(centers, 0.90)), t_loss)
    return fit_loglog_slope(centers, mean_S, T_min, T_max)


def measure_spreading_exponents(sp: SpreadingResults, gamma_loss: float,
                                dt: float) -> Dict[str, float]:
    """delta from P_surv(t), eta from N(t) and z from R^2(t), each fitted within
    its own window inside the small-loss interval."""
    p_t_min, p_t_max = spreading_fit_window("P", sp.t, dt, gamma_loss)
    n_t_min, n_t_max = spreading_fit_window("N", sp.t, dt, gamma_loss)
    r2_t_min, r2_t_max = spreading_fit_window("R2", sp.t, dt, gamma_loss)

    ok = (sp.t > 0) & (sp.P_surv > 0)
    slope, slope_err = fit_loglog_slope(sp.t[ok], sp.P_surv[ok], p_t_min, p_t_max)
    delta = -slope if not np.isnan(slope) else np.nan
    delta_err = slope_err

    ok = (sp.t > 0) & np.isfinite(sp.N_all) & (sp.N_all > 0)
    eta, eta_err = fit_loglog_slope(sp.t[ok], sp.N_all[ok], n_t_min, n_t_max)

    ok = (sp.t > 0) & np.isfinite(sp.R2) & (sp.R2 > 0)
    two_over_z, two_over_z_err = fit_loglog_slope(sp.t[ok], sp.R2[ok], r2_t_min, r2_t_max)
    z = 2.0 / two_over_z if not np.isnan(two_over_z) and two_over_z > 0 else np.nan
    z_err = 2.0 * two_over_z_err / (two_over_z * two_over_z) if not np.isnan(z) else np.nan

    return dict(delta=delta, delta_err=delta_err, eta=eta, eta_err=eta_err,
                z=z, z_err=z_err, two_over_z=two_over_z,
                p_t_min=p_t_min, p_t_max=p_t_max,
                n_t_min=n_t_min, n_t_max=n_t_max,
                r2_t_min=r2_t_min, r2_t_max=r2_t_max,
                t_loss=dp_window_time(gamma_loss))


def measure_cutoff_exponents(fss_results: Dict[int, Dict]) -> Tuple[float, float, float, float]:
    """Space-time fractal dimension D from S_cut ~ L^D and the dynamic exponent
    z from T_cut ~ L^z."""
    if not fss_results:
        return np.nan, np.nan, np.nan, np.nan
    Ls = np.array(sorted(fss_results.keys()), dtype=float)
    Sc = np.array([fss_results[int(L)]["S_cut"] for L in Ls])
    Tc = np.array([fss_results[int(L)]["T_cut"] for L in Ls])
    fss_cfg = FIT_CONFIG.get("FSS_cutoff", FIT_CONFIG.get("FSS", {}))
    fit_L_mask = np.ones_like(Ls, dtype=bool)
    if fss_cfg.get("L_min") is not None:
        fit_L_mask &= Ls >= fss_cfg["L_min"]
    if fss_cfg.get("L_max") is not None:
        fit_L_mask &= Ls <= fss_cfg["L_max"]
    fitS = np.isfinite(Sc) & (Sc > 0) & fit_L_mask
    fitT = np.isfinite(Tc) & (Tc > 0) & fit_L_mask
    D, D_err = fit_loglog_slope(Ls[fitS], Sc[fitS]) if fitS.sum() >= 3 else (np.nan, np.nan)
    zF, zF_err = fit_loglog_slope(Ls[fitT], Tc[fitT]) if fitT.sum() >= 3 else (np.nan, np.nan)
    return D, D_err, zF, zF_err


def measure_critical_lambda(scan_results: Dict[float, Dict]) -> float:
    """Coupling at which the spreading observables are flattest and closest to
    the reference exponents; the numerical estimate of lambda_c."""
    ref = DP_AVALANCHE[REFERENCE_CLASS]
    lambdas, curvatures, plateau_stds, deltas, etas = [], [], [], [], []
    for lam, rec in sorted(scan_results.items()):
        met = rec["metrics"]
        lambdas.append(lam)
        curvatures.append(met["curvature_rms"])
        plateau_stds.append(met["delta_eff_std"])
        deltas.append(met["delta"])
        etas.append(met["eta"])
    score = (np.array(curvatures, dtype=float) + np.array(plateau_stds, dtype=float)
             + np.abs(np.array(deltas, dtype=float) - ref["delta"])
             + np.abs(np.array(etas, dtype=float) - ref["eta"]))
    valid = np.isfinite(score)
    if not np.any(valid):
        return np.nan
    return float(np.array(lambdas)[valid][np.argmin(score[valid])])


def extrapolate_fss_exponents(fss_results: Dict[int, Dict]) -> Dict[str, Tuple[float, float]]:
    """Avalanche exponents of each lattice, extrapolated to 1/L -> 0."""
    records, _ = fss_exponent_records(fss_results)
    if not records:
        return {}
    Ls = np.array(sorted(records.keys()), dtype=float)
    out: Dict[str, Tuple[float, float]] = {}
    for key in ("tau_S", "alpha_T", "tau_A", "gamma_ST"):
        ys = np.array([records[int(L)][key] for L in Ls], dtype=float)
        es = np.array([records[int(L)].get(key + "_err", np.nan) for L in Ls], dtype=float)
        ok = np.isfinite(ys)
        if ok.sum() >= 3:
            value, error, _, _ = linear_extrapolate_invL(Ls[ok], ys[ok], es[ok])
        else:
            value, error = np.nan, np.nan
        out[key] = (value, error)
    return out



def _row(name: str, value: float, error: float, reference: float) -> str:
    ref = "  --  " if reference is None or not np.isfinite(reference) else f"{reference:6.3f}"
    return f"  {name:10s} {value:8.3f} +/- {error:6.3f}    {ref}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ref = DP_AVALANCHE[REFERENCE_CLASS]
    print(f"gamma_loss = {SIM_PARAMS['gamma_loss']:g}, small-loss window ends at "
          f"t_loss = {dp_window_time(SIM_PARAMS['gamma_loss']):.4g}")

    print("\nAvalanche statistics")
    sizes, durations, areas, _ = simulate_spatial_dp_multicore(**SIM_PARAMS,
                                                               periodic=PERIODIC)
    fits = measure_avalanche_exponents(sizes, durations, areas,
                                       SIM_PARAMS["gamma_loss"])
    gamma_ST, gamma_ST_err = measure_size_duration_exponent(
        durations, sizes, SIM_PARAMS["gamma_loss"])
    print(_row("tau_S", fits["tau_S_DPwindow"].exponent, fits["tau_S_DPwindow"].error, ref["tau"]))
    print(_row("alpha_T", fits["alpha_T_DPwindow"].exponent, fits["alpha_T_DPwindow"].error, ref["alpha_T"]))
    print(_row("tau_A", fits["tau_A_DPwindow"].exponent, fits["tau_A_DPwindow"].error, None))
    print(_row("gamma_ST", gamma_ST, gamma_ST_err, ref["gamma_ST"]))

    print("\nSpreading exponents")
    spread_params = {k: SIM_PARAMS[k] for k in
                     ["L", "lambda_fac", "alpha", "gamma_base", "beta",
                      "gamma_loss", "r_cut", "d_0"]}
    sp = simulate_spreading_dynamics(**spread_params,
                                     num_runs=SPREADING_NUM_RUNS,
                                     max_T=SPREADING_MAX_T,
                                     dt=SIM_PARAMS["dt"],
                                     periodic=PERIODIC)
    spreading = measure_spreading_exponents(sp, SIM_PARAMS["gamma_loss"],
                                            SIM_PARAMS["dt"])
    print(_row("delta", spreading["delta"], spreading["delta_err"], ref["delta"]))
    print(_row("eta", spreading["eta"], spreading["eta_err"], ref["eta"]))
    print(_row("z", spreading["z"], spreading["z_err"], ref["z"]))

    if RUN_CRITICAL_SCAN and CRITICAL_SCAN_LAMBDA_VALUES:
        print("\nCoupling scan")
        scan = run_critical_scan(CRITICAL_SCAN_LAMBDA_VALUES, SIM_PARAMS,
                                 CRITICAL_SCAN_RUNS, CRITICAL_SCAN_MAX_T)
        print(f"  flattest coupling of the scan: lambda = "
              f"{measure_critical_lambda(scan):.4g}")

    if RUN_LOSS_SWEEP and LOSS_SWEEP_GAMMA_VALUES:
        print("\nRobustness of the spreading exponents across the loss sweep")
        run_loss_sweep(LOSS_SWEEP_GAMMA_VALUES, SIM_PARAMS,
                       LOSS_SWEEP_RUNS, LOSS_SWEEP_MAX_T)

    if RUN_AVALANCHE_LOSS_EXPONENT_SWEEP and AVALANCHE_LOSS_EXPONENT_GAMMA_VALUES:
        print("\nAvalanche exponents across the loss sweep")
        loss_exponents = run_avalanche_loss_exponent_sweep(
            AVALANCHE_LOSS_EXPONENT_GAMMA_VALUES, SIM_PARAMS,
            AVALANCHE_LOSS_EXPONENT_RUNS, AVALANCHE_LOSS_EXPONENT_MAX_T,
            precomputed={float(SIM_PARAMS["gamma_loss"]): (sizes, durations, areas)})
        for gl in sorted(loss_exponents):
            rec = loss_exponents[gl]
            print(f"  gamma_loss={gl:<8g} tau_S={rec['tau_S']:.3f} "
                  f"alpha_T={rec['alpha_T']:.3f} gamma_ST={rec['gamma_ST']:.3f}")

    if RUN_ACTIVE_DENSITY_DECAY and ACTIVE_DECAY_LAMBDA_VALUES:
        print("\nActive-density decay from a fully active state")
        decay = run_active_density_decay_scan(
            ACTIVE_DECAY_LAMBDA_VALUES, ACTIVE_DECAY_L_VALUES, SIM_PARAMS,
            ACTIVE_DECAY_RUNS, ACTIVE_DECAY_MAX_T, ACTIVE_DECAY_GAMMA_LOSS)
        for (L_val, lam) in sorted(decay):
            met = density_decay_metrics(decay[(L_val, lam)], SIM_PARAMS["dt"])
            print(f"  L={L_val:<3d} lambda={lam:<6g} decay exponent "
                  f"{met['decay_exp']:.3f} +/- {met['decay_exp_err']:.3f}")

    if FSS_L_VALUES:
        print("\nFinite-size scaling")
        fss_params = {k: SIM_PARAMS[k] for k in
                      ["lambda_fac", "alpha", "gamma_base", "beta",
                       "gamma_loss", "r_cut", "d_0", "max_T", "dt"]}
        fss_params["gamma_loss"] = FSS_GAMMA_LOSS
        fss_params["max_T"] = FSS_MAX_T
        if FSS_LAMBDA_FAC is not None:
            fss_params["lambda_fac"] = FSS_LAMBDA_FAC
        print_fss_feasibility(FSS_L_VALUES, fss_params["gamma_loss"], fss_params["max_T"])
        fss_results = run_finite_size_scaling(FSS_L_VALUES, fss_params, FSS_RUNS_EACH)
        D, D_err, zF, zF_err = measure_cutoff_exponents(fss_results)
        print(_row("D", D, D_err, None))
        print(_row("z_FSS", zF, zF_err, ref["z"]))
        if RUN_FSS_EXPONENT_EXTRAPOLATION:
            print("\nExponents extrapolated to 1/L -> 0")
            extrap = extrapolate_fss_exponents(fss_results)
            for key, refkey in (("tau_S", "tau"), ("alpha_T", "alpha_T"),
                                ("tau_A", None), ("gamma_ST", "gamma_ST")):
                value, error = extrap.get(key, (np.nan, np.nan))
                print(_row(key, value, error, ref[refkey] if refkey else None))


if __name__ == "__main__":
    mp.freeze_support()
    main()
