"""
Self-contained core dynamics for the new simulations (avalanche temporal shape
and conjugate-field equation of state).  
"""
from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple

PARAMS = dict(L=30, lambda_fac=2.008, alpha=0.13, gamma_base=1.85, beta=0.1,
              gamma_loss=0.0002, r_cut=2.0, d_0=1.0, dt=0.09)

def build_shifts(r_cut: float, d_0: float):
    shifts = []
    R = int(math.floor(r_cut))
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                d_sq = dx * dx + dy * dy + dz * dz
                if d_sq <= r_cut * r_cut:
                    shifts.append((dx, dy, dz, np.float32(math.exp(-math.sqrt(d_sq) / d_0))))
    return shifts

def compute_phi(active_mask, shifts):
    active_f32 = active_mask.astype(np.float32)
    Phi = np.zeros_like(active_f32, dtype=np.float32)
    for dx, dy, dz, weight in shifts:
        Phi += np.roll(active_f32, shift=(dx, dy, dz), axis=(0, 1, 2)) * weight
    return Phi

def update_grid(grid, shifts, dt, lambda_fac, alpha, gamma_base, beta,
                gamma_loss, rng, h_field: float = 0.0):
    active_mask = grid == 1
    inactive_mask = grid == 0
    Phi = compute_phi(active_mask, shifts)

    spread_rate = lambda_fac * np.expm1(alpha * Phi)
    spread_rate = np.clip(spread_rate, 0.0, None)
    if h_field > 0.0:
        spread_rate = spread_rate + h_field
    p_spread = 1.0 - np.exp(-spread_rate * dt)
    spread_roll = rng.random(grid.shape, dtype=np.float32)
    spread_mask = inactive_mask & (spread_roll < p_spread)

    decay_rate = gamma_base * np.exp(beta * Phi)
    total_rate = decay_rate + gamma_loss
    p_event = 1.0 - np.exp(-total_rate * dt)
    event_roll = rng.random(grid.shape, dtype=np.float32)
    branch_roll = rng.random(grid.shape, dtype=np.float32)
    event_mask = active_mask & (event_roll < p_event)
    p_loss = gamma_loss / np.maximum(total_rate, 1e-30)
    loss_mask = event_mask & (branch_roll < p_loss)
    decay_mask = event_mask & ~loss_mask

    new_grid = grid.copy()
    new_grid[spread_mask] = 1
    new_grid[decay_mask] = 0
    new_grid[loss_mask] = 2
    return new_grid, spread_mask, loss_mask
