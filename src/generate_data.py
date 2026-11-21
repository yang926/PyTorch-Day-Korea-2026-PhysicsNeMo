"""Steady temperatures on an insulated 64 mm aluminum heat spreader.

Each case moves one 8 mm square heater to a specified grid-aligned location and
sets its electrical power. f stores areal heat generation (W/m²); u stores the
cell-center temperature (°C). The effective face cooling coefficient already
represents all heat loss to the ambient. No heat crosses the four plate edges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA = "insulated_aluminum_plate_heater_v1"
MODULE_DIR = Path(__file__).resolve().parent
GRID_SIZE = 64
CELL_M = 0.001
HEATER_CELLS = 8
CONDUCTIVITY_W_MK = 200.0
THICKNESS_M = 0.001
COOLING_W_M2K = 20.0
AMBIENT_C = 25.0
SHEET_CONDUCTANCE = CONDUCTIVITY_W_MK * THICKNESS_M
POWERS_W = (2.0, 1.0, 3.0, 4.0)
PDE = "-k*t*laplacian(T) + h*(T - ambient) = q"


def design_splits(counts, seed):
    """Assign entire heater locations to one split across all four powers."""
    side = GRID_SIZE - HEATER_CELLS + 1
    reserved = [(28, 28), (8, 8)]
    # 677 is coprime to 57²: this visits every possible placement once.
    order = ((int(seed) + 677 * i) % (side * side) for i in range(side * side))
    sites = reserved + [divmod(i, side) for i in order if divmod(i, side) not in reserved]
    needed = {split: (n + len(POWERS_W) - 1) // len(POWERS_W)
              for split, n in counts.items()}
    if min(counts.values()) < 1 or sum(needed.values()) > len(sites):
        raise ValueError("The requested splits need more distinct heater sites than the plate permits, or are empty")
    result, offset = {}, 0
    for split in ("test", "val", "train"):
        positions = sites[offset:offset + needed[split]]
        offset += len(positions)
        result[split] = [(i, j, w) for w in POWERS_W for i, j in positions][:counts[split]]
    return result


def solve_temperatures(q):
    """Invert the cell-centered, zero-flux five-point stencil by even reflection."""
    n = q.shape[-1]
    # Reflection about the outer cell face sets each exterior ghost cell equal
    # to its neighboring interior cell, i.e. makes the outward flux zero.
    even_x = np.concatenate((q, np.flip(q, axis=-2)), axis=-2)
    even_xy = np.concatenate((even_x, np.flip(even_x, axis=-1)), axis=-1)
    eig = 4 * np.sin(np.pi * np.fft.fftfreq(2 * n)) ** 2 / CELL_M ** 2
    eig_y = eig[:n + 1]
    denominator = COOLING_W_M2K + SHEET_CONDUCTANCE * (eig[:, None] + eig_y[None, :])
    spectrum = np.fft.rfft2(even_xy, axes=(-2, -1))
    rise = np.fft.irfft2(spectrum / denominator, s=(2*n, 2*n), axes=(-2, -1))
    return AMBIENT_C + rise[..., :n, :n]


def reference_error(temperature, q):
    """Check the heat balance independently in every cell, including the edges."""
    pad = np.pad(temperature, ((0, 0), (1, 1), (1, 1)), mode="edge")
    lap = (pad[:, 2:, 1:-1] + pad[:, :-2, 1:-1] + pad[:, 1:-1, 2:]
           + pad[:, 1:-1, :-2] - 4 * temperature) / CELL_M ** 2
    balance = -SHEET_CONDUCTANCE * lap + COOLING_W_M2K * (temperature - AMBIENT_C) - q
    return float(np.max(np.abs(balance)))


def write_split(path, cases, split, seed, batch_size):
    import h5py

    xx = np.arange(GRID_SIZE)[None, :, None]
    yy = np.arange(GRID_SIZE)[None, None, :]
    i, j, power = np.asarray(cases).T
    max_residual, max_power_error = 0.0, 0.0
    heater_area = (HEATER_CELLS * CELL_M) ** 2
    with h5py.File(path, "x") as hf:
        hf.attrs.update(schema=SCHEMA, pde=PDE, boundary="insulated_edges",
                        ambient_c=AMBIENT_C, cooling_w_m2k=COOLING_W_M2K,
                        grid_size=GRID_SIZE, cell_m=CELL_M, thickness_m=THICKNESS_M,
                        conductivity_w_mk=CONDUCTIVITY_W_MK, split=split, seed=int(seed))
        hf["heater_x0_mm"] = i.astype("i2")
        hf["heater_y0_mm"] = j.astype("i2")
        hf["power_w"] = power.astype("f4")
        shape = (len(cases), 1, GRID_SIZE, GRID_SIZE)
        fs = hf.create_dataset("f", shape=shape, dtype="f4", chunks=(1, 1, GRID_SIZE, GRID_SIZE), compression="lzf")
        us = hf.create_dataset("u", shape=shape, dtype="f4", chunks=(1, 1, GRID_SIZE, GRID_SIZE), compression="lzf")
        fs.attrs["units"], us.attrs["units"] = "W/m²", "°C"
        for start in range(0, len(cases), batch_size):
            chunk = slice(start, min(start + batch_size, len(cases)))
            ix, iy = i[chunk, None, None], j[chunk, None, None]
            mask = (xx >= ix) & (xx < ix + HEATER_CELLS) & (yy >= iy) & (yy < iy + HEATER_CELLS)
            q = mask * (power[chunk, None, None] / heater_area)
            temperature = solve_temperatures(q)
            cooling_w = COOLING_W_M2K * (temperature - AMBIENT_C).sum(axis=(1, 2)) * CELL_M ** 2
            power_error = max(np.max(np.abs(cooling_w - power[chunk])),
                              np.max(np.abs(q.sum(axis=(1, 2)) * CELL_M ** 2 - power[chunk])))
            max_power_error = max(max_power_error, float(power_error))
            max_residual = max(max_residual, reference_error(temperature, q))
            if max_residual > 1e-5 or max_power_error > 1e-9:
                raise ArithmeticError(f"Reference solution violates heat balance: {max_residual} W/m², {max_power_error} W")
            fs[chunk] = q[:, None].astype("f4")
            us[chunk] = temperature[:, None].astype("f4")
        hf.attrs["max_reference_residual_w_m2_float64"] = max_residual
        hf.attrs["max_power_balance_error_w_float64"] = max_power_error
    return {"file": path.name, "samples": len(cases), "max_reference_residual_w_m2": max_residual,
            "max_power_balance_error_w": max_power_error}


def generate_splits(output_dir, *, train_samples=8000, val_samples=1000, test_samples=1000,
                    grid_size=GRID_SIZE, seed=42, batch_size=32):
    if grid_size != GRID_SIZE or batch_size < 1:
        raise ValueError("This 64 mm plate uses 64 cells per axis; batch_size must be positive")
    output_dir = Path(output_dir)
    counts = dict(train=train_samples, val=val_samples, test=test_samples)
    cases = design_splits(counts, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: output_dir / (s + ".hdf5") for s in counts}
    if any(path.exists() for path in (*paths.values(), output_dir / "manifest.json")):
        raise FileExistsError("Choose a fresh output directory; existing data is preserved")
    records = [write_split(paths[s], cases[s], s, seed, batch_size) for s in counts]
    manifest = {"schema": SCHEMA, "pde": PDE, "seed": seed, "grid_size": grid_size,
                "inputs": "8 mm square heater position and power", "conditioning": "disjoint sites across splits",
                "powers_w": sorted(POWERS_W), "splits": records}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=MODULE_DIR.parent / "data/heated_plate")
    p.add_argument("--train-samples", type=int, default=8000)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--test-samples", type=int, default=1000)
    p.add_argument("--grid-size", type=int, default=GRID_SIZE)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    print(json.dumps(generate_splits(**vars(p.parse_args())), indent=2))
