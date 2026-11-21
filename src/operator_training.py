"""Train and evaluate a neural operator for steady heat conduction in a plate."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

try:
    from .generate_data import (AMBIENT_C, CELL_M, COOLING_W_M2K, GRID_SIZE, HEATER_CELLS,
                                PDE, POWERS_W, SCHEMA, SHEET_CONDUCTANCE)
except ImportError:
    from generate_data import (AMBIENT_C, CELL_M, COOLING_W_M2K, GRID_SIZE, HEATER_CELLS,
                               PDE, POWERS_W, SCHEMA, SHEET_CONDUCTANCE)

MODULE_DIR = Path(__file__).resolve().parent
WORKSHOP_DIR = MODULE_DIR.parent
from .artifacts import staged_output


class BatchedPhysicsInformer:
    """Apply the scalar-field PhysicsInformer adapter independently per sample.

    PhysicsNeMo 2.2.2's spectral gradient adapter extracts u[0, 0]. Calling it
    on an entire batch can broadcast the first sample's derivatives. Separate
    calls preserve each field and its autograd graph.
    """
    def __init__(self, informer):
        self.informer = informer

    def forward(self, fields):
        u, f = fields["u"], fields["f"]
        if u.shape != f.shape or u.ndim != 4 or u.shape[1] != 1 or u.shape[0] == 0:
            raise ValueError("Physics residual expects matching [batch, 1, nx, ny] fields")
        values = [self.informer.forward({"u": u[index:index + 1], "f": f[index:index + 1]})[
                  "reaction_diffusion"] for index in range(u.shape[0])]
        return {"reaction_diffusion": torch.cat(values, dim=0)}


class UnfinishedExerciseError(RuntimeError):
    """A participant has not yet implemented a documented learning task."""


def checked_datasets(normalized_pairs, dataset_builder):
    """Check sample order and values against the supplied train, validation and test splits."""
    exercise_pairs = [tuple(tensor.clone() for tensor in pair) for pair in normalized_pairs]
    datasets = dataset_builder(*exercise_pairs)
    if not isinstance(datasets, (tuple, list)) or len(datasets) != 3:
        raise ValueError("build_datasets must return train, validation and test TensorDataset objects")
    for name, dataset, expected in zip(("train", "val", "test"), datasets, normalized_pairs):
        if type(dataset) is not TensorDataset or len(dataset.tensors) != 2:
            raise ValueError(f"{name}: return TensorDataset(f, u), without a custom dataset wrapper")
        for field, actual, canonical in zip(("f", "u"), dataset.tensors, expected):
            if (actual.shape != canonical.shape or actual.dtype != canonical.dtype
                    or actual.device != canonical.device or not torch.equal(actual, canonical)):
                raise ValueError(
                    f"{name}.{field}: preserve the provided split, sample order and normalized values; "
                    "do not swap splits, replace targets or normalize again"
                )
    return (datasets[0], TensorDataset(*normalized_pairs[1]), TensorDataset(*normalized_pairs[2]))


def load_pair(path, expected_split):
    with h5py.File(path, "r") as hf:
        if hf.attrs.get("schema") != SCHEMA or hf.attrs.get("split") != expected_split:
            raise ValueError(f"{path}: wrong dataset schema or split. Run generate_data.py into a fresh directory.")
        f, u = torch.from_numpy(hf["f"][:]), torch.from_numpy(hf["u"][:])
        x0, y0, power = hf["heater_x0_mm"][:], hf["heater_y0_mm"][:], hf["power_w"][:]
        if any(a.shape != (len(f),) for a in (x0, y0, power)):
            raise ValueError(f"{path}: heater metadata and field counts do not agree")
        if (not np.isin(power, POWERS_W).all() or min(x0.min(), y0.min()) < 0
                or max(x0.max(), y0.max()) > GRID_SIZE - HEATER_CELLS):
            raise ValueError(f"{path}: invalid heater location or electrical power")
        if len(set(zip(x0, y0, power))) != len(f):
            raise ValueError(f"{path}: duplicate heater design")
        locations = set(zip(x0, y0))
    if f.shape != u.shape or f.ndim != 4 or f.shape[1] != 1 or f.shape[-2:] != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{path}: expected matching [samples, 1, n, n] tensors")
    if f.dtype != torch.float32 or u.dtype != torch.float32 or f.shape[-1] < 4:
        raise ValueError(f"{path}: expected float32 fields on a grid of at least 4 points per axis")
    if len(f) < 1 or not torch.isfinite(f).all() or not torch.isfinite(u).all():
        raise ValueError(f"{path}: empty or non-finite data")
    measured_power = f.mean(dim=(-3, -2, -1)) * (GRID_SIZE * CELL_M) ** 2
    target_power = torch.from_numpy(power).to(f)
    if not torch.allclose(measured_power, target_power, atol=2e-5, rtol=1e-5):
        raise ValueError(f"{path}: heat source and electrical power disagree")
    if not torch.allclose(u.mean(dim=(-3, -2, -1)), AMBIENT_C + measured_power /
                          ((GRID_SIZE * CELL_M) ** 2 * COOLING_W_M2K), atol=1e-4):
        raise ValueError(f"{path}: reference temperatures violate total heat balance")
    return (f, u), locations


def load_data(data_dir):
    pairs, used_locations = [], set()
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.hdf5"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python generate_data.py --output-dir {data_dir} first.")
        pair, locations = load_pair(path, split)
        if used_locations.intersection(locations):
            raise ValueError("A heater location occurs in more than one split")
        used_locations.update(locations)
        pairs.append(pair)
    return pairs


def predict_field(model, normalized_forcing):
    prediction = model(normalized_forcing)
    if not isinstance(prediction, torch.Tensor) or prediction.shape != normalized_forcing.shape:
        raise ValueError("The operator must return a tensor with the same [batch, 1, n, n] shape as its input")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite operator prediction")
    # The mean is fixed by the integrated heat balance; the model supplies the spatial contrast.
    return prediction - prediction.mean(dim=(-2, -1), keepdim=True)


def mean_temperature(q):
    return AMBIENT_C + q.mean(dim=(-2, -1), keepdim=True) / COOLING_W_M2K


def plate_residual(temperature, q):
    # Insulated faces have zero outward flux. Double precision avoids losing small
    # differences between neighboring temperatures in the five-point stencil.
    t = temperature.double()
    pad = torch.nn.functional.pad(t, (1, 1, 1, 1), mode="replicate")
    lap = (pad[..., 2:, 1:-1] + pad[..., :-2, 1:-1] + pad[..., 1:-1, 2:]
           + pad[..., 1:-1, :-2] - 4 * t) / CELL_M ** 2
    return -SHEET_CONDUCTANCE * lap + COOLING_W_M2K * (t - AMBIENT_C) - q.double()


def physics_residual(physics, prediction, forcing):
    residual = physics.forward({"u": prediction, "f": forcing})["reaction_diffusion"]
    if not isinstance(residual, torch.Tensor) or residual.shape != prediction.shape:
        raise ValueError("The physics residual must preserve every [batch, 1, n, n] field")
    if not torch.isfinite(residual).all():
        raise FloatingPointError("Non-finite physics residual")
    return residual


def evaluate(model, loader, device, stats, physics=None, *, collect_predictions=True):
    """Accumulate metrics by grid point, retaining fields only when exporting.

    Float64 sums avoid float32 overflow in squared errors. Weighting by point
    count (not batch count) also handles a smaller final minibatch correctly.
    """
    predictions, truths, forcings = [], [], []
    error_sum = truth_rise_sum = residual_sum = abs_sum = peak_sum = sym_mse_sum = 0.0
    count = case_count = 0
    model.eval()
    with torch.no_grad():
        for normalized_f, normalized_u in loader:
            normalized_f, normalized_u = normalized_f.to(device), normalized_u.to(device)
            forcing = normalized_f * stats["f_std"] + stats["f_mean"]
            mean_t = mean_temperature(forcing)
            pred = predict_field(model, normalized_f) * stats["u_std"] + mean_t
            truth = normalized_u * stats["u_std"] + mean_t
            error = pred.double() - truth.double()
            error_sum += error.square().sum().item()
            abs_sum += error.abs().sum().item()
            peak_sum += (pred.amax(dim=(-2, -1)) - truth.amax(dim=(-2, -1))).abs().sum().item()
            case_count += len(truth)
            truth_rise_sum += (truth.double() - AMBIENT_C).square().sum().item()
            residual = plate_residual(pred, forcing)
            residual_sum += residual.square().sum().item()
            if physics is not None:
                sym_residual = physics_residual(physics, pred, forcing)
                if not torch.allclose(sym_residual.double(), residual, rtol=2e-3, atol=2.0):
                    raise RuntimeError("Supplied physics residual disagrees with the insulated-plate heat balance")
                sym_mse_sum += sym_residual.double().square().sum().item()
            count += truth.numel()
            if collect_predictions:
                predictions.append(pred.cpu())
                truths.append(truth.cpu())
                forcings.append(forcing.cpu())
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    metrics = {"rmse": (error_sum / count) ** 0.5,
               "relative_l2": (error_sum / max(truth_rise_sum, 1e-24)) ** 0.5,
               "mae_c": abs_sum / count, "peak_mae_c": peak_sum / case_count,
               "pde_rmse_w_m2": (residual_sum / count) ** 0.5}
    if physics is not None:
        metrics["pde_rmse_physicsinformer"] = (sym_mse_sum / count) ** 0.5
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Non-finite evaluation metric; check data and normalization")
    arrays = ({name: torch.cat(items).numpy() for name, items in
               (("f", forcings), ("u_true", truths), ("u_pred", predictions))}
              if collect_predictions else None)
    return metrics, arrays


def validate_config(config, steps):
    """Reject common YAML editing mistakes before constructing a model."""
    if not isinstance(config, dict) or any(not isinstance(config.get(key), dict)
                                           for key in ("data", "model", "training")):
        raise ValueError("Config must contain data, model and training mappings")
    training = config["training"]
    integers = {"steps": steps, "grid_size": config["data"].get("grid_size"),
                "batch_size": training.get("batch_size"), "cpu_threads": training.get("cpu_threads", 2)}
    for name, value in integers.items():
        minimum = 4 if name == "grid_size" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("learning_rate", training.get("learning_rate")),
                        ("physics_weight", training.get("physics_weight", 1.0))):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (name == "learning_rate" and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'learning_rate' else 'nonnegative'}")


def training_clock(device):
    """Measure completed CUDA work, not just asynchronous kernel submission."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def run(level, model_builder, *, dataset_builder, physics_builder=None,
        argv=None):
    parser = argparse.ArgumentParser(description=f"Level {level}: heated-plate neural operator")
    config_names = {1: "FNO", 2: "AFNO", 3: "PINO"}
    parser.add_argument("--config", type=Path, default=WORKSHOP_DIR / f"config/config_{config_names[level]}.yaml")
    parser.add_argument("--data-dir", type=Path, default=WORKSHOP_DIR / "data/heated_plate")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=WORKSHOP_DIR / f"runs/fno-level{level}")
    args = parser.parse_args(argv)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a fresh --output-dir; previous results are preserved: {output}")
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    steps = args.steps if args.steps is not None else (
        config.get("training", {}).get("steps") if isinstance(config, dict)
        and isinstance(config.get("training"), dict) else None)
    validate_config(config, steps)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Small classroom grids benefit from a bounded thread count on shared CPUs.
    torch.set_num_threads(int(config["training"].get("cpu_threads", 2)))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable. Use --device cpu.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if level == 3 and physics_builder is None:
        raise ValueError("Level 3 requires the learner's PINO physics builder")
    grid_size = int(config["data"]["grid_size"])
    model = model_builder(config["model"], grid_size).to(device)
    physics = physics_builder(device) if physics_builder is not None else None
    pairs = load_data(args.data_dir.resolve())
    actual_grid_size = pairs[0][0].shape[-1]
    if actual_grid_size != grid_size:
        raise ValueError(f"Dataset grid {actual_grid_size} != config grid {grid_size}; use a matching --config.")
    train_f, train_u = pairs[0]
    train_contrast = train_u - mean_temperature(train_f)
    stats = {"f_mean": train_f.mean().item(), "f_std": train_f.std(unbiased=False).clamp_min(1e-12).item(),
             "u_mean": 0.0, "u_std": train_contrast.std(unbiased=False).clamp_min(1e-12).item(),
             "ambient_c": AMBIENT_C, "cooling_w_m2k": COOLING_W_M2K}
    del train_contrast
    if not all(math.isfinite(value) for value in stats.values()):
        raise FloatingPointError("Non-finite training normalization statistics")
    normalized = [((f - stats["f_mean"]) / stats["f_std"], (u - mean_temperature(f)) / stats["u_std"])
                  for f, u in pairs]
    datasets = checked_datasets(normalized, dataset_builder)
    # Dataset wrappers now own the required normalized tensors. Release raw
    # fields and the temporary canonical training copy before optimization.
    del pairs, normalized, train_f, train_u
    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(datasets[1], batch_size=batch_size)
    test_loader = DataLoader(datasets[2], batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    physics_weight = float(config["training"].get("physics_weight", 1.0))
    initial_test, _ = evaluate(model, test_loader, device, stats, physics, collect_predictions=False)
    history, iterator = [], iter(train_loader)
    training_started = training_clock(device)
    for step in range(1, steps + 1):
        model.train()
        try:
            normalized_f, normalized_u = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            normalized_f, normalized_u = next(iterator)
        normalized_f, normalized_u = normalized_f.to(device), normalized_u.to(device)
        optimizer.zero_grad(set_to_none=True)
        normalized_prediction = predict_field(model, normalized_f)
        data_loss = (normalized_prediction - normalized_u).square().mean()
        physics_loss = torch.zeros((), device=device)
        if physics is not None:
            forcing = normalized_f * stats["f_std"] + stats["f_mean"]
            prediction = normalized_prediction * stats["u_std"] + mean_temperature(forcing)
            residual = physics_residual(physics, prediction, forcing)
            physics_loss = (residual / stats["f_std"]).square().mean()
        loss = data_loss + physics_weight * physics_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        history.append({"step": step, "loss": loss.item(), "data_loss": data_loss.item(),
                        "physics_loss": physics_loss.item()})
        if step == 1 or step == steps or step % 100 == 0:
            print(json.dumps(history[-1]), flush=True)
    training_seconds = training_clock(device) - training_started
    validation, _ = evaluate(model, val_loader, device, stats, physics, collect_predictions=False)
    test, arrays = evaluate(model, test_loader, device, stats, physics)
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise FloatingPointError("Non-finite predictions")
    metrics = {"level": level, "method": config_names[level], "pde": PDE, "field_units": {"f": "W/m²", "u": "°C"},
               "relative_l2_reference": "temperature rise above 25 °C",
               "model_target": "spatial temperature contrast around the per-case heat-balance mean",
               "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo"),
               "torch_version": str(torch.__version__), "device": str(device), "steps": steps, "seed": args.seed,
               "training_seconds": training_seconds,
               "training_seconds_scope": "optimizer loop, batch transfers and logging; excludes setup, evaluation and export",
               "reference": False, "status": "completed",
               "evaluation_equations": "insulated_cell_centered_heat_balance_W_per_m2",
               "evaluation_data": "canonical_local_validation_and_test_splits",
               "dataset_exercise_checked": True,
               "evaluation_protocol": "heated_plate_disjoint_heater_sites_v1",
               "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"],
               "loss_measurement": "training minibatch loss before each optimizer step",
               "initial_test": initial_test, "test_relative_l2_before": initial_test["relative_l2"],
               "test_relative_l2_after": test["relative_l2"],
               "validation": validation, "test": test, "normalization_train_only": stats,
               "split_samples": dict(zip(("train", "val", "test"), map(len, datasets)))}
    # json allow_nan=False rejects non-finite scalar metrics as well.
    serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    with staged_output(output) as destination:
        with (destination / "loss.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        np.savez_compressed(destination / "predictions.npz", **arrays)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            metrics["preview"] = "unavailable: install matplotlib"
            serialized = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
            try:
                fields = [arrays["u_true"][0, 0], arrays["u_pred"][0, 0],
                          arrays["u_pred"][0, 0] - arrays["u_true"][0, 0]]
                low = min(fields[0].min(), fields[1].min())
                high = max(fields[0].max(), fields[1].max())
                limit = max(np.abs(fields[2]).max(), 1e-8)
                for ax, values, title, vmin, vmax, cmap in zip(
                    axes, fields, ("Reference temperature", "FNO temperature", "Prediction − reference"),
                    (low, low, -limit), (high, high, limit), ("inferno", "inferno", "coolwarm")
                ):
                    im = ax.imshow(values.T, origin="lower", extent=(0, 64, 0, 64), cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
                    fig.colorbar(im, ax=ax, label="°C")
                fig.suptitle(f"{config_names[level]}: {steps} training steps")
                fig.savefig(destination / "preview.png", dpi=140)
            finally:
                plt.close(fig)
        torch.save({"model_state_dict": model.state_dict(), "config": config,
                    "normalization": stats, "metrics": metrics, "steps": steps, "seed": args.seed,
                    "physicsnemo_version": metrics["physicsnemo_version"], "device": str(device)},
                   destination / "model.pt")
        (destination / "metrics.json").write_text(serialized)
    print(serialized, flush=True)
    return metrics
