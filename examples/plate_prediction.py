"""Run the trained plate model on a specified heater position and power."""
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.generate_data import CELL_M, GRID_SIZE, HEATER_CELLS, PDE, POWERS_W, solve_temperatures
from src.operator_training import mean_temperature, predict_field


def predict_heater_case(model_path, heater_center_mm, power_w, model_builder, device):
    """Return applied heat flux, the numerical reference, and the trained model prediction."""
    centers = np.asarray(heater_center_mm)
    if centers.shape != (2,) or centers.dtype.kind not in "iuf":
        raise ValueError("히터 중심 좌표 두 개를 mm 단위로 입력하세요.")
    half = HEATER_CELLS // 2
    if (not np.isfinite(centers).all() or np.any(centers != np.round(centers))
            or np.any(centers < half) or np.any(centers > GRID_SIZE - half)):
        raise ValueError("8 mm 히터의 중심 x, y는 각각 4~60 mm 사이의 정수여야 합니다.")
    if isinstance(power_w, bool) or not isinstance(power_w, Real) or power_w not in POWERS_W:
        raise ValueError("히터 전력은 1, 2, 3, 4 W 중 하나를 선택하세요.")

    x0, y0 = centers.astype(int) - half
    q = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    q[x0:x0 + HEATER_CELLS, y0:y0 + HEATER_CELLS] = power_w / (HEATER_CELLS * CELL_M) ** 2
    reference = solve_temperatures(q.astype(np.float64)[None])[0]

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if checkpoint["metrics"].get("pde") != PDE:
        raise ValueError("4절에서 학습한 알루미늄 판 모델의 결과 폴더를 선택하세요.")
    config = checkpoint["config"]
    model = model_builder(config["model"], config["data"]["grid_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = checkpoint["normalization"]
    with torch.inference_mode():
        forcing = torch.from_numpy(q)[None, None].to(device)
        normalized = (forcing - stats["f_mean"]) / stats["f_std"]
        predicted = predict_field(model, normalized) * stats["u_std"] + mean_temperature(forcing)
    return q, reference, predicted[0, 0].cpu().numpy()


def plot_heater_cases(cases, labels):
    """Show each heat source, numerical temperature and FNO temperature on physical axes."""
    heat_max = max(q.max() for q, _, _ in cases)
    temp_min = min(min(target.min(), pred.min()) for _, target, pred in cases)
    temp_max = max(max(target.max(), pred.max()) for _, target, pred in cases)
    headings = ("Heat flux (W/m²)", "Reference (°C)", "FNO (°C)")
    length_mm = 1000 * GRID_SIZE * CELL_M
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 3.2 * len(cases)),
                             squeeze=False, constrained_layout=True)
    for row, (fields, label) in enumerate(zip(cases, labels)):
        for col, (ax, field, heading) in enumerate(zip(axes[row], fields, headings)):
            low, high = (0, heat_max) if col == 0 else (temp_min, temp_max)
            ax.imshow(field.T, origin="lower", extent=(0, length_mm, 0, length_mm),
                      cmap="magma" if col == 0 else "inferno", vmin=low, vmax=high)
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
            if col == 0:
                ax.set_title(f"{label}\n{heading}")
            elif row == 0:
                ax.set_title(heading)
    fig.colorbar(axes[0, 0].images[0], ax=axes[:, 0], shrink=0.8, label="W/m²")
    fig.colorbar(axes[0, 1].images[0], ax=axes[:, 1:], shrink=0.8, label="°C")
    return fig
