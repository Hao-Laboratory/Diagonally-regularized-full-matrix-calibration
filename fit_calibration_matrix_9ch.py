"""
Diagonally regularized full-matrix calibration

Created by: Qianqian Yu
Created on: 2026-05-22

This public demo optimize the calibration matrix from the measured and theoretical sample set:

    calibration_samples_9ch_measured.npy      # measured sampling values, shape (K, 9)
    calibration_samples_9ch_theoretical.npy   # theoretical sampling values, shape (K, 9)

Both arrays should use the same channel order, e.g. QE0...QE8.

Supported calibration modes
---------------------------
1) diag:
   Channel-wise diagonal calibration. It first fits
       M[:, c] ~= g[c] * T[:, c]
   and applies
       T_hat[:, c] = M[:, c] / g[c]
   Equivalently, T_hat = M @ diag(1 / g).

2) full:
   Full-matrix calibration. It fits
       T ~= M @ W
   with optional diagonal regularization:
       min_W ||M W - T||_F^2 + alpha ||W - W0||_F^2
   where W0 can be diag(1/g) or identity.

Outputs
-------
- calib_coeff_W.npy: calibration matrix W, shape (9, 9), mapping measured values to theoretical domain
- calib_coeff_g.npy: channel-wise gain g, only for diag mode
- calib_coeff_meta.json: metadata
- meas_calibrated_24x9.npy: calibrated measured values, shape (K, 9)
- compare_norm_sample_XX.png: raw measured vs theoretical normalized channel response
- compare_norm_sample_XX_calibrated.png: calibrated measured vs theoretical normalized channel response
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union


BASE_DIR = Path(__file__).resolve().parent
CONFIG: Dict[str, Any] = {
    # Input sample sets. Both should have shape (K, 9), with the same channel order.
    "measured_samples_path":  BASE_DIR / "calibration_set" / "calibration_samples_9ch_measured.npy",
    "theoretical_samples_path": BASE_DIR / "calibration_set" / "calibration_samples_9ch_theoretical.npy",

    # Output directory.
    "save_dir": BASE_DIR / "calib_out",

    # Calibration mode: "diag" or "full".
    "calibration_mode": "diag",

    # Samples used for fitting: "all" or a 1-based list, e.g. [1, 2, 3, 4].
    # All samples are still evaluated and plotted.
    # "fit_indices_1based": "all",
    # "fit_indices_1based": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "fit_indices_1based": [1],  

    # Full-matrix regularization.
    "full_reg_enable": True,
    "full_reg_alpha": 200.0,
    "full_reg_target": "diag",  # "diag" uses diag(1/g); "I" uses identity.

    # Row-wise normalization used only for visualization.
    "norm_mode": "max",  # "max" or "pctl"
    "pctl": 95.0,
    "eps": 1e-8,

    # Output filenames.
    "save_coeff_name": "calib_coeff",
    "calibrated_meas_npy_filename": "meas_calibrated_24x9.npy",
}


def assert_ok(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_sample_sets(measured_path: str, theoretical_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load measured and theoretical sampling values."""
    M = np.load(measured_path).astype(np.float64)
    T = np.load(theoretical_path).astype(np.float64)

    assert_ok(M.ndim == 2 and T.ndim == 2, "Both sample arrays must be 2D.")
    assert_ok(M.shape == T.shape, f"Shape mismatch: measured={M.shape}, theoretical={T.shape}")
    assert_ok(M.shape[1] == 9, f"Expected 9 channels, got shape {M.shape}")
    assert_ok(np.all(np.isfinite(M)), "Measured samples contain NaN or Inf.")
    assert_ok(np.all(np.isfinite(T)), "Theoretical samples contain NaN or Inf.")

    return M, T


def select_rows(A: np.ndarray, indices_1based: Union[str, Sequence[int]]) -> np.ndarray:
    """Select rows using 1-based sample indices, or return all rows."""
    if indices_1based == "all":
        return A

    sel = np.asarray(indices_1based, dtype=int) - 1
    assert_ok(sel.ndim == 1 and sel.size > 0, "fit_indices_1based must be 'all' or a non-empty list.")
    assert_ok(np.min(sel) >= 0 and np.max(sel) < A.shape[0], "fit_indices_1based contains out-of-range indices.")
    return A[sel]


def normalize_row(v: np.ndarray, mode: str = "max", pctl: float = 95.0, eps: float = 1e-8) -> np.ndarray:
    """Normalize one 9-channel row for visualization."""
    v = np.asarray(v, dtype=np.float64)
    if mode == "max":
        factor = np.max(v) + eps
    elif mode == "pctl":
        factor = np.percentile(v, pctl) + eps
    else:
        raise ValueError("norm_mode must be 'max' or 'pctl'.")
    return (v / factor).astype(np.float32)


def fit_diag_calibration(T: np.ndarray, M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Fit channel-wise gain g from M[:, c] ~= g[c] * T[:, c].

    The calibration applied to measured data is T_hat = M / g.
    """
    assert_ok(T.shape == M.shape, "T and M must have the same shape.")
    C = T.shape[1]
    g = np.zeros(C, dtype=np.float64)

    for c in range(C):
        x = T[:, c]
        y = M[:, c]
        denominator = np.dot(x, x) + eps
        g[c] = np.dot(x, y) / denominator
        if g[c] < 0:
            g[c] = 0.0

    return g


def fit_full_calibration(
    T: np.ndarray,
    M: np.ndarray,
    reg_alpha: float = 0.0,
    reg_target: str = "diag",
) -> np.ndarray:
    """
    Fit full calibration matrix W for T ~= M @ W.

    If reg_alpha > 0, solve:
        min_W ||M W - T||_F^2 + reg_alpha ||W - W0||_F^2
    """
    T = np.asarray(T, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    assert_ok(T.shape == M.shape, "T and M must have the same shape.")

    _, C = T.shape

    if reg_alpha <= 0.0:
        W, *_ = np.linalg.lstsq(M, T, rcond=None)
        return W

    if reg_target == "diag":
        g = fit_diag_calibration(T, M)
        W0 = np.diag(1.0 / (g + 1e-12))
    elif reg_target == "I":
        W0 = np.eye(C, dtype=np.float64)
    else:
        raise ValueError("full_reg_target must be 'diag' or 'I'.")

    sqrt_alpha = np.sqrt(reg_alpha)
    X_aug = np.vstack([M, sqrt_alpha * np.eye(C, dtype=np.float64)])
    Y_aug = np.vstack([T, sqrt_alpha * W0])
    W, *_ = np.linalg.lstsq(X_aug, Y_aug, rcond=None)
    return W


def evaluate_calibration(T: np.ndarray, T_hat: np.ndarray, eps: float = 1e-12) -> Dict[str, Any]:
    """Compute per-channel and averaged fitting metrics."""
    assert_ok(T.shape == T_hat.shape, "T and T_hat must have the same shape.")

    mse = np.mean((T_hat - T) ** 2, axis=0)
    var = np.var(T, axis=0) + eps
    r2 = 1.0 - mse / var
    rel = np.mean(np.abs(T_hat - T) / (np.abs(T) + eps), axis=0)

    return {
        "mse_per_ch": mse.astype(float).tolist(),
        "mse_mean": float(np.mean(mse)),
        "r2_per_ch": r2.astype(float).tolist(),
        "r2_mean": float(np.mean(r2)),
        "relerr_per_ch": rel.astype(float).tolist(),
        "relerr_mean": float(np.mean(rel)),
    }


def analyze_matrix(W: np.ndarray) -> Dict[str, Any]:
    """Return simple diagnostics for a calibration matrix."""
    W = np.asarray(W, dtype=np.float64)
    diag = np.diag(W)
    diag_mat = np.diag(diag)
    offdiag_mat = W - diag_mat
    diag_norm = float(np.linalg.norm(diag_mat))
    offdiag_norm = float(np.linalg.norm(offdiag_mat))

    return {
        "condition_number": float(np.linalg.cond(W)),
        "diag": diag.astype(float).tolist(),
        "diag_norm": diag_norm,
        "offdiag_norm": offdiag_norm,
        "offdiag_over_diag": float(offdiag_norm / (diag_norm + 1e-12)),
    }


def setup_plot_style() -> None:
    """Apply a compact paper-style matplotlib configuration."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["savefig.facecolor"] = "none"
    plt.rcParams["savefig.edgecolor"] = "none"


def plot_normalized_comparison(
    measured_row: np.ndarray,
    theoretical_row: np.ndarray,
    save_path: str,
    norm_mode: str = "max",
    pctl: float = 95.0,
    eps: float = 1e-8,
    measured_label: str = "Meas",
    theoretical_label: str = "Theory",
) -> None:
    """Plot normalized 9-channel measured/theoretical comparison."""
    measured_norm = normalize_row(measured_row, mode=norm_mode, pctl=pctl, eps=eps)
    theoretical_norm = normalize_row(theoretical_row, mode=norm_mode, pctl=pctl, eps=eps)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    ax.plot(
        measured_norm,
        "o-",
        color="#104680",
        linewidth=1.5,
        markersize=5,
        label=measured_label,
    )
    ax.plot(
        theoretical_norm,
        "s-",
        color="#B72230",
        linewidth=1.5,
        markersize=5,
        label=theoretical_label,
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Normalized channel response")
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"ch{c}" for c in range(9)])
    ax.tick_params(width=1.0, length=4, direction="out")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.grid(False)
    ax.legend(loc="best", frameon=True, framealpha=0.80, ncol=1)

    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.18, top=0.95)
    fig.savefig(save_path, dpi=300, transparent=True)
    plt.close(fig)


def save_all_comparison_plots(
    M: np.ndarray,
    T: np.ndarray,
    T_hat: np.ndarray,
    save_dir: str,
    norm_mode: str,
    pctl: float,
    eps: float,
) -> None:
    """Save raw and calibrated normalized comparison plots for every sample."""
    for i in range(M.shape[0]):
        raw_path = os.path.join(save_dir, f"compare_norm_sample_{i + 1:02d}.png")
        cal_path = os.path.join(save_dir, f"compare_norm_sample_{i + 1:02d}_calibrated.png")

        plot_normalized_comparison(
            M[i],
            T[i],
            raw_path,
            norm_mode=norm_mode,
            pctl=pctl,
            eps=eps,
            measured_label="Meas",
            theoretical_label="Theory",
        )
        plot_normalized_comparison(
            T_hat[i],
            T[i],
            cal_path,
            norm_mode=norm_mode,
            pctl=pctl,
            eps=eps,
            measured_label="Calibrated meas",
            theoretical_label="Theory",
        )


def main(cfg: Dict[str, Any]) -> None:
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True) 
    setup_plot_style()

    M_all, T_all = load_sample_sets(
        cfg["measured_samples_path"],
        cfg["theoretical_samples_path"],
    )

    use_idx = cfg.get("fit_indices_1based", "all")
    M_fit = select_rows(M_all, use_idx)
    T_fit = select_rows(T_all, use_idx)

    mode = cfg.get("calibration_mode", "full")
    eps = float(cfg.get("eps", 1e-8))
    prefix = cfg.get("save_coeff_name", "calib_coeff")

    if mode == "diag":
        g = fit_diag_calibration(T_fit, M_fit)
        W = np.diag(1.0 / (g + eps))
        T_hat = M_all @ W

        np.save(os.path.join(cfg["save_dir"], f"{prefix}_g.npy"), g.astype(np.float32))
        print("\n=== Diagonal calibration ===")
        print("fit_indices =", use_idx)
        print("g =", np.round(g, 6))

    elif mode == "full":
        reg_alpha = float(cfg.get("full_reg_alpha", 0.0)) if cfg.get("full_reg_enable", False) else 0.0
        reg_target = cfg.get("full_reg_target", "diag")
        W = fit_full_calibration(
            T_fit,
            M_fit,
            reg_alpha=reg_alpha,
            reg_target=reg_target,
        )
        T_hat = M_all @ W

        print("\n=== Full-matrix calibration ===")
        print("fit_indices =", use_idx)
        print("full_reg_enable =", bool(cfg.get("full_reg_enable", False)))
        print("full_reg_alpha =", reg_alpha)
        print("full_reg_target =", reg_target)
        print("W =\n", np.round(W, 6))

    else:
        raise ValueError("calibration_mode must be 'diag' or 'full'.")

    matrix_info = analyze_matrix(W)
    metrics = evaluate_calibration(T_all, T_hat)

    print("\n=== Calibration matrix diagnostics ===")
    print("cond(W) =", matrix_info["condition_number"])
    print("diag(W) =", np.round(np.asarray(matrix_info["diag"]), 6))
    print("offdiag_norm / diag_norm =", matrix_info["offdiag_over_diag"])

    print("\n=== Evaluation on all samples ===")
    print("R2 per channel =", np.round(np.asarray(metrics["r2_per_ch"]), 4))
    print("Mean R2 =", round(metrics["r2_mean"], 4))
    print("Relative error per channel =", np.round(np.asarray(metrics["relerr_per_ch"]), 4))
    print("Mean relative error =", round(metrics["relerr_mean"], 4))

    np.save(os.path.join(cfg["save_dir"], f"{prefix}_W.npy"), W.astype(np.float32))
    np.save(
        os.path.join(cfg["save_dir"], cfg.get("calibrated_meas_npy_filename", "meas_calibrated_24x9.npy")),
        T_hat.astype(np.float32),
    )

    meta = {
        "mode": mode,
        "fit_indices_1based": use_idx,
        "measured_samples_path": str(cfg["measured_samples_path"]),
        "theoretical_samples_path": str(cfg["theoretical_samples_path"]),
        "calibration_formula": "T_hat = M @ W",
        "full_reg_enable": bool(cfg.get("full_reg_enable", False)) if mode == "full" else False,
        "full_reg_alpha": float(cfg.get("full_reg_alpha", 0.0)) if mode == "full" else None,
        "full_reg_target": cfg.get("full_reg_target", None) if mode == "full" else None,
        "matrix_info": matrix_info,
        "metrics_all_samples": metrics,
    }
    with open(os.path.join(cfg["save_dir"], f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_all_comparison_plots(
        M_all,
        T_all,
        T_hat,
        cfg["save_dir"],
        norm_mode=cfg.get("norm_mode", "max"),
        pctl=float(cfg.get("pctl", 95.0)),
        eps=eps,
    )

    print("\nSaved outputs to:", os.path.abspath(cfg["save_dir"]))


if __name__ == "__main__":
    main(CONFIG)
