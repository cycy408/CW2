import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ========== matplotlib (no sklearn) ==========
def plot_mse_comparison(mse_dict, save_path="output/mse_comparison.png",
                        title="Feature Selection Method vs. Test Set MSE",
                        verbose=True):
    _ensure_dir(save_path)

    models = list(mse_dict.keys())
    mse_values = list(mse_dict.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, mse_values, color=colors)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=15, ha='right')

    for bar, val in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.01,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    if verbose:
        print(f"[OK] Chart saved to {save_path}")


# ========== NumPy-only implementation (grid calculation) + matplotlib ==========
def plot_geospatial_heatmap(X_test, y_test, y_pred, lat_idx, lon_idx,
                            grid_size=10, save_path="output/geospatial_heatmap.png",
                            vmin=None, vmax=None, verbose=True):
    _ensure_dir(save_path)

    lats = X_test[:, lat_idx]
    lons = X_test[:, lon_idx]
    errors = np.abs(np.asarray(y_test) - np.asarray(y_pred))

    lat_bins = np.linspace(lats.min(), lats.max(), grid_size + 1)
    lon_bins = np.linspace(lons.min(), lons.max(), grid_size + 1)

    grid = np.full((grid_size, grid_size), np.nan)
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (
                (lats >= lat_bins[i]) & (lats < lat_bins[i + 1]) &
                (lons >= lon_bins[j]) & (lons < lon_bins[j + 1])
            )
            if mask.sum() > 0:
                grid[i, j] = errors[mask].mean()

    if vmin is None:
        vmin = np.nanmin(grid)
    if vmax is None:
        vmax = np.nanmax(grid)

    plt.figure(figsize=(10, 7))
    img = plt.imshow(
        grid, origin='lower', aspect='auto', cmap='RdYlGn_r',
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(img, label='Mean Absolute Error')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Geospatial Prediction Error Heatmap ({grid_size}x{grid_size} grid)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    if verbose:
        print(f"[OK] Heatmap saved to {save_path}")
    return vmin, vmax


# ========== matplotlib (no sklearn) ==========
def plot_fusion_vs_single(results_dict, save_path="output/fusion_vs_single.png",
                          verbose=True):
    """
    results_dict: {model_name: (MSE, MAE), ...}
    Draws grouped bar chart: each model has MSE + MAE bars side by side.
    """
    _ensure_dir(save_path)

    names = list(results_dict.keys())
    mse_vals = [results_dict[n][0] for n in names]
    mae_vals = [results_dict[n][1] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_mse = ax.bar(x - width / 2, mse_vals, width, label="MSE", color="#1f77b4")
    bars_mae = ax.bar(x + width / 2, mae_vals, width, label="MAE", color="#ff7f0e")

    ax.set_ylabel("Error Value", fontsize=12)
    ax.set_title("Single Models vs Weighted Fusion: MSE & MAE Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()

    for bar, val in zip(bars_mse, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_mae, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    if verbose:
        print(f"[OK] Chart saved to {save_path}")


# ========== matplotlib (no sklearn) ==========
def plot_outlier_removal_comparison(mse_before, mae_before, mse_after, mae_after,
                                     model_name, save_path="output/outlier_removal_comparison.png",
                                     verbose=True):
    _ensure_dir(save_path)

    metrics = ["MSE", "MAE"]
    before_vals = [mse_before, mae_before]
    after_vals = [mse_after, mae_after]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, before_vals, width, label="Before Removal",
                   color="#d62728", edgecolor="white")
    bars2 = ax.bar(x + width / 2, after_vals, width, label="After Removal",
                   color="#2ca02c", edgecolor="white")

    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Outlier Removal: Before vs After\n({model_name})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    for bar, val in zip(bars1, before_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.4f}", ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, after_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    if verbose:
        print(f"[OK] Chart saved to {save_path}")
