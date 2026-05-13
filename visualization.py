import os
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_mse_comparison(mse_dict, save_path="output/mse_comparison.png",
                        title="Feature Selection Method vs. Test Set MSE"):
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
    print(f"[OK] Chart saved to {save_path}")


def plot_geospatial_heatmap(X_test, y_test, y_pred, lat_idx, lon_idx,
                            grid_size=10, save_path="output/geospatial_heatmap.png"):
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

    plt.figure(figsize=(10, 7))
    img = plt.imshow(
        grid, origin='lower', aspect='auto', cmap='RdYlGn_r',
        extent=[lons.min(), lons.max(), lats.min(), lats.max()]
    )
    plt.colorbar(img, label='Mean Absolute Error')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Geospatial Prediction Error Heatmap ({grid_size}x{grid_size} grid)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Heatmap saved to {save_path}")
