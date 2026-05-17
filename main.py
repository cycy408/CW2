import os
import numpy as np
from sklearn.linear_model import LinearRegression

# ============================================================
# RANDOM SEED CONFIGURATION
#   Set to an integer (e.g. 42) → fixed seed, results reproducible
#   Set to None                  → random seed, results vary each run
#   Change the value below to toggle:
# ============================================================
RANDOM_SEED = 42

from data_prep import load_data, split_data, fill_missing_values, standardize_features
from feature_selection import select_features_numpy, select_features_fregression, compare_feature_sets
from models import train_random_forest, print_feature_importance
from evaluation import compute_mse, compute_mae, compute_r2
from visualization import plot_mse_comparison, plot_geospatial_heatmap, plot_outlier_removal_comparison
from analysis import detect_outliers_iqr, analyze_outlier_features, retrain_best_model_after_outlier_removal
import advanced


def main():
    print("\n" + "=" * 60)
    print("INT101 California Housing Price Prediction")
    print("=" * 60 + "\n")

    # ==================== Basic Layer ====================

    # --- 1. Data loading & preprocessing ---
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_SEED)
    X_train, X_test = fill_missing_values(X_train, X_test)
    X_train_scaled, X_test_scaled, _ = standardize_features(X_train, X_test)

    # --- 2. Baseline: Linear Regression (all features) ---
    print("=" * 50)
    print("Training: Linear Regression (all features)")
    print("=" * 50)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    mse_lr = compute_mse(y_test, y_pred_lr)
    mae_lr = compute_mae(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)

    print(f"[OK] Linear Regression trained")
    print(f"  Intercept: {lr_model.intercept_:.6f}")
    print(f"  Coefficients:")
    for name, coef in zip(feature_names, lr_model.coef_):
        print(f"    {name:15s}: {coef:.6f}")
    print(f"\n  MSE:  {mse_lr:.4f}")
    print(f"  MAE:  {mae_lr:.4f}")
    print(f"  RMSE: {rmse_lr:.4f}\n")

    # --- 3. Feature selection ---
    print("=" * 50)
    print("Feature Selection 1: NumPy Pearson Correlation")
    print("=" * 50)
    indices_numpy, scores_numpy = select_features_numpy(X_train, y_train, k=5)
    print(f"Selected indices: {indices_numpy}")
    print(f"Feature names:    {[feature_names[i] for i in indices_numpy]}")
    print(f"Pearson scores:   {scores_numpy}")

    print(f"\n{'=' * 50}")
    print("Feature Selection 2: F-Regression / F-Test (NumPy)")
    print("=" * 50)
    indices_freg, scores_freg = select_features_fregression(X_train, y_train, k=5)
    print(f"Selected indices: {indices_freg}")
    print(f"Feature names:    {[feature_names[i] for i in indices_freg]}")
    print(f"F-scores:         {scores_freg}")

    compare_feature_sets(indices_numpy, indices_freg, feature_names)

    # --- 4. Random Forest (all features) ---
    print(f"\n{'=' * 50}")
    print("Training: Random Forest (all features)")
    print("=" * 50)
    rf_all_model = train_random_forest(X_train_scaled, y_train, random_state=RANDOM_SEED)
    y_pred_rf_all = rf_all_model.predict(X_test_scaled)
    mse_rf_all = compute_mse(y_test, y_pred_rf_all)
    mae_rf_all = compute_mae(y_test, y_pred_rf_all)
    rmse_rf_all = np.sqrt(mse_rf_all)
    print(f"RF (all features) MSE:  {mse_rf_all:.4f}")
    print(f"RF (all features) MAE:  {mae_rf_all:.4f}")
    print(f"RF (all features) RMSE: {rmse_rf_all:.4f}")
    print(f"RF (all features) R2:   {compute_r2(y_test, y_pred_rf_all):.4f}")
    print_feature_importance(rf_all_model, feature_names)

    # --- 5. Random Forest (NumPy Pearson features) ---
    print(f"\n{'=' * 50}")
    print("Training: Random Forest (NumPy Pearson features)")
    print("=" * 50)
    rf_np_model = train_random_forest(X_train_scaled[:, indices_numpy], y_train, random_state=RANDOM_SEED)
    y_pred_rf_np = rf_np_model.predict(X_test_scaled[:, indices_numpy])
    mse_rf_numpy = compute_mse(y_test, y_pred_rf_np)
    mae_rf_numpy = compute_mae(y_test, y_pred_rf_np)
    r2_rf_numpy = compute_r2(y_test, y_pred_rf_np)
    print(f"RF (NumPy Pearson) MSE: {mse_rf_numpy:.4f}")
    print(f"RF (NumPy Pearson) MAE: {mae_rf_numpy:.4f}")
    print(f"RF (NumPy Pearson) R2:  {r2_rf_numpy:.4f}")
    print_feature_importance(rf_np_model, [feature_names[i] for i in indices_numpy])

    # --- 6. Random Forest (F-Regression features) ---
    print(f"\n{'=' * 50}")
    print("Training: Random Forest (F-Regression features)")
    print("=" * 50)
    rf_freg_model = train_random_forest(X_train_scaled[:, indices_freg], y_train, random_state=RANDOM_SEED)
    y_pred_rf_freg = rf_freg_model.predict(X_test_scaled[:, indices_freg])
    mse_rf_freg = compute_mse(y_test, y_pred_rf_freg)
    mae_rf_freg = compute_mae(y_test, y_pred_rf_freg)
    r2_rf_freg = compute_r2(y_test, y_pred_rf_freg)
    print(f"RF (F-Regression) MSE: {mse_rf_freg:.4f}")
    print(f"RF (F-Regression) MAE: {mae_rf_freg:.4f}")
    print(f"RF (F-Regression) R2:  {r2_rf_freg:.4f}")
    print_feature_importance(rf_freg_model, [feature_names[i] for i in indices_freg])

    # --- 7. Charts ---
    os.makedirs("output", exist_ok=True)

    mse_dict_all = {
        "Linear Regression\n(all features)": mse_lr,
        "Random Forest\n(all features)": mse_rf_all,
        "Random Forest\n(NumPy Pearson)": mse_rf_numpy,
        "Random Forest\n(F-Regression)": mse_rf_freg,
    }
    plot_mse_comparison(mse_dict_all, save_path="output/mse_comparison.png",
                        title="All Models: Test Set MSE Comparison")

    fs_mse_dict = {
        "NumPy Pearson": mse_rf_numpy,
        "F-Regression": mse_rf_freg,
    }
    plot_mse_comparison(fs_mse_dict, save_path="output/feature_selection_mse.png",
                        title="Feature Selection Method vs. Test Set MSE")

    # --- 8. Summary ---
    print("\n" + "=" * 85)
    print("Basic Layer: Model Performance Summary")
    print("=" * 85)
    print(f"{'Model':<35} {'MSE':<12} {'RMSE':<12} {'R2':<12}")
    print("-" * 75)
    print(f"{'Linear Regression (all)':<35} {mse_lr:<12.4f} {rmse_lr:<12.4f} {compute_r2(y_test, y_pred_lr):<12.4f}")
    print(f"{'Random Forest (all)':<35} {mse_rf_all:<12.4f} {rmse_rf_all:<12.4f} {compute_r2(y_test, y_pred_rf_all):<12.4f}")
    print(f"{'RF (NumPy Pearson 5)':<35} {mse_rf_numpy:<12.4f} {np.sqrt(mse_rf_numpy):<12.4f} {r2_rf_numpy:<12.4f}")
    print(f"{'RF (F-Regression 5)':<35} {mse_rf_freg:<12.4f} {np.sqrt(mse_rf_freg):<12.4f} {r2_rf_freg:<12.4f}")
    print("=" * 85)

    best_basic_mse = min(mse_rf_all, mse_rf_numpy, mse_rf_freg)
    print(f"\nBest RF MSE: {best_basic_mse:.4f}")
    print(f"vs. Linear Regression: MSE reduced by "
          f"{(mse_lr - best_basic_mse) / mse_lr * 100:.1f}%")

    # --- 9. Save results ---
    with open("output/baseline_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Model Performance Comparison\n")
        f.write("=" * 50 + "\n")
        f.write(f"Linear Regression (all)    MSE: {mse_lr:.6f}   RMSE: {rmse_lr:.6f}   R2: {compute_r2(y_test, y_pred_lr):.6f}\n")
        f.write(f"Random Forest (all)        MSE: {mse_rf_all:.6f}   RMSE: {rmse_rf_all:.6f}   R2: {compute_r2(y_test, y_pred_rf_all):.6f}\n")
        f.write(f"RF (NumPy Pearson 5)       MSE: {mse_rf_numpy:.6f}   R2: {r2_rf_numpy:.6f}\n")
        f.write(f"RF (F-Regression 5)        MSE: {mse_rf_freg:.6f}   R2: {r2_rf_freg:.6f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nLinear Regression Coefficients:\n")
        for name, coef in zip(feature_names, lr_model.coef_):
            f.write(f"  {name:15s}: {coef:.6f}\n")

    print("\n[OK] Results saved to output/baseline_results.txt")

    # ==================== Advanced Layer ====================
    print("\n" + "=" * 60)
    print("Advanced Layer")
    print("=" * 60)

    advanced_results = advanced.main(data={
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'random_state': RANDOM_SEED,
    })

    # ==================== Extended Layer ====================
    print("\n" + "=" * 60)
    print("Extended Layer: Geospatial Heatmap & Outlier Analysis")
    print("=" * 60)

    # --- Geospatial heatmap (10x10 grid) ---
    lat_idx = feature_names.index('Latitude')
    lon_idx = feature_names.index('Longitude')
    _, heatmap_vmax = plot_geospatial_heatmap(
        X_test, y_test, y_pred_rf_all,
        lat_idx, lon_idx,
        grid_size=10,
        save_path="output/geospatial_heatmap.png",
    )

    # --- Outlier detection & analysis ---
    outlier_mask = detect_outliers_iqr(y, factor=1.5)
    analyze_outlier_features(X, y, outlier_mask, feature_names)

    # --- Determine best model across Basic + Advanced ---
    all_model_metrics = {"RF (all features)": (mse_rf_all, mae_rf_all)}
    if advanced_results and 'all_results' in advanced_results:
        for name, (mse_val, mae_val, _) in advanced_results['all_results'].items():
            all_model_metrics[name] = (mse_val, mae_val)

    best_name = min(all_model_metrics, key=lambda k: all_model_metrics[k][0])
    best_mse_before, best_mae_before = all_model_metrics[best_name]

    print(f"\nBest model overall: {best_name} (MSE = {best_mse_before:.4f})")
    print(f"Retraining after removing outliers...")

    mse_after, mae_after, X_clean_test, y_clean_test, y_clean_pred = \
        retrain_best_model_after_outlier_removal(
            X, y, best_name, outlier_mask, random_state=RANDOM_SEED
        )

    print(f"\n{'=' * 55}")
    print(f"Outlier Removal: Performance Comparison")
    print(f"{'=' * 55}")
    print(f"Model: {best_name}")
    print(f"  Before: MSE = {best_mse_before:.4f}, MAE = {best_mae_before:.4f}")
    print(f"  After:  MSE = {mse_after:.4f}, MAE = {mae_after:.4f}")
    mse_change = (best_mse_before - mse_after) / best_mse_before * 100
    mae_change = (best_mae_before - mae_after) / best_mae_before * 100
    print(f"  MSE change: {mse_change:+.2f}%")
    print(f"  MAE change: {mae_change:+.2f}%")

    plot_outlier_removal_comparison(
        best_mse_before, best_mae_before, mse_after, mae_after,
        best_name, save_path="output/outlier_removal_comparison.png",
    )

    # --- Geospatial heatmap after outlier removal (same color scale) ---
    plot_geospatial_heatmap(
        X_clean_test, y_clean_test, y_clean_pred,
        lat_idx, lon_idx,
        grid_size=10,
        save_path="output/geospatial_heatmap_after_outlier_removal.png",
        vmin=0, vmax=heatmap_vmax,
    )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
