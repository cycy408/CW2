import os
import numpy as np
from sklearn.linear_model import LinearRegression

# ============================================================
# RANDOM SEED CONFIGURATION
#   Set to an integer (e.g. 42) → fixed seed, results reproducible
#   Set to None                  → random seed, results vary each run
# ============================================================
RANDOM_SEED = 42

from data_prep import load_data, split_data, fill_missing_values, standardize_features
from feature_selection import select_features_numpy, select_features_sklearn_freg, compare_feature_sets
from models import train_random_forest, print_feature_importance
from evaluation import compute_mse, compute_mae, compute_r2
from visualization import plot_mse_comparison, plot_geospatial_heatmap, plot_outlier_removal_comparison
from analysis import detect_outliers_iqr, analyze_outlier_features, retrain_best_model_after_outlier_removal
import advanced


def main():
    print("\n" + "=" * 60)
    print("INT101 California Housing Price Prediction")
    print("=" * 60)

    # ==================== Basic Layer: Data Preprocessing ====================
    print("\n" + "=" * 60)
    print("  Basic Layer - 1. Data Loading & Preprocessing")
    print("=" * 60)

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_SEED)
    X_train, X_test = fill_missing_values(X_train, X_test)
    X_train_scaled, X_test_scaled, _ = standardize_features(X_train, X_test)

    # ==================== Basic Layer: Feature Selection ====================
    print("=" * 60)
    print("  Basic Layer - 2. Feature Selection")
    print("=" * 60)

    print("\n--- Method A: NumPy Pearson Correlation (manual implementation) ---")
    indices_numpy, scores_numpy = select_features_numpy(X_train, y_train, k=5)
    print(f"  Selected indices: {indices_numpy}")
    print(f"  Feature names:    {[feature_names[i] for i in indices_numpy]}")
    print(f"  Pearson r:        {[f'{v:+.4f}' for v in scores_numpy]}")

    print("\n--- Method B: sklearn SelectKBest + F-Regression ---")
    indices_skfreg, scores_skfreg = select_features_sklearn_freg(X_train, y_train, k=5)
    print(f"  Selected indices: {indices_skfreg}")
    print(f"  Feature names:    {[feature_names[i] for i in indices_skfreg]}")
    print(f"  F-scores:         {[f'{v:.2f}' for v in scores_skfreg]}")

    compare_feature_sets(indices_numpy, indices_skfreg, feature_names)

    # ==================== Basic Layer: Model Training ====================
    print("\n" + "=" * 60)
    print("  Basic Layer - 3. Model Training")
    print("=" * 60)

    # --- Linear Regression ---
    print("\n[1/4] Training Linear Regression (all features)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    print(f"  Intercept: {lr_model.intercept_:.6f}")
    for name, coef in zip(feature_names, lr_model.coef_):
        print(f"    {name:15s}: {coef:+.6f}")

    # --- Random Forest (all features) ---
    print("\n[2/4] Training Random Forest (all features)...")
    rf_all_model = train_random_forest(X_train_scaled, y_train, random_state=RANDOM_SEED)
    y_pred_rf_all = rf_all_model.predict(X_test_scaled)
    print("  Feature Importance:")
    print_feature_importance(rf_all_model, feature_names)

    # --- Random Forest (NumPy Pearson top-5) ---
    print("[3/4] Training Random Forest (NumPy Pearson features)")
    rf_np_model = train_random_forest(
        X_train_scaled[:, indices_numpy], y_train, random_state=RANDOM_SEED)
    y_pred_rf_np = rf_np_model.predict(X_test_scaled[:, indices_numpy])
    names_np = [feature_names[i] for i in indices_numpy]
    print("  Feature Importance (NumPy Pearson top-5):")
    print_feature_importance(rf_np_model, names_np)

    # --- Random Forest (sklearn F-Reg top-5) ---
    print("[4/4] Training Random Forest (sklearn F-Reg features)")
    rf_skfreg_model = train_random_forest(
        X_train_scaled[:, indices_skfreg], y_train, random_state=RANDOM_SEED)
    y_pred_rf_skfreg = rf_skfreg_model.predict(X_test_scaled[:, indices_skfreg])
    names_sk = [feature_names[i] for i in indices_skfreg]
    print("  Feature Importance (sklearn F-Reg top-5):")
    print_feature_importance(rf_skfreg_model, names_sk)

    # Compute metrics helper
    def metrics(y_true, y_pred):
        mse = compute_mse(y_true, y_pred)
        return mse, np.sqrt(mse), compute_mae(y_true, y_pred), compute_r2(y_true, y_pred)

    mse_lr,   rmse_lr,   mae_lr,   r2_lr   = metrics(y_test, y_pred_lr)
    mse_rf_all, rmse_rf_all, mae_rf_all, r2_rf_all = metrics(y_test, y_pred_rf_all)
    mse_rf_np,  rmse_rf_np,  mae_rf_np,  r2_rf_np  = metrics(y_test, y_pred_rf_np)
    mse_rf_sk,  rmse_rf_sk,  mae_rf_sk,  r2_rf_sk  = metrics(y_test, y_pred_rf_skfreg)

    # ==================== Basic Layer: Summary ====================
    print("\n" + "=" * 60)
    print("  Basic Layer - 4. Single-Model Performance Summary")
    print("=" * 60)
    print(f"{'Model':<35} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R2':<10}")
    print("-" * 75)
    print(f"{'Linear Regression (all)':<35} {mse_lr:<10.4f} {rmse_lr:<10.4f} {mae_lr:<10.4f} {r2_lr:<10.4f}")
    print(f"{'Random Forest (all)':<35} {mse_rf_all:<10.4f} {rmse_rf_all:<10.4f} {mae_rf_all:<10.4f} {r2_rf_all:<10.4f}")
    print(f"{'RF (NumPy Pearson 5)':<35} {mse_rf_np:<10.4f} {rmse_rf_np:<10.4f} {mae_rf_np:<10.4f} {r2_rf_np:<10.4f}")
    print(f"{'RF (sklearn F-Reg 5)':<35} {mse_rf_sk:<10.4f} {rmse_rf_sk:<10.4f} {mae_rf_sk:<10.4f} {r2_rf_sk:<10.4f}")
    print("-" * 75)

    best_basic_mse = min(mse_rf_all, mse_rf_np, mse_rf_sk)
    print(f"  Best RF MSE: {best_basic_mse:.4f} "
          f"(vs. Linear Regression: {(mse_lr - best_basic_mse) / mse_lr * 100:.1f}% reduction)")

    # --- Basic layer charts ---
    os.makedirs("output", exist_ok=True)
    plot_mse_comparison(
        {"LR (all)": mse_lr, "RF (all)": mse_rf_all,
         "RF (NP)": mse_rf_np, "RF (F-Reg)": mse_rf_sk},
        save_path="output/mse_comparison.png",
        title="Basic Layer: Test Set MSE Comparison",
    )
    plot_mse_comparison(
        {"NumPy Pearson": mse_rf_np, "sklearn F-Reg": mse_rf_sk},
        save_path="output/feature_selection_mse.png",
        title="Feature Selection: NumPy Pearson vs sklearn F-Reg (MSE)",
    )

    # ==================== Advanced Layer: Weighted Fusion ====================
    print("\n" + "=" * 60)
    print("  Advanced Layer - Model Fusion (Weighted Fusion)")
    print("=" * 60)

    advanced_results = advanced.main(data={
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'random_state': RANDOM_SEED,
        'predictions': (y_pred_lr, y_pred_rf_all, y_pred_rf_np, y_pred_rf_skfreg),
        'verbose': False,
    })

    fusion_entry = advanced_results['all_results']['Weighted Fusion']
    mse_fusion, mae_fusion, r2_fusion = fusion_entry
    rmse_fusion = np.sqrt(mse_fusion)

    print(f"\n  Fusion weights: LR=0.10, RF(all)=0.35, RF(NP)=0.275, RF(F-Reg)=0.275")
    print(f"  Weighted Fusion  MSE: {mse_fusion:.4f}  RMSE: {rmse_fusion:.4f}  "
          f"MAE: {mae_fusion:.4f}  R2: {r2_fusion:.4f}")

    # ==================== Extended Layer ====================
    print("\n" + "=" * 60)
    print("  Extended Layer - Geospatial Visualization & Outlier Analysis")
    print("=" * 60)

    # --- Geospatial heatmap ---
    print("\n--- Geospatial Prediction Error Heatmap ---")
    lat_idx = feature_names.index('Latitude')
    lon_idx = feature_names.index('Longitude')
    _, heatmap_vmax = plot_geospatial_heatmap(
        X_test, y_test, y_pred_rf_all,
        lat_idx, lon_idx, grid_size=10,
        save_path="output/geospatial_heatmap.png",
    )

    # --- Outlier detection ---
    print("\n--- Outlier Detection & Analysis ---")
    outlier_mask = detect_outliers_iqr(y, factor=1.5)
    analyze_outlier_features(X, y, outlier_mask, feature_names)

    # --- Retrain after outlier removal ---
    print("\n--- Retraining Best Model After Outlier Removal ---")
    all_model_metrics = {
        "Linear Regression (all)": (mse_lr, mae_lr),
        "RF (all features)": (mse_rf_all, mae_rf_all),
        "RF (NumPy Pearson)": (mse_rf_np, mae_rf_np),
        "RF (sklearn F-Reg)": (mse_rf_sk, mae_rf_sk),
        "Weighted Fusion": (mse_fusion, mae_fusion),
    }
    best_name = min(all_model_metrics, key=lambda k: all_model_metrics[k][0])
    best_mse_before, best_mae_before = all_model_metrics[best_name]

    print(f"  Overall best model: {best_name} (MSE = {best_mse_before:.4f})")

    mse_after, mae_after, X_clean_test, y_clean_test, y_clean_pred = \
        retrain_best_model_after_outlier_removal(
            X, y, best_name, outlier_mask, random_state=RANDOM_SEED
        )

    mse_change = (best_mse_before - mse_after) / best_mse_before * 100
    mae_change = (best_mae_before - mae_after) / best_mae_before * 100

    print(f"  Before removal: MSE = {best_mse_before:.4f}, MAE = {best_mae_before:.4f}")
    print(f"  After removal:  MSE = {mse_after:.4f}, MAE = {mae_after:.4f}")
    print(f"  MSE change: {mse_change:+.2f}%")
    print(f"  MAE change: {mae_change:+.2f}%")

    plot_outlier_removal_comparison(
        best_mse_before, best_mae_before, mse_after, mae_after,
        best_name, save_path="output/outlier_removal_comparison.png",
    )
    plot_geospatial_heatmap(
        X_clean_test, y_clean_test, y_clean_pred,
        lat_idx, lon_idx, grid_size=10,
        save_path="output/geospatial_heatmap_after_outlier_removal.png",
        vmin=0, vmax=heatmap_vmax,
    )

    # ==================== Final Overview ====================
    print("\n" + "=" * 90)
    print("  FINAL OVERVIEW - All Models Performance Comparison")
    print("=" * 90)
    print(f"{'Model':<30} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print("-" * 72)
    print(f"{'Linear Regression (all)':<30} {mse_lr:>10.4f} {rmse_lr:>10.4f} {mae_lr:>10.4f} {r2_lr:>10.4f}")
    print(f"{'Random Forest (all)':<30} {mse_rf_all:>10.4f} {rmse_rf_all:>10.4f} {mae_rf_all:>10.4f} {r2_rf_all:>10.4f}")
    print(f"{'RF (NumPy Pearson 5)':<30} {mse_rf_np:>10.4f} {rmse_rf_np:>10.4f} {mae_rf_np:>10.4f} {r2_rf_np:>10.4f}")
    print(f"{'RF (sklearn F-Reg 5)':<30} {mse_rf_sk:>10.4f} {rmse_rf_sk:>10.4f} {mae_rf_sk:>10.4f} {r2_rf_sk:>10.4f}")
    print(f"{'Weighted Fusion':<30} {mse_fusion:>10.4f} {rmse_fusion:>10.4f} {mae_fusion:>10.4f} {r2_fusion:>10.4f}")
    print("-" * 72)
    print(f"\n  Best model: {best_name} (MSE = {min(all_model_metrics.values(), key=lambda x: x[0])[0]:.4f})")
    print(f"  Outlier removal ({best_name}): MSE {mse_change:+.1f}%, MAE {mae_change:+.1f}%")
    print("=" * 90)

    # --- Save text results ---
    with open("output/baseline_results.txt", "w", encoding="utf-8") as f:
        f.write("Model Performance Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Model':<30} {'MSE':>10} {'RMSE':>10} {'R2':>10}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Linear Regression (all)':<30} {mse_lr:>10.6f} {rmse_lr:>10.6f} {r2_lr:>10.6f}\n")
        f.write(f"{'Random Forest (all)':<30} {mse_rf_all:>10.6f} {rmse_rf_all:>10.6f} {r2_rf_all:>10.6f}\n")
        f.write(f"{'RF (NumPy Pearson 5)':<30} {mse_rf_np:>10.6f} {rmse_rf_np:>10.6f} {r2_rf_np:>10.6f}\n")
        f.write(f"{'RF (sklearn F-Reg 5)':<30} {mse_rf_sk:>10.6f} {rmse_rf_sk:>10.6f} {r2_rf_sk:>10.6f}\n")
        f.write(f"{'Weighted Fusion':<30} {mse_fusion:>10.6f} {rmse_fusion:>10.6f} {r2_fusion:>10.6f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nLinear Regression Coefficients:\n")
        for name, coef in zip(feature_names, lr_model.coef_):
            f.write(f"  {name:15s}: {coef:.6f}\n")

    print("\n[OK] Results saved to output/baseline_results.txt")
    print("=== Pipeline Complete ===\n")


if __name__ == "__main__":
    main()
