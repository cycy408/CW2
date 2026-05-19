import numpy as np
from sklearn.linear_model import LinearRegression

# ============================================================
# RANDOM SEED CONFIGURATION
#   Set to an integer (e.g. 42) → fixed seed, results reproducible
#   Set to None                  → random seed, results vary each run
#   NOTE: When called from main.py, the seed is passed via data dict.
# ============================================================
RANDOM_SEED = 42

from data_prep import load_data, split_data, standardize_features
from feature_selection import select_features_numpy, select_features_sklearn_freg, compare_feature_sets
from models import train_random_forest, print_feature_importance
from model_fusion import fuse_predictions
from evaluation import compute_mse, compute_mae, compute_r2



def main(data=None):
    if data is None:
        random_state = RANDOM_SEED
        X, y, feature_names = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_SEED)
        X_train_scaled, X_test_scaled, _ = standardize_features(X_train, X_test)
    else:
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        X_train_scaled = data['X_train_scaled']
        X_test_scaled = data['X_test_scaled']
        feature_names = data['feature_names']
        random_state = data.get('random_state', RANDOM_SEED)

    # --- Feature selection ---
    print("=" * 50)
    print("Feature Selection A: NumPy Pearson Correlation")
    print("=" * 50)
    indices_np, scores_np = select_features_numpy(X_train, y_train, k=5)
    print(f"Selected indices: {indices_np}")
    print(f"Feature names:    {[feature_names[i] for i in indices_np]}")
    print(f"Pearson scores:   {scores_np}")

    print(f"\n{'=' * 50}")
    print("Feature Selection B: sklearn SelectKBest + F-Regression")
    print("=" * 50)
    indices_skfreg, scores_skfreg = select_features_sklearn_freg(X_train, y_train, k=5)
    print(f"Selected indices: {indices_skfreg}")
    print(f"Feature names:    {[feature_names[i] for i in indices_skfreg]}")
    print(f"F-scores:         {scores_skfreg}")

    compare_feature_sets(indices_np, indices_skfreg, feature_names)

    # --- Feature subsets ---
    X_train_np = X_train_scaled[:, indices_np]
    X_test_np = X_test_scaled[:, indices_np]
    X_train_skfreg = X_train_scaled[:, indices_skfreg]
    X_test_skfreg = X_test_scaled[:, indices_skfreg]

    # --- Train models ---
    print(f"\n{'=' * 50}")
    print("Training: Linear Regression (all features)")
    print("=" * 50)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    print(f"{'=' * 50}")
    print("Training: Random Forest (all features)")
    print("=" * 50)
    rf_all_model = train_random_forest(X_train_scaled, y_train, random_state=random_state)
    print("Feature Importance (all features):")
    print_feature_importance(rf_all_model, feature_names)

    print(f"\n{'=' * 50}")
    print("Training: Random Forest (NumPy Pearson features)")
    print("=" * 50)
    rf_np_model = train_random_forest(X_train_np, y_train, random_state=random_state)
    print("Feature Importance (NumPy Pearson features):")
    print_feature_importance(rf_np_model, [feature_names[i] for i in indices_np])

    print(f"\n{'=' * 50}")
    print("Training: Random Forest (sklearn F-Regression features)")
    print("=" * 50)
    rf_skfreg_model = train_random_forest(X_train_skfreg, y_train, random_state=random_state)
    print("Feature Importance (sklearn F-Regression features):")
    print_feature_importance(rf_skfreg_model, [feature_names[i] for i in indices_skfreg])

    # --- Predictions ---
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_rf_all = rf_all_model.predict(X_test_scaled)
    y_pred_rf_np = rf_np_model.predict(X_test_np)
    y_pred_rf_skfreg = rf_skfreg_model.predict(X_test_skfreg)

    # --- Weighted fusion ---
    # Weights: LR gets lowest (0.10) because linear model underfits nonlinear housing data;
    # RF (all features) gets highest (0.35) because it performs best individually;
    # RF (NumPy Pearson) = 0.275, RF (sklearn F-Reg) = 0.275
    # (the two feature-selected RFs share equal weight as their MSEs are close).
    y_pred_fused = fuse_predictions(
        [y_pred_lr, y_pred_rf_all, y_pred_rf_np, y_pred_rf_skfreg],
        weights=[0.10, 0.35, 0.275, 0.275],
    )

    # --- Performance comparison ---
    results = {
        "Linear Regression (all features)": (
            compute_mse(y_test, y_pred_lr), compute_mae(y_test, y_pred_lr),
            compute_r2(y_test, y_pred_lr)),
        "RF (all features)": (
            compute_mse(y_test, y_pred_rf_all), compute_mae(y_test, y_pred_rf_all),
            compute_r2(y_test, y_pred_rf_all)),
        "RF (NumPy Pearson features)": (
            compute_mse(y_test, y_pred_rf_np), compute_mae(y_test, y_pred_rf_np),
            compute_r2(y_test, y_pred_rf_np)),
        "RF (sklearn F-Reg features)": (
            compute_mse(y_test, y_pred_rf_skfreg), compute_mae(y_test, y_pred_rf_skfreg),
            compute_r2(y_test, y_pred_rf_skfreg)),
        "Weighted Fusion": (
            compute_mse(y_test, y_pred_fused), compute_mae(y_test, y_pred_fused),
            compute_r2(y_test, y_pred_fused)),
    }

    print("\n" + "=" * 80)
    print("Advanced Layer: Model Performance Comparison")
    print("=" * 80)
    print(f"{'Model':<40} {'MSE':>10} {'MAE':>10} {'R2':>10}")
    print("-" * 72)
    for name, (mse, mae, r2) in results.items():
        print(f"{name:<40} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
    print("=" * 80)

    # Save results to text file
    with open("output/advanced_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Advanced Layer: Single Models vs Weighted Fusion\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Model':<40} {'MSE':>10} {'MAE':>10} {'R2':>10}\n")
        f.write("-" * 70 + "\n")
        for name, (mse, mae, r2) in results.items():
            f.write(f"{name:<40} {mse:>10.6f} {mae:>10.6f} {r2:>10.6f}\n")
        f.write("=" * 70 + "\n")

    print("\n[OK] Results saved to output/advanced_results.txt")

    best_name = min(results, key=lambda k: results[k][0])
    best_mse, best_mae, _ = results[best_name]
    print(f"\nBest model: {best_name} (MSE = {best_mse:.4f}, MAE = {best_mae:.4f})")

    return {
        'best_model_name': best_name,
        'best_mse': best_mse,
        'best_mae': best_mae,
        'all_results': results,
    }


if __name__ == "__main__":
    main()
