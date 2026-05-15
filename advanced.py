import numpy as np
from sklearn.linear_model import LinearRegression

from data_prep import load_data, split_data, standardize_features
from feature_selection import select_features_numpy, select_features_fregression, compare_feature_sets
from models import train_random_forest, fuse_predictions, print_feature_importance
from evaluation import compute_mse, compute_mae, compute_r2
from visualization import plot_mse_comparison


def main(data=None):
    if data is None:
        X, y, feature_names = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, _ = standardize_features(X_train, X_test)
    else:
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        X_train_scaled = data['X_train_scaled']
        X_test_scaled = data['X_test_scaled']
        feature_names = data['feature_names']

    # --- Feature selection ---
    print("=" * 50)
    print("Feature Selection A: NumPy Pearson Correlation")
    print("=" * 50)
    indices_np, scores_np = select_features_numpy(X_train, y_train, k=5)
    print(f"Selected indices: {indices_np}")
    print(f"Feature names:    {[feature_names[i] for i in indices_np]}")
    print(f"Pearson scores:   {scores_np}")

    print(f"\n{'=' * 50}")
    print("Feature Selection B: F-Regression / F-Test (NumPy)")
    print("=" * 50)
    indices_freg, scores_freg = select_features_fregression(X_train, y_train, k=5)
    print(f"Selected indices: {indices_freg}")
    print(f"Feature names:    {[feature_names[i] for i in indices_freg]}")
    print(f"F-scores:         {scores_freg}")

    compare_feature_sets(indices_np, indices_freg, feature_names)

    # --- Feature subsets ---
    X_train_np = X_train_scaled[:, indices_np]
    X_test_np = X_test_scaled[:, indices_np]
    X_train_freg = X_train_scaled[:, indices_freg]
    X_test_freg = X_test_scaled[:, indices_freg]

    # --- Train models ---
    print(f"\n{'=' * 50}")
    print("Training: Linear Regression (all features)")
    print("=" * 50)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    print(f"{'=' * 50}")
    print("Training: Random Forest (NumPy Pearson features)")
    print("=" * 50)
    rf_np_model = train_random_forest(X_train_np, y_train)
    print("Feature Importance (NumPy Pearson features):")
    print_feature_importance(rf_np_model, [feature_names[i] for i in indices_np])

    print(f"\n{'=' * 50}")
    print("Training: Random Forest (F-Regression features)")
    print("=" * 50)
    rf_freg_model = train_random_forest(X_train_freg, y_train)
    print("Feature Importance (F-Regression features):")
    print_feature_importance(rf_freg_model, [feature_names[i] for i in indices_freg])

    # --- Predictions ---
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_rf_np = rf_np_model.predict(X_test_np)
    y_pred_rf_freg = rf_freg_model.predict(X_test_freg)

    # --- Weighted fusion (default weights: LR=0.2, RF_Pearson=0.4, RF_SKB=0.4) ---
    y_pred_fused = fuse_predictions([y_pred_lr, y_pred_rf_np, y_pred_rf_freg])

    # --- Performance comparison ---
    results = {
        "Linear Regression (all features)": (
            compute_mse(y_test, y_pred_lr), compute_mae(y_test, y_pred_lr),
            compute_r2(y_test, y_pred_lr)),
        "RF (NumPy Pearson features)": (
            compute_mse(y_test, y_pred_rf_np), compute_mae(y_test, y_pred_rf_np),
            compute_r2(y_test, y_pred_rf_np)),
        "RF (F-Regression features)": (
            compute_mse(y_test, y_pred_rf_freg), compute_mae(y_test, y_pred_rf_freg),
            compute_r2(y_test, y_pred_rf_freg)),
        "Weighted Fusion": (
            compute_mse(y_test, y_pred_fused), compute_mae(y_test, y_pred_fused),
            compute_r2(y_test, y_pred_fused)),
    }

    print("\n" + "=" * 80)
    print("Advanced Layer: Model Performance Comparison")
    print("=" * 80)
    print(f"{'Model':<40} {'MSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 72)
    for name, (mse, mae, r2) in results.items():
        print(f"{name:<40} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
    print("=" * 80)

    # --- Required chart: Feature Selection Method vs. Test Set MSE ---
    fs_mse_dict = {
        "NumPy Pearson": compute_mse(y_test, y_pred_rf_np),
        "F-Regression": compute_mse(y_test, y_pred_rf_freg),
    }
    plot_mse_comparison(
        fs_mse_dict,
        save_path="output/advanced_mse_comparison.png",
        title="Feature Selection Method vs. Test Set MSE",
    )

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
