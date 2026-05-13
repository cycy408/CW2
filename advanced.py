import numpy as np
from sklearn.linear_model import LinearRegression

from data_prep import load_data, split_data, standardize_features
from feature_selection import select_features_numpy, select_features_sklearn, compare_feature_sets
from models import train_random_forest, fuse_predictions
from evaluation import compute_mse, compute_mae
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
    print("Feature Selection B: SelectKBest + f_regression")
    print("=" * 50)
    indices_sk, scores_sk = select_features_sklearn(X_train, y_train, k=5)
    print(f"Selected indices: {indices_sk}")
    print(f"Feature names:    {[feature_names[i] for i in indices_sk]}")
    print(f"F-scores:         {scores_sk}")

    compare_feature_sets(indices_np, indices_sk, feature_names)

    # --- Feature subsets ---
    X_train_np = X_train_scaled[:, indices_np]
    X_test_np = X_test_scaled[:, indices_np]
    X_train_sk = X_train_scaled[:, indices_sk]
    X_test_sk = X_test_scaled[:, indices_sk]

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

    print(f"{'=' * 50}")
    print("Training: Random Forest (SelectKBest features)")
    print("=" * 50)
    rf_sk_model = train_random_forest(X_train_sk, y_train)

    # --- Predictions ---
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_rf_np = rf_np_model.predict(X_test_np)
    y_pred_rf_sk = rf_sk_model.predict(X_test_sk)

    # --- Weighted fusion (default weights: LR=0.2, RF_Pearson=0.4, RF_SKB=0.4) ---
    y_pred_fused = fuse_predictions([y_pred_lr, y_pred_rf_np, y_pred_rf_sk])

    # --- Performance comparison ---
    results = {
        "Linear Regression (all features)": (
            compute_mse(y_test, y_pred_lr), compute_mae(y_test, y_pred_lr)),
        "RF (NumPy Pearson features)": (
            compute_mse(y_test, y_pred_rf_np), compute_mae(y_test, y_pred_rf_np)),
        "RF (SelectKBest features)": (
            compute_mse(y_test, y_pred_rf_sk), compute_mae(y_test, y_pred_rf_sk)),
        "Weighted Fusion": (
            compute_mse(y_test, y_pred_fused), compute_mae(y_test, y_pred_fused)),
    }

    print("\n" + "=" * 62)
    print("Advanced Layer: Model Performance Comparison")
    print("=" * 62)
    print(f"{'Model':<40} {'MSE':>10} {'MAE':>10}")
    print("-" * 62)
    for name, (mse, mae) in results.items():
        print(f"{name:<40} {mse:>10.4f} {mae:>10.4f}")
    print("=" * 62)

    # --- Required chart: Feature Selection Method vs. Test Set MSE ---
    fs_mse_dict = {
        "NumPy Pearson": compute_mse(y_test, y_pred_rf_np),
        "SelectKBest": compute_mse(y_test, y_pred_rf_sk),
    }
    plot_mse_comparison(
        fs_mse_dict,
        save_path="output/advanced_mse_comparison.png",
        title="Feature Selection Method vs. Test Set MSE",
    )

    best_name = min(results, key=lambda k: results[k][0])
    best_mse, best_mae = results[best_name]
    print(f"\nBest model: {best_name} (MSE = {best_mse:.4f}, MAE = {best_mae:.4f})")

    return {
        'best_model_name': best_name,
        'best_mse': best_mse,
        'best_mae': best_mae,
        'all_results': results,
    }


if __name__ == "__main__":
    main()
