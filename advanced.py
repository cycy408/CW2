from data_prep import load_data, split_data, standardize_features
from model_fusion import train_and_fuse_models
from evaluation import compute_mse, compute_mae, compute_r2
from visualization import plot_fusion_vs_single

RANDOM_SEED = 42



def main(data=None):
    verbose = data.get('verbose', True) if data else True

    if data is None:
        random_state = RANDOM_SEED
        X, y, feature_names = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_SEED)
        X_train_scaled, X_test_scaled, _ = standardize_features(X_train, X_test)
        precomputed_predictions = None
    else:
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        X_train_scaled = data['X_train_scaled']
        X_test_scaled = data['X_test_scaled']
        feature_names = data['feature_names']
        random_state = data.get('random_state', RANDOM_SEED)
        precomputed_predictions = data.get('predictions', None)

    # If predictions are pre-computed (called from main.py), skip directly to fusion
    if precomputed_predictions is not None:
        y_pred_lr, y_pred_rf_all, y_pred_rf_np, y_pred_rf_skfreg = precomputed_predictions
        y_pred_fused = (0.10 * y_pred_lr + 0.35 * y_pred_rf_all +
                        0.275 * y_pred_rf_np + 0.275 * y_pred_rf_skfreg)
    else:
        fusion_result = train_and_fuse_models(
            X_train, X_train_scaled, X_test_scaled, y_train,
            random_state=random_state, verbose=verbose,
        )
        y_pred_lr = fusion_result['pred_lr']
        y_pred_rf_all = fusion_result['pred_rf_all']
        y_pred_rf_np = fusion_result['pred_rf_np']
        y_pred_rf_skfreg = fusion_result['pred_rf_skfreg']
        y_pred_fused = fusion_result['pred_fused']

    # --- Performance comparison ---
    results = {
        "Linear Regression (all)": (
            compute_mse(y_test, y_pred_lr), compute_mae(y_test, y_pred_lr),
            compute_r2(y_test, y_pred_lr)),
        "RF (all features)": (
            compute_mse(y_test, y_pred_rf_all), compute_mae(y_test, y_pred_rf_all),
            compute_r2(y_test, y_pred_rf_all)),
        "RF (NumPy Pearson)": (
            compute_mse(y_test, y_pred_rf_np), compute_mae(y_test, y_pred_rf_np),
            compute_r2(y_test, y_pred_rf_np)),
        "RF (sklearn F-Reg)": (
            compute_mse(y_test, y_pred_rf_skfreg), compute_mae(y_test, y_pred_rf_skfreg),
            compute_r2(y_test, y_pred_rf_skfreg)),
        "Weighted Fusion": (
            compute_mse(y_test, y_pred_fused), compute_mae(y_test, y_pred_fused),
            compute_r2(y_test, y_pred_fused)),
    }

    if verbose:
        print("\n" + "=" * 80)
        print("Advanced Layer: Model Performance Comparison")
        print("=" * 80)
        print(f"{'Model':<40} {'MSE':>10} {'MAE':>10} {'R2':>10}")
        print("-" * 72)
        for name, (mse, mae, r2) in results.items():
            print(f"{name:<40} {mse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
        print("=" * 80)

    # --- Compare single models vs fusion ---
    single_vs_fusion = {name: (mse, mae) for name, (mse, mae, _) in results.items()}
    plot_fusion_vs_single(
        single_vs_fusion,
        save_path="output/fusion_vs_single.png",
        verbose=verbose,
    )

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

    if verbose:
        print("[OK] Results saved to output/advanced_results.txt")

    best_name = min(results, key=lambda k: results[k][0])
    best_mse, best_mae, _ = results[best_name]
    if verbose:
        print(f"Best model: {best_name} (MSE = {best_mse:.4f}, MAE = {best_mae:.4f})")

    return {
        'best_model_name': best_name,
        'best_mse': best_mse,
        'best_mae': best_mae,
        'all_results': results,
    }


if __name__ == "__main__":
    main()
