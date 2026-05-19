import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from models import train_random_forest
from model_fusion import fuse_predictions
from feature_selection import select_features_numpy, select_features_sklearn_freg
from evaluation import compute_mse, compute_mae


# ========== NumPy-only implementation ==========
def detect_outliers_iqr(y, factor=1.5):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    return (y < q1 - factor * iqr) | (y > q3 + factor * iqr)


# ========== NumPy-only implementation ==========
def analyze_outlier_features(X, y, outlier_mask, feature_names, verbose=True):
    outliers_X = X[outlier_mask]
    normal_X = X[~outlier_mask]
    outliers_y = y[outlier_mask]
    normal_y = y[~outlier_mask]

    if verbose:
        print("\n--- Outlier Analysis ---")
        print(f"Outliers: {np.sum(outlier_mask)} ({np.mean(outlier_mask) * 100:.2f}%)")
        print(f"Outlier target range: [{outliers_y.min():.4f}, {outliers_y.max():.4f}]")
        print(f"Normal  target range: [{normal_y.min():.4f}, {normal_y.max():.4f}]")
        print(f"\nFeature mean comparison:")
        print(f"{'Feature':<15} {'Normal':>12} {'Outlier':>12}")
        print("-" * 42)
        for i, name in enumerate(feature_names):
            print(f"{name:<15} {np.mean(normal_X[:, i]):>12.4f} {np.mean(outliers_X[:, i]):>12.4f}")

    return outliers_X, normal_X


# ========== sklearn (StandardScaler + models) + NumPy fusion ==========
def retrain_best_model_after_outlier_removal(X, y, best_model_name, outlier_mask,
                                             test_size=0.2, random_state=None,
                                             verbose=True):
    X_clean = X[~outlier_mask]
    y_clean = y[~outlier_mask]

    if verbose:
        print(f"  Samples before: {len(y)}, after: {len(y_clean)} "
              f"(removed {np.sum(outlier_mask)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if "Weighted Fusion" in best_model_name:
        indices_np, _ = select_features_numpy(X_train, y_train, k=5)
        indices_skfreg, _ = select_features_sklearn_freg(X_train, y_train, k=5)

        lr = LinearRegression()
        lr.fit(X_train_s, y_train)
        pred_lr = lr.predict(X_test_s)

        rf_all = train_random_forest(X_train_s, y_train, random_state=random_state)
        pred_rf_all = rf_all.predict(X_test_s)

        rf_np = train_random_forest(X_train_s[:, indices_np], y_train, random_state=random_state)
        pred_rf_np = rf_np.predict(X_test_s[:, indices_np])

        rf_skfreg = train_random_forest(X_train_s[:, indices_skfreg], y_train, random_state=random_state)
        pred_rf_skfreg = rf_skfreg.predict(X_test_s[:, indices_skfreg])

        y_pred = fuse_predictions(
            [pred_lr, pred_rf_all, pred_rf_np, pred_rf_skfreg],
            weights=[0.10, 0.35, 0.275, 0.275],
        )

    elif "sklearn" in best_model_name.lower():
        indices_skfreg, _ = select_features_sklearn_freg(X_train, y_train, k=5)
        model = train_random_forest(X_train_s[:, indices_skfreg], y_train, random_state=random_state)
        y_pred = model.predict(X_test_s[:, indices_skfreg])

    elif "NumPy Pearson" in best_model_name:
        indices_np, _ = select_features_numpy(X_train, y_train, k=5)
        model = train_random_forest(X_train_s[:, indices_np], y_train, random_state=random_state)
        y_pred = model.predict(X_test_s[:, indices_np])

    elif "all" in best_model_name.lower():
        model = train_random_forest(X_train_s, y_train, random_state=random_state)
        y_pred = model.predict(X_test_s)

    else:
        model = train_random_forest(X_train_s, y_train, random_state=random_state)
        y_pred = model.predict(X_test_s)

    mse = compute_mse(y_test, y_pred)
    mae = compute_mae(y_test, y_pred)
    return mse, mae, X_test, y_test, y_pred
