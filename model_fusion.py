import numpy as np
from sklearn.linear_model import LinearRegression

from models import train_random_forest
from feature_selection import select_features_numpy, select_features_sklearn_freg


def train_and_fuse_models(X_train, X_train_scaled, X_test_scaled, y_train,
                          random_state=None, weights=None, verbose=True):
    """Train 4 base models and return their predictions + weighted fusion result.

    Models: LinearRegression (all), RF (all), RF (NumPy Pearson top-5),
            RF (sklearn F-Reg top-5).
    """
    if weights is None:
        weights = [0.10, 0.35, 0.275, 0.275]

    indices_np, _ = select_features_numpy(X_train, y_train, k=5)
    indices_skfreg, _ = select_features_sklearn_freg(X_train, y_train, k=5)

    if verbose:
        print("  Training base models for fusion...")

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    pred_lr = lr.predict(X_test_scaled)

    rf_all = train_random_forest(X_train_scaled, y_train, random_state=random_state)
    pred_rf_all = rf_all.predict(X_test_scaled)

    rf_np = train_random_forest(
        X_train_scaled[:, indices_np], y_train, random_state=random_state)
    pred_rf_np = rf_np.predict(X_test_scaled[:, indices_np])

    rf_skfreg = train_random_forest(
        X_train_scaled[:, indices_skfreg], y_train, random_state=random_state)
    pred_rf_skfreg = rf_skfreg.predict(X_test_scaled[:, indices_skfreg])

    # Weighted fusion
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    pred_fused = (w[0] * np.asarray(pred_lr) + w[1] * np.asarray(pred_rf_all) +
                  w[2] * np.asarray(pred_rf_np) + w[3] * np.asarray(pred_rf_skfreg))

    return {
        'pred_lr': pred_lr,
        'pred_rf_all': pred_rf_all,
        'pred_rf_np': pred_rf_np,
        'pred_rf_skfreg': pred_rf_skfreg,
        'pred_fused': pred_fused,
        'indices_np': indices_np,
        'indices_skfreg': indices_skfreg,
        'weights': weights,
    }
