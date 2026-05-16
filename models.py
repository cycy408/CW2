import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X_train, y_train, random_state=None):
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def print_feature_importance(model, feature_names):
    importances = model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    print(f"  {'Feature':<15}  Importance")
    print(f"  {'-' * 25}")
    for i in idx_sorted:
        bar = "#" * int(importances[i] * 50)
        print(f"  {feature_names[i]:<15}  {importances[i]:.4f}  {bar}")


# ========== implementation ==========
def fuse_predictions(pred_list, weights=None):
    if weights is None:
        weights = [0.2, 0.4, 0.4]
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    result = np.zeros(len(pred_list[0]), dtype=float)
    for w, p in zip(weights, pred_list):
        result += w * np.asarray(p)
    return result
