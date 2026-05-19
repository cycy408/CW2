import numpy as np
from sklearn.ensemble import RandomForestRegressor


# ========== sklearn ==========
def train_random_forest(X_train, y_train, random_state=None):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def print_feature_importance(model, feature_names, verbose=True):
    if not verbose:
        return
    importances = model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    print(f"  {'Feature':<15}  Importance")
    print(f"  {'-' * 25}")
    for i in idx_sorted:
        bar = "#" * int(importances[i] * 50)
        print(f"  {feature_names[i]:<15}  {importances[i]:.4f}  {bar}")


