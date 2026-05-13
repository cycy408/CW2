import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression


# ========== NumPy-only implementation ==========
def pearson_correlation_numpy(X, y):
    n_features = X.shape[1]
    correlations = np.zeros(n_features)
    for i in range(n_features):
        x_i = X[:, i]
        x_centered = x_i - np.mean(x_i)
        y_centered = y - np.mean(y)
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
        if denominator != 0:
            correlations[i] = numerator / denominator
        else:
            correlations[i] = 0.0
    return correlations


def select_features_numpy(X, y, k=5):
    corr = pearson_correlation_numpy(X, y)
    abs_corr = np.abs(corr)
    selected_indices = np.argsort(abs_corr)[::-1][:k]
    scores = corr[selected_indices]
    return selected_indices, scores


def select_features_sklearn(X, y, k=5):
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_[selected_indices]
    return selected_indices, scores


def compare_feature_sets(indices_np, indices_sk, feature_names):
    set_np = set(indices_np)
    set_sk = set(indices_sk)
    intersection = set_np & set_sk
    only_np = set_np - set_sk
    only_sk = set_sk - set_np

    print("\n=== Feature Selection Comparison ===")
    print(f"NumPy Pearson top-5: {[feature_names[i] for i in indices_np]}")
    print(f"SelectKBest   top-5: {[feature_names[i] for i in indices_sk]}")
    print(f"Intersection:        {[feature_names[i] for i in intersection]}")
    print(f"Only in NumPy:       {[feature_names[i] for i in only_np]}")
    print(f"Only in SelectKBest: {[feature_names[i] for i in only_sk]}")
    print("=" * 50)

    return {
        "intersection": [feature_names[i] for i in intersection],
        "only_numpy": [feature_names[i] for i in only_np],
        "only_sklearn": [feature_names[i] for i in only_sk],
    }
