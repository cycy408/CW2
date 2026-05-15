import numpy as np


# ========== Method 1: Pearson Correlation (NumPy) ==========
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


# ========== Method 2: F-Regression / F-Test (NumPy) ==========
def f_regression_numpy(X, y):
    """Compute F-statistic for each feature using only NumPy.

    F = (r^2 / (1 - r^2)) * (n - 2)
    where r is the Pearson correlation between feature and target.
    """
    r = pearson_correlation_numpy(X, y)
    n = X.shape[0]
    r_squared = r ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        f_scores = (r_squared / (1.0 - r_squared)) * (n - 2)
        f_scores[np.isinf(f_scores)] = np.finfo(np.float64).max
        f_scores[np.isnan(f_scores)] = 0.0
    return f_scores


def select_features_fregression(X, y, k=5):
    f_scores = f_regression_numpy(X, y)
    selected_indices = np.argsort(f_scores)[::-1][:k]
    scores = f_scores[selected_indices]
    return selected_indices, scores


def compare_feature_sets(indices_np, indices_freg, feature_names):
    set_np = set(indices_np)
    set_freg = set(indices_freg)
    intersection = set_np & set_freg
    only_np = set_np - set_freg
    only_freg = set_freg - set_np

    print("\n=== Feature Selection Comparison ===")
    print(f"NumPy Pearson  top-5: {[feature_names[i] for i in indices_np]}")
    print(f"NumPy F-Test   top-5: {[feature_names[i] for i in indices_freg]}")
    print(f"Intersection:         {[feature_names[i] for i in intersection]}")
    print(f"Only in Pearson:      {[feature_names[i] for i in only_np]}")
    print(f"Only in F-Test:       {[feature_names[i] for i in only_freg]}")
    print("=" * 50)

    return {
        "intersection": [feature_names[i] for i in intersection],
        "only_pearson": [feature_names[i] for i in only_np],
        "only_ftest": [feature_names[i] for i in only_freg],
    }
