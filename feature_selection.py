"""
feature_selection.py - 特征选择模块
包含：
1. 纯 NumPy 实现的 Pearson 相关系数
2. 用 sklearn 的 SelectKBest + f_regression
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

def pearson_correlation_numpy(X, y):
    """
    纯 NumPy 计算每个特征与目标 y 的 Pearson 相关系数
    参数:
        X: numpy array, shape (n_samples, n_features)
        y: numpy array, shape (n_samples,)
    返回:
        correlations: shape (n_features,) 每个特征的相关系数
    """
    n_features = X.shape[1]
    correlations = np.zeros(n_features)
    for i in range(n_features):
        x_i = X[:, i]
        # 去均值
        x_centered = x_i - np.mean(x_i)
        y_centered = y - np.mean(y)
        # 分子：协方差
        numerator = np.sum(x_centered * y_centered)
        # 分母：标准差乘积
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        if denominator != 0:
            correlations[i] = numerator / denominator
        else:
            correlations[i] = 0.0
    return correlations

def select_features_numpy(X, y, k=5):
    """
    使用纯 NumPy 相关系数选择 top-k 特征
    返回:
        selected_indices: 选中的特征索引
        scores: 对应的相关系数值
    """
    corr = pearson_correlation_numpy(X, y)
    # 按绝对值排序，取前 k 个
    abs_corr = np.abs(corr)
    selected_indices = np.argsort(abs_corr)[::-1][:k]
    scores = corr[selected_indices]
    return selected_indices, scores

def select_features_sklearn(X, y, k=5):
    """
    使用 Sklearn 的 SelectKBest + f_regression 选择 top-k 特征
    返回:
        selected_indices: 选中的特征索引
        scores: 对应的 F 值
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    scores = selector.scores_[selected_indices]
    return selected_indices, scores

def compare_feature_sets(indices_np, indices_sk, feature_names):
    """
    比较两种方法选出的特征，打印结果并返回分析字典（用于报告）
    """
    set_np = set(indices_np)
    set_sk = set(indices_sk)
    intersection = set_np & set_sk
    only_np = set_np - set_sk
    only_sk = set_sk - set_np

    print("\n=== 特征选择结果对比 ===")
    print(f"NumPy 选中的特征索引: {indices_np}")
    print(f"Sklearn 选中的特征索引: {indices_sk}")
    print(f"NumPy 选中的特征名: {[feature_names[i] for i in indices_np]}")
    print(f"Sklearn 选中的特征名: {[feature_names[i] for i in indices_sk]}")
    print(f"交集 (两者都选): {[feature_names[i] for i in intersection]}")
    print(f"仅 NumPy 选中: {[feature_names[i] for i in only_np]}")
    print(f"仅 Sklearn 选中: {[feature_names[i] for i in only_sk]}")
    print("="*50)

    return {
        "intersection": [feature_names[i] for i in intersection],
        "only_numpy": [feature_names[i] for i in only_np],
        "only_sklearn": [feature_names[i] for i in only_sk]
    }