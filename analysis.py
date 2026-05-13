"""
analysis.py - 极端值检测与移除分析
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import train_random_forest


def detect_outliers_iqr(y, factor=1.5):
    """用 IQR 方法检测极端值，返回布尔掩码（True 表示极端值）"""
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    return (y < q1 - factor * iqr) | (y > q3 + factor * iqr)


def analyze_outlier_features(X, y, outlier_mask, feature_names):
    """对比极端值与正常样本的特征均值分布"""
    outliers_X = X[outlier_mask]
    normal_X   = X[~outlier_mask]
    outliers_y = y[outlier_mask]
    normal_y   = y[~outlier_mask]

    print("\n=== 极端值分析 ===")
    print(f"极端值数量: {np.sum(outlier_mask)} ({np.mean(outlier_mask)*100:.2f}%)")
    print(f"极端值房价范围:  [{outliers_y.min():.2f}, {outliers_y.max():.2f}]")
    print(f"正常样本房价范围: [{normal_y.min():.2f}, {normal_y.max():.2f}]")

    print("\n特征均值对比:")
    print(f"{'Feature':<25} {'正常样本':>12} {'极端值样本':>12}")
    print("-" * 52)
    for i, name in enumerate(feature_names):
        print(f"{name:<25} {np.mean(normal_X[:, i]):>12.4f} {np.mean(outliers_X[:, i]):>12.4f}")

    return outliers_X, normal_X


def evaluate_after_outlier_removal(X, y, outlier_mask, test_size=0.2, random_state=42):
    """
    移除极端值后重新划分、标准化、训练 Random Forest 并返回性能指标。
    返回: (mse, mae, y_test, y_pred)
    """
    X_clean = X[~outlier_mask]
    y_clean = y[~outlier_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    _, mse, mae, y_pred = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    return mse, mae, y_test, y_pred
