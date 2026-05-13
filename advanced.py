"""
advanced.py - Advanced 层主流程
包含：
- 线性回归（全特征，baseline）
- 随机森林（NumPy Pearson 特征 / SelectKBest 特征）
- 纯 NumPy 加权平均融合
- 四模型性能对比表格
- 柱状图：仅对比两种特征选择方法对应的 Random Forest MSE
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from data_prep import load_data, split_data, standardize_features
from feature_selection import select_features_numpy, select_features_sklearn, compare_feature_sets
from models import train_random_forest
from visualization import plot_mse_comparison


def fuse_predictions(pred_list, weights):
    """纯 NumPy 加权平均融合，自动归一化权重"""
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    result = np.zeros(len(pred_list[0]), dtype=float)
    for w, p in zip(weights, pred_list):
        result += w * np.asarray(p)
    return result


def main():
    # ========== 1. 加载数据 ==========
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = standardize_features(X_train, X_test)

    # ========== 2. 特征选择（在训练集原始数据上计算，避免数据泄露） ==========
    print("=" * 50)
    print("特征选择方法A：纯 NumPy Pearson 相关系数")
    print("=" * 50)
    indices_np, scores_np = select_features_numpy(X_train, y_train, k=5)
    print(f"选中特征索引: {indices_np}")
    print(f"对应特征名:   {[feature_names[i] for i in indices_np]}")
    print(f"Pearson 相关系数: {scores_np}")

    print(f"\n{'=' * 50}")
    print("特征选择方法B：SelectKBest + f_regression")
    print("=" * 50)
    indices_sk, scores_sk = select_features_sklearn(X_train, y_train, k=5)
    print(f"选中特征索引: {indices_sk}")
    print(f"对应特征名:   {[feature_names[i] for i in indices_sk]}")
    print(f"F 值: {scores_sk}")

    compare_feature_sets(indices_np, indices_sk, feature_names)

    # ========== 3. 提取特征子集（在标准化数据上切片） ==========
    X_train_np = X_train_scaled[:, indices_np]
    X_test_np  = X_test_scaled[:, indices_np]
    X_train_sk = X_train_scaled[:, indices_sk]
    X_test_sk  = X_test_scaled[:, indices_sk]

    # ========== 4. 训练模型 ==========
    print(f"\n{'=' * 50}")
    print("训练 baseline：线性回归（全特征）")
    print("=" * 50)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    print(f"\n{'=' * 50}")
    print("训练 Random Forest（NumPy Pearson 特征集）")
    print("=" * 50)
    rf_np_model, _, _, _ = train_random_forest(X_train_np, y_train, X_test_np, y_test)

    print(f"\n{'=' * 50}")
    print("训练 Random Forest（SelectKBest 特征集）")
    print("=" * 50)
    rf_sk_model, _, _, _ = train_random_forest(X_train_sk, y_train, X_test_sk, y_test)

    # ========== 5. 预测 ==========
    y_pred_lr    = lr_model.predict(X_test_scaled)
    y_pred_rf_np = rf_np_model.predict(X_test_np)
    y_pred_rf_sk = rf_sk_model.predict(X_test_sk)

    # ========== 6. 加权融合（纯 NumPy，权重在报告中说明） ==========
    weights = [0.2, 0.4, 0.4]   # LR / RF_Pearson / RF_SKB
    y_pred_fused = fuse_predictions([y_pred_lr, y_pred_rf_np, y_pred_rf_sk], weights)

    # ========== 7. 纯 NumPy 计算 MSE / MAE ==========
    def compute_mse(y_true, y_pred):
        return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

    def compute_mae(y_true, y_pred):
        return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

    # ========== 8. 打印四模型性能对比表格 ==========
    models_perf = {
        "Linear Regression (all features)": y_pred_lr,
        "RF (NumPy Pearson features)":      y_pred_rf_np,
        "RF (SelectKBest features)":        y_pred_rf_sk,
        "Weighted Fusion":                  y_pred_fused,
    }
    print("\n" + "=" * 62)
    print("模型性能对比")
    print("=" * 62)
    print(f"{'Model':<35} {'MSE':>12} {'MAE':>12}")
    print("-" * 62)
    for name, pred in models_perf.items():
        print(f"{name:<35} {compute_mse(y_test, pred):>12.4f} {compute_mae(y_test, pred):>12.4f}")
    print("=" * 62)

    # ========== 9. 柱状图：仅对比两种特征选择方法的 Random Forest MSE ==========
    rf_mse_dict = {
        "RF (NumPy Pearson)": compute_mse(y_test, y_pred_rf_np),
        "RF (SelectKBest)":   compute_mse(y_test, y_pred_rf_sk),
    }
    plot_mse_comparison(
        rf_mse_dict,
        save_path="output/advanced_mse_comparison.png",
        title="Feature Selection Method vs Test MSE"
    )


if __name__ == "__main__":
    main()
