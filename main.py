"""
main.py - 加州房价回归项目
执行顺序：
1. 加载并预处理数据
2. 基准模型：线性回归（全特征）+ NumPy 手动计算误差
3. 两种特征选择方法对比
4. Random Forest（全特征 / NumPy 特征 / Sklearn 特征）
5. 对比 MSE 并画图
6. Extended 层：地理热图 & 极端值分析
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression

from data_prep import load_data, split_data, standardize_features
from feature_selection import select_features_numpy, select_features_sklearn, compare_feature_sets
from models import train_random_forest, train_random_forest_on_selected
from visualization import plot_mse_comparison, plot_geospatial_heatmap
from analysis import detect_outliers_iqr, analyze_outlier_features, evaluate_after_outlier_removal
import advanced


def calculate_metrics_numpy(y_true, y_pred):
    """纯 NumPy 计算 MSE / MAE / RMSE"""
    errors = np.asarray(y_true) - np.asarray(y_pred)
    mse  = np.mean(errors ** 2)
    mae  = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)
    return mse, mae, rmse


def main():
    print("\n" + "=" * 60)
    print("INT101 加州房价回归项目")
    print("=" * 60 + "\n")

    # ========== 1. 数据加载与预处理 ==========
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = standardize_features(X_train, X_test)

    # ========== 2. 基准模型：线性回归（全特征） ==========
    print("=" * 50)
    print("训练基准模型：线性回归（全特征）")
    print("=" * 50)

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    mse_lr, mae_lr, rmse_lr = calculate_metrics_numpy(y_test, y_pred_lr)

    print(f"[OK] 线性回归模型训练完成")
    print(f"  模型截距 (Intercept): {lr_model.intercept_:.6f}")
    print(f"  模型系数 (Coefficients):")
    for name, coef in zip(feature_names, lr_model.coef_):
        print(f"    {name:25s}: {coef:8.6f}")
    print(f"\n  MSE:  {mse_lr:.2f}")
    print(f"  MAE:  {mae_lr:.2f}")
    print(f"  RMSE: {rmse_lr:.2f}\n")

    # ========== 3. 特征选择 ==========
    print("=" * 50)
    print("特征选择方法1：纯 NumPy 相关系数")
    print("=" * 50)
    indices_numpy, scores_numpy = select_features_numpy(X_train, y_train, k=5)
    print(f"选中特征索引: {indices_numpy}")
    print(f"对应特征名:   {[feature_names[i] for i in indices_numpy]}")
    print(f"相关系数:     {scores_numpy}")

    print(f"\n{'=' * 50}")
    print("特征选择方法2：SelectKBest + f_regression")
    print("=" * 50)
    indices_sklearn, scores_sklearn = select_features_sklearn(X_train, y_train, k=5)
    print(f"选中特征索引: {indices_sklearn}")
    print(f"对应特征名:   {[feature_names[i] for i in indices_sklearn]}")
    print(f"F值:          {scores_sklearn}")

    compare_feature_sets(indices_numpy, indices_sklearn, feature_names)

    # ========== 4. Random Forest（全特征） ==========
    print(f"\n{'=' * 50}")
    print("训练 Random Forest（全特征）")
    print("=" * 50)
    _, mse_rf_all, mae_rf_all, y_pred_rf_all = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    rmse_rf_all = np.sqrt(mse_rf_all)
    print(f"Random Forest (全特征) MSE:  {mse_rf_all:.2f}")
    print(f"Random Forest (全特征) MAE:  {mae_rf_all:.2f}")
    print(f"Random Forest (全特征) RMSE: {rmse_rf_all:.2f}")

    # ========== 5. Random Forest（NumPy 特征集） ==========
    print(f"\n{'=' * 50}")
    print("训练 Random Forest（NumPy 特征集）")
    print("=" * 50)
    _, mse_rf_numpy, _, _ = train_random_forest_on_selected(
        X_train_scaled, y_train, X_test_scaled, y_test, indices_numpy
    )
    print(f"Random Forest (NumPy 特征) MSE: {mse_rf_numpy:.2f}")

    # ========== 6. Random Forest（Sklearn 特征集） ==========
    print(f"\n{'=' * 50}")
    print("训练 Random Forest（Sklearn 特征集）")
    print("=" * 50)
    _, mse_rf_sklearn, _, _ = train_random_forest_on_selected(
        X_train_scaled, y_train, X_test_scaled, y_test, indices_sklearn
    )
    print(f"Random Forest (Sklearn 特征) MSE: {mse_rf_sklearn:.2f}")

    # ========== 7. 绘制柱状图 ==========
    mse_dict = {
        "Linear Regression\n(all features)":    mse_lr,
        "Random Forest\n(all features)":         mse_rf_all,
        "Random Forest\n(NumPy features)":        mse_rf_numpy,
        "Random Forest\n(Sklearn features)":      mse_rf_sklearn,
    }
    plot_mse_comparison(mse_dict, save_path="output/mse_comparison.png")

    # ========== 8. 结果总结 ==========
    print("\n" + "=" * 60)
    print("模型性能总结")
    print("=" * 60)
    print(f"{'模型':<35} {'MSE':<20} {'RMSE':<15}")
    print("-" * 70)
    print(f"{'Linear Regression (全特征)':<35} {mse_lr:<20.2f} {rmse_lr:<15.2f}")
    print(f"{'Random Forest (全特征)':<35} {mse_rf_all:<20.2f} {rmse_rf_all:<15.2f}")
    print(f"{'Random Forest (NumPy 5特征)':<35} {mse_rf_numpy:<20.2f} {np.sqrt(mse_rf_numpy):<15.2f}")
    print(f"{'Random Forest (Sklearn 5特征)':<35} {mse_rf_sklearn:<20.2f} {np.sqrt(mse_rf_sklearn):<15.2f}")
    print("=" * 70)

    best_mse = min(mse_rf_all, mse_rf_numpy, mse_rf_sklearn)
    if best_mse == mse_rf_all:
        best_method = "全特征"
    elif best_mse == mse_rf_numpy:
        best_method = "NumPy 特征集"
    else:
        best_method = "Sklearn 特征集"
    print(f"\n最佳模型: Random Forest ({best_method}), MSE = {best_mse:.2f}")
    print(f"相比线性回归 (MSE={mse_lr:.2f})，MSE 降低 {(mse_lr - best_mse) / mse_lr * 100:.1f}%")

    # ========== 9. 保存结果 ==========
    os.makedirs("output", exist_ok=True)
    with open("output/baseline_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("模型性能对比\n")
        f.write("=" * 50 + "\n")
        f.write(f"Linear Regression (全特征)    MSE: {mse_lr:.6f}   RMSE: {rmse_lr:.6f}\n")
        f.write(f"Random Forest (全特征)        MSE: {mse_rf_all:.6f}   RMSE: {rmse_rf_all:.6f}\n")
        f.write(f"Random Forest (NumPy 5特征)   MSE: {mse_rf_numpy:.6f}\n")
        f.write(f"Random Forest (Sklearn 5特征) MSE: {mse_rf_sklearn:.6f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"\n最佳模型: Random Forest ({best_method}), MSE = {best_mse:.6f}\n")
        f.write(f"相比线性回归，MSE 降低 {(mse_lr - best_mse) / mse_lr * 100:.1f}%\n")
        f.write("\n线性回归模型系数:\n")
        for name, coef in zip(feature_names, lr_model.coef_):
            f.write(f"  {name:25s}: {coef:8.6f}\n")

    print("\n[OK] 结果已保存到 output/baseline_results.txt")
    print("[OK] 柱状图已保存到 output/mse_comparison.png")
    print("=" * 60)

    # ========== 10. Extended 层：地理热图 & 极端值分析 ==========
    print("\n" + "=" * 60)
    print("Extended 层：地理热图 & 极端值分析")
    print("=" * 60)

    # 地理热图（使用原始未标准化的 X_test 中的经纬度）
    feature_names_lower = [n.lower() for n in feature_names]
    try:
        lat_idx = feature_names_lower.index('latitude')
        lon_idx = feature_names_lower.index('longitude')
        plot_geospatial_heatmap(
            X_test, y_test, y_pred_rf_all,
            lat_idx, lon_idx,
            grid_size=10,
            save_path="output/geospatial_heatmap.png"
        )
    except ValueError:
        print("[WARN] 未找到经纬度特征列，跳过地理热图")

    # 极端值检测与特征分析
    outlier_mask = detect_outliers_iqr(y, factor=1.5)
    analyze_outlier_features(X, y, outlier_mask, feature_names)

    # 移除极端值后重新训练 RF 并评估
    mse_clean, mae_clean, _, _ = evaluate_after_outlier_removal(X, y, outlier_mask)
    print(f"\n移除极端值后 RF 重训结果:")
    print(f"  MSE: {mse_clean:.4f}  (原全特征 RF MSE: {mse_rf_all:.4f})")
    print(f"  MAE: {mae_clean:.4f}")

    print("\n=== Pipeline 完成 ===")


if __name__ == "__main__":
    print()
    main()
    print("\n" + "=" * 60)
    print("Advanced 层分析")
    print("=" * 60)
    advanced.main()
