"""
主程序 - 第一步：基础线性回归 + NumPy 计算 MSE/MAE
加州房价回归项目
"""

import os

import numpy as np
from sklearn.linear_model import LinearRegression

from data_prep import load_data, split_data, standardize_features


def calculate_metrics_numpy(y_true, y_pred):
    """
    参数:
        y_true: 真实房价
        y_pred: 预测房价
    
    返回:
        mse: 均方误差
        mae: 平均绝对误差
        rmse: 均方根误差
    """
    # 计算误差向量
    errors = y_true - y_pred
    
    # MSE = 均方误差
    mse = np.mean(errors ** 2)
    
    # MAE = 平均绝对误差
    mae = np.mean(np.abs(errors))
    
    # RMSE = 均方根误差（便于理解）
    rmse = np.sqrt(mse)
    
    return mse, mae, rmse


def main():
    """主函数：运行完整的预处理和训练流程"""
    
    print("\n" + "=" * 60)
    print("INT101 加州房价回归项目 - 第一步")
    print("基础线性回归模型 + NumPy 手动计算误差")
    print("=" * 60 + "\n")
    
    # ============================================
    # 1. 加载数据
    # ============================================
    X, y, feature_names = load_data()
    
    # ============================================
    # 2. 划分训练集和测试集
    # ============================================
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # ============================================
    # 3. 标准化特征
    # ============================================
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # ============================================
    # 4. 训练线性回归模型
    # ============================================
    print("=" * 50)
    print("步骤4：训练线性回归模型")
    print("=" * 50)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    print(f"[OK] 线性回归模型训练完成")
    print(f"  模型截距 (Intercept): {lr_model.intercept_:.6f}")
    print(f"  模型系数 (Coefficients):")
    for name, coef in zip(feature_names, lr_model.coef_):
        print(f"    {name:12s}: {coef:8.6f}")
    print()
    
    # ============================================
    # 5. 在测试集上预测
    # ============================================
    print("=" * 50)
    print("步骤5：测试集预测")
    print("=" * 50)
    
    y_pred = lr_model.predict(X_test_scaled)
    
    print(f"[OK] 预测完成")
    print(f"  真实房价示例 (前5个): {y_test[:5]}")
    print(f"  预测房价示例 (前5个): {y_pred[:5]}")
    print()
    
    # ============================================
    # 6. 用纯 NumPy 计算误差（作业强制要求）
    # ============================================
    print("=" * 50)
    print("步骤6：用纯 NumPy 计算 MSE 和 MAE")
    print("=" * 50)
    
    mse, mae, rmse = calculate_metrics_numpy(y_test, y_pred)
    
    print(f"[OK] 使用纯 NumPy 计算误差指标")
    print(f"  均方误差 (MSE):  {mse:.6f}")
    print(f"  平均绝对误差 (MAE): {mae:.6f}")
    print(f"  均方根误差 (RMSE): {rmse:.6f}")
    print()
    
    # ============================================
    # 7. 结果总结
    # ============================================
    print("=" * 50)
    print("基础线性回归模型性能总结")
    print("=" * 50)
    print(f"{'指标':<12} {'数值':<15} {'说明':<20}")
    print("-" * 50)
    print(f"{'MSE':<12} {mse:<15.2f} 均方误差")
    print(f"{'MAE':<12} {mae:<15.2f} 平均绝对误差 (约 {mae:.0f} 美元)")
    print(f"{'RMSE':<12} {rmse:<15.2f} 均方根误差 (约 {rmse:.0f} 美元)")
    print("=" * 50)
    
    # ============================================
    # 8. 保存结果到文件（可选）
    # ============================================
    os.makedirs("output", exist_ok=True)
    
    with open("output/baseline_results.txt", "w") as f:
        f.write("=" * 50 + "\n")
        f.write("基础线性回归模型性能\n")
        f.write("=" * 50 + "\n")
        f.write(f"MSE:  {mse:.6f}\n")
        f.write(f"MAE:  {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write("=" * 50 + "\n")
        f.write("\n模型系数:\n")
        for name, coef in zip(feature_names, lr_model.coef_):
            f.write(f"  {name:12s}: {coef:8.6f}\n")
    
    print("\n[OK] 结果已保存到 output/baseline_results.txt")
    
    # ============================================
    # 9. 完成提示
    # ============================================
    print("\n" + "=" * 60)
    print("[DONE] 第一步完成！")
    print("=" * 60)
    print("\n后续步骤预告:")
    print("  1. 特征选择（NumPy 相关系数 + SelectKBest）")
    print("  2. 训练 Random Forest 模型")
    print("  3. 手动模型融合（加权平均）")
    print("  4. 地理热力图（10×10 网格）")
    print("  5. 极端值分析")
    print("=" * 60)


# 确保直接运行 main.py 时执行 main 函数
if __name__ == "__main__":
    main()