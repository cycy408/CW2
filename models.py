"""
models.py - 模型训练模块
包含：
- 训练 Random Forest 回归器（在指定特征子集上）
- 纯 NumPy 计算 MSE / MAE
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    """
    训练 Random Forest 并计算测试集上的 MSE 和 MAE (纯 NumPy)
    参数:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
    返回:
        model: 训练好的模型
        mse: 测试集 MSE
        mae: 测试集 MAE
        y_pred: 测试集预测值
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors = y_test - y_pred
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    return model, mse, mae, y_pred

def train_random_forest_on_selected(X_train, y_train, X_test, y_test, selected_indices, **kwargs):
    """
    在选定的特征子集上训练 Random Forest
    参数 selected_indices: 要保留的特征索引列表
    """
    X_train_sub = X_train[:, selected_indices]
    X_test_sub = X_test[:, selected_indices]
    return train_random_forest(X_train_sub, y_train, X_test_sub, y_test, **kwargs)