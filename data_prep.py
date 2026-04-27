import os
import tarfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """从本地 housing.tgz 加载加州房价数据集"""
    print("=" * 50)
    print("步骤1：加载加州房价数据集 (housing.tgz)")
    print("=" * 50)

    tgz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "housing.tgz")
    with tarfile.open(tgz_path, "r:gz") as tar:
        csv_file = tar.extractfile("housing.csv")
        df = pd.read_csv(csv_file)

    print(f"  原始数据量: {len(df)} 行, {len(df.columns)} 列")

    # 处理缺失值：total_bedrooms 有少量缺失，用中位数填充
    missing = df.isnull().sum()
    if missing.any():
        print(f"  缺失值处理: {dict(missing[missing > 0])}, 用中位数填充")
        df = df.fillna(df.median(numeric_only=True))

    # 对分类特征 ocean_proximity 做 one-hot 编码
    df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=float)

    # 分离特征和目标
    y = df["median_house_value"].values
    X = df.drop(columns=["median_house_value"]).values
    feature_names = [c for c in df.columns if c != "median_house_value"]

    print(f"  加载完成，数据量: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"  特征列: {feature_names}")
    print()
    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    """划分训练集和测试集"""
    print("=" * 50)
    print("步骤2：划分训练集和测试集")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print()
    return X_train, X_test, y_train, y_test


def standardize_features(X_train, X_test):
    """使用 StandardScaler 对特征进行标准化"""
    print("=" * 50)
    print("步骤3：标准化特征")
    print("=" * 50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  标准化完成")
    print(f"  训练集均值范围: {X_train_scaled.mean(axis=0).min():.4f} ~ {X_train_scaled.mean(axis=0).max():.4f}")
    print(f"  训练集标准差范围: {X_train_scaled.std(axis=0).min():.4f} ~ {X_train_scaled.std(axis=0).max():.4f}")
    print()
    return X_train_scaled, X_test_scaled, scaler