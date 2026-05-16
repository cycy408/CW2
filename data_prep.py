import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    print("=" * 50)
    print("Step 1: Load California Housing Dataset")
    print("=" * 50)

    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = list(housing.feature_names)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Feature names: {feature_names}")
    print(f"  Target: median house value (unit: $100k)")
    print()
    return X, y, feature_names


def split_data(X, y, test_size=0.2):
    print("=" * 50)
    print("Step 2: Train/Test Split")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    print(f"  Training set: {X_train.shape[0]}")
    print(f"  Test set:     {X_test.shape[0]}")
    print()
    return X_train, X_test, y_train, y_test


def fill_missing_values(X_train, X_test):
    print("=" * 50)
    print("Step 3: Fill Missing Values with Median")
    print("=" * 50)

    col_medians = np.nanmedian(X_train, axis=0)

    train_filled = np.where(np.isnan(X_train), col_medians, X_train)
    test_filled = np.where(np.isnan(X_test), col_medians, X_test)

    nan_count_train = np.sum(np.isnan(train_filled))
    nan_count_test = np.sum(np.isnan(test_filled))
    print(f"  NaN in train after fill: {nan_count_train}")
    print(f"  NaN in test  after fill: {nan_count_test}")
    print()
    return train_filled, test_filled


def standardize_features(X_train, X_test):
    print("=" * 50)
    print("Step 4: Standardize Features")
    print("=" * 50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Done.")
    print(f"  Train mean range: [{X_train_scaled.mean(axis=0).min():.4f}, {X_train_scaled.mean(axis=0).max():.4f}]")
    print(f"  Train std  range: [{X_train_scaled.std(axis=0).min():.4f}, {X_train_scaled.std(axis=0).max():.4f}]")
    print()
    return X_train_scaled, X_test_scaled, scaler
