import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ========== sklearn (fetch_california_housing) + pandas/numpy ==========
def load_data(verbose=True):
    if verbose:
        print("=" * 50)
        print("Step 1: Load California Housing Dataset (pandas + numpy)")
        print("=" * 50)

    housing = fetch_california_housing()
    feature_names = list(housing.feature_names)

    df = pd.DataFrame(housing.data, columns=feature_names)
    df['target'] = housing.target

    df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
    df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
    df['BedroomsPerRoom'] = df['BedroomsPerRoom'].replace([np.inf, -np.inf], 0).fillna(0)

    engineered_names = ['RoomsPerHousehold', 'BedroomsPerRoom']
    all_feature_names = feature_names + engineered_names

    if verbose:
        print(f"  Dataset loaded as DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  Original features: {feature_names}")
        print(f"  Engineered features: {engineered_names}")
        print(f"  Target: median house value (unit: $100k)")
        print(f"  Data types:\n{df.dtypes}\n")

    X = df[all_feature_names].to_numpy()
    y = df['target'].to_numpy()

    if verbose:
        print(f"  Converted to numpy — X: {X.shape}, y: {y.shape}")
        print()
    return X, y, all_feature_names


# ========== sklearn (train_test_split) ==========
def split_data(X, y, test_size=0.2, random_state=None, verbose=True):
    if verbose:
        print("=" * 50)
        print("Step 2: Train/Test Split")
        print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print(f"  Training set: {X_train.shape[0]}")
        print(f"  Test set:     {X_test.shape[0]}")
        print()
    return X_train, X_test, y_train, y_test


# ========== NumPy-only implementation ==========
def fill_missing_values(X_train, X_test, verbose=True):
    if verbose:
        print("=" * 50)
        print("Step 3: Fill Missing Values with Median")
        print("=" * 50)

    col_medians = np.nanmedian(X_train, axis=0)

    train_filled = np.where(np.isnan(X_train), col_medians, X_train)
    test_filled = np.where(np.isnan(X_test), col_medians, X_test)

    if verbose:
        nan_count_train = np.sum(np.isnan(train_filled))
        nan_count_test = np.sum(np.isnan(test_filled))
        print(f"  NaN in train after fill: {nan_count_train}")
        print(f"  NaN in test  after fill: {nan_count_test}")
        print()
    return train_filled, test_filled


# ========== sklearn (StandardScaler) ==========
def standardize_features(X_train, X_test, verbose=True):
    if verbose:
        print("=" * 50)
        print("Step 4: Standardize Features")
        print("=" * 50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose:
        print(f"  Done.")
        print(f"  Train mean range: [{X_train_scaled.mean(axis=0).min():.4f}, {X_train_scaled.mean(axis=0).max():.4f}]")
        print(f"  Train std  range: [{X_train_scaled.std(axis=0).min():.4f}, {X_train_scaled.std(axis=0).max():.4f}]")
        print()
    return X_train_scaled, X_test_scaled, scaler
