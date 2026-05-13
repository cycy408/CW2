import numpy as np


# ========== NumPy-only implementation ==========
def compute_mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)


# ========== NumPy-only implementation ==========
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
