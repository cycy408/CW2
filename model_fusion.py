import numpy as np


# ========== NumPy-only implementation ==========
def fuse_predictions(pred_list, weights=None):
    if weights is None:
        weights = [0.2, 0.4, 0.4]
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    result = np.zeros(len(pred_list[0]), dtype=float)
    for w, p in zip(weights, pred_list):
        result += w * np.asarray(p)
    return result
