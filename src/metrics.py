import numpy as np

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))

def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(mse(y_pred, y_true)))