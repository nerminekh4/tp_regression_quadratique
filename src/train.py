import numpy as np
from .models import predict_linear, predict_quadratic
from .metrics import rmse

# --------- LINEAR ---------
def step_linear(a: float, b: float, x: np.ndarray, y: np.ndarray, lr: float):
    """
    1 step GD for y_hat = a x + b with MSE loss.
    Returns updated (a,b) and current rmse.
    """
    n = x.shape[0]
    y_pred = predict_linear(a, b, x)
    e = y_pred - y  # error vector

    # Gradients for MSE: (2/n) sum e * d(y_hat)/d(param)
    dL_da = (2.0 / n) * np.sum(e * x)
    dL_db = (2.0 / n) * np.sum(e)

    a = a - lr * dL_da
    b = b - lr * dL_db

    return a, b, rmse(y_pred, y)

def train_linear(x: np.ndarray, y: np.ndarray, lr: float, epochs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = float(rng.normal(scale=0.1))
    b = float(rng.normal(scale=0.1))

    history = []
    for _ in range(epochs):
        a, b, r = step_linear(a, b, x, y, lr)
        history.append(r)

    return a, b, history


# --------- QUADRATIC ---------
def step_quadratic(a: float, b: float, c: float, x: np.ndarray, y: np.ndarray, lr: float):
    """
    1 step GD for y_hat = a x^2 + b x + c with MSE loss.
    Returns updated (a,b,c) and current rmse.
    """
    n = x.shape[0]
    y_pred = predict_quadratic(a, b, c, x)
    e = y_pred - y

    # From the statement: dL/da = (2/n) sum e * x^2, etc. :contentReference[oaicite:3]{index=3}
    dL_da = (2.0 / n) * np.sum(e * (x ** 2))
    dL_db = (2.0 / n) * np.sum(e * x)
    dL_dc = (2.0 / n) * np.sum(e)

    a = a - lr * dL_da
    b = b - lr * dL_db
    c = c - lr * dL_dc

    return a, b, c, rmse(y_pred, y)

def train_quadratic(x: np.ndarray, y: np.ndarray, lr: float, epochs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = float(rng.normal(scale=0.1))
    b = float(rng.normal(scale=0.1))
    c = float(rng.normal(scale=0.1))

    history = []
    for _ in range(epochs):
        a, b, c, r = step_quadratic(a, b, c, x, y, lr)
        history.append(r)

    return a, b, c, history