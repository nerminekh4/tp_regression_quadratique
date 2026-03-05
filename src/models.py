import numpy as np

def predict_linear(a: float, b: float, x: np.ndarray) -> np.ndarray:
    return a * x + b

def predict_quadratic(a: float, b: float, c: float, x: np.ndarray) -> np.ndarray:
    return a * (x ** 2) + b * x + c