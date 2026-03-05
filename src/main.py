import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import predict_linear, predict_quadratic
from .train import train_linear, train_quadratic
from .metrics import rmse

DATA_PATH = os.path.join("data", "prix_maisons.csv")
OUT_DIR = "outputs"


def standardize(v: np.ndarray):
    mu = float(np.mean(v))
    sigma = float(np.std(v))
    if sigma == 0:
        raise ValueError("Écart-type = 0, impossible de standardiser.")
    return (v - mu) / sigma, mu, sigma


def plot_fit(x, y, y_pred, title, outpath):
    plt.figure()
    plt.scatter(x, y)
    # Pour tracer une courbe propre, on trie x
    idx = np.argsort(x)
    plt.plot(x[idx], y_pred[idx])
    plt.title(title)
    plt.xlabel("surface (standardisée)")
    plt.ylabel("prix (standardisé)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_rmse(history, title, outpath):
    plt.figure()
    plt.plot(history)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Lire CSV :contentReference[oaicite:4]{index=4}
    df = pd.read_csv(DATA_PATH)

    # Adapter si tes colonnes s'appellent autrement
    # Ici on suppose: surface, prix
    x_raw = df["surface"].to_numpy(dtype=float)
    y_raw = df["prix"].to_numpy(dtype=float)

    # 2) Standardiser x et y :contentReference[oaicite:5]{index=5}
    x, mux, sigx = standardize(x_raw)
    y, muy, sigy = standardize(y_raw)

    # 3) Entraîner linéaire + quadratique :contentReference[oaicite:6]{index=6}
    lr = 0.05
    epochs = 2000

    aL, bL, histL = train_linear(x, y, lr=lr, epochs=epochs, seed=0)
    y_pred_L = predict_linear(aL, bL, x)
    rmse_L = rmse(y_pred_L, y)

    aQ, bQ, cQ, histQ = train_quadratic(x, y, lr=lr, epochs=epochs, seed=0)
    y_pred_Q = predict_quadratic(aQ, bQ, cQ, x)
    rmse_Q = rmse(y_pred_Q, y)

    print("=== Résultats (sur données standardisées) ===")
    print(f"Linear:    a={aL:.4f}, b={bL:.4f}, RMSE={rmse_L:.4f}")
    print(f"Quadratic: a={aQ:.4f}, b={bQ:.4f}, c={cQ:.4f}, RMSE={rmse_Q:.4f}")

    # 4) Figures demandées (fit + rmse curve) :contentReference[oaicite:7]{index=7}
    plot_fit(x, y, y_pred_L, "Régression linéaire (standardisée)",
             os.path.join(OUT_DIR, "fit_linear.png"))
    plot_rmse(histL, "RMSE vs epochs (linéaire)",
              os.path.join(OUT_DIR, "rmse_linear.png"))

    plot_fit(x, y, y_pred_Q, "Régression quadratique (standardisée)",
             os.path.join(OUT_DIR, "fit_quadratic.png"))
    plot_rmse(histQ, "RMSE vs epochs (quadratique)",
              os.path.join(OUT_DIR, "rmse_quadratic.png"))

    # 5) Petit résumé comparaison :contentReference[oaicite:8]{index=8}
    if rmse_Q < rmse_L:
        print(" Quadratique meilleur (RMSE plus petite).")
    else:
        print("Linéaire meilleur ou équivalent (selon données / lr / epochs).")


if __name__ == "__main__":
    main()