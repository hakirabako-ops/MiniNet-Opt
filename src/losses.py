"""Fonctions de perte (NumPy)."""
import numpy as np

EPS = 1e-12


def bce(y_hat, y, eps=EPS):
    """Binary cross-entropy moyennée sur le batch :
    F_B = 1/|B| sum_i [ -y_i log(y_hat_i) - (1 - y_i) log(1 - y_hat_i) ].
    y_hat est clippé dans [eps, 1 - eps] pour éviter log(0).
    """
    p = np.clip(y_hat, eps, 1.0 - eps)
    return float(np.mean(-y * np.log(p) - (1.0 - y) * np.log(1.0 - p)))


def bce_grad(y_hat, y, eps=EPS):
    """Gradient de F_B par rapport à y_hat (chaque composante) :
    dF/dy_hat_i = 1/|B| * (y_hat_i - y_i) / (y_hat_i (1 - y_hat_i)).
    """
    p = np.clip(y_hat, eps, 1.0 - eps)
    return (p - y) / (p * (1.0 - p)) / y.shape[0]