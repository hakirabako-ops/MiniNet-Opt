"""Vérification numérique des gradients par différences centrées."""
import numpy as np
from src.losses import bce


def numerical_gradient(model, X, y, eps=1e-5):
    """dF/dtheta_j ~ (F(theta + eps e_j) - F(theta - eps e_j)) / (2 eps), pour tout j."""
    theta = model.get_params()
    g = np.zeros_like(theta)
    for j in range(theta.size):
        old = theta[j]
        theta[j] = old + eps; model.set_params(theta); fp = bce(model.forward(X), y)
        theta[j] = old - eps; model.set_params(theta); fm = bce(model.forward(X), y)
        theta[j] = old
        g[j] = (fp - fm) / (2 * eps)
    model.set_params(theta)                 # remet les paramètres d'origine
    return g


def relative_error(g_a, g_n, eps=1e-12):
    """err = ||g_a - g_n|| / (||g_a|| + ||g_n|| + eps)."""
    return np.linalg.norm(g_a - g_n) / (np.linalg.norm(g_a) + np.linalg.norm(g_n) + eps)


def gradient_check(model, X, y, eps=1e-5):
    """Retourne (erreur globale, gradient analytique, gradient numérique)."""
    model.loss_and_grad(X, y)
    g_a = model.get_grads().copy()
    g_n = numerical_gradient(model, X, y, eps)
    return relative_error(g_a, g_n), g_a, g_n