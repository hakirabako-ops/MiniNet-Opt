"""Couches du réseau (NumPy). Convention : un batch X a la forme (n, d).

Les paramètres suivent la notation du sujet : W^(l) de forme (n_out, n_in),
donc pour un batch  Z = X @ W.T + b  (équivalent à z = W x + b pour une observation).

forward() stocke dans `cache` ce dont backward() a besoin.
backward(grad_out) reçoit dF/d(sortie de la couche) et renvoie dF/d(entrée de la couche).
"""
import numpy as np


class Linear:
    """Couche affine : z = W x + b."""

    def __init__(self, n_in, n_out, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        # Initialisation de He (adaptée à ReLU) ; biais nuls
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_out, n_in))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache = None

    def forward(self, X):
        self.cache = X                      # entrée a^(l-1), utile pour dW
        return X @ self.W.T + self.b        # (n, n_in) -> (n, n_out)

    def backward(self, dZ):
        X = self.cache
        self.dW = dZ.T @ X                  # dF/dW = sum_i delta_i a_i^T  (le 1/n est déjà dans dZ)
        self.db = dZ.sum(axis=0)            # dF/db = sum_i delta_i
        return dZ @ self.W                  # dF/dX = W^T delta  (version batch)


class ReLU:
    """ReLU(z) = max(0, z)."""

    def __init__(self):
        self.cache = None

    def forward(self, Z):
        self.cache = Z                      # utile pour la dérivée au backward
        return np.maximum(0.0, Z)

    def backward(self, dA):
        return dA * (self.cache > 0)        # ReLU'(z) = 1[z > 0]


class Sigmoid:
    """sigma(z) = 1 / (1 + exp(-z)), calculée de façon numériquement stable."""

    def __init__(self):
        self.cache = None

    def forward(self, Z):
        out = np.empty_like(Z, dtype=float)
        pos = Z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
        e = np.exp(Z[~pos])                 # z < 0 : évite exp(-z) qui déborde
        out[~pos] = e / (1.0 + e)
        self.cache = out                    # sigma(z), utile pour la dérivée
        return out

    def backward(self, dA):
        s = self.cache
        return dA * s * (1.0 - s)           # sigma'(z) = sigma(z)(1 - sigma(z))


class Dropout:
    """Dropout inversé (sujet 7.3) : M_j ~ Bernoulli(p),  h~_j = (M_j / p) h_j.

    p est la probabilité de GARDER une activation (taux de dropout = 1 - p).
    Actif seulement quand self.training est True ; sinon c'est l'identité
    (désactivé en validation et en test). Par défaut le mode est « évaluation » :
    c'est la boucle d'entraînement qui appelle model.train().
    """

    def __init__(self, p=0.8, rng=None):
        self.p = p
        self.rng = np.random.default_rng() if rng is None else rng
        self.training = False
        self.mask = None

    def forward(self, X):
        if not self.training or self.p >= 1.0:
            self.mask = None
            return X
        self.mask = (self.rng.random(X.shape) < self.p) / self.p   # M / p
        return X * self.mask

    def backward(self, dA):
        return dA if self.mask is None else dA * self.mask