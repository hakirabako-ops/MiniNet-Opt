"""Assemblage du réseau d -> n1 -> n2 -> 1 (ReLU, ReLU, Sigmoid)."""
import numpy as np
from src.layers import Linear, ReLU, Sigmoid, Dropout
from src.losses import bce, bce_grad


class MLP:
    def __init__(self, sizes=(2, 16, 8, 1), seed=0, drop_rate=0.0):
        rng = np.random.default_rng(seed)
        rng_drop = np.random.default_rng(seed + 10_000)   # rng séparé : l'init des poids ne change pas
        self.sizes = tuple(sizes)
        self.layers = []
        n_linear = len(sizes) - 1
        for i in range(n_linear):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng))
            # ReLU sur les couches cachées, Sigmoid en sortie
            if i == n_linear - 1:
                self.layers.append(Sigmoid())
            else:
                self.layers.append(ReLU())
                if drop_rate > 0:                       # dropout après chaque ReLU caché
                    self.layers.append(Dropout(p=1.0 - drop_rate, rng=rng_drop))

    # ---------- mode entraînement / évaluation (dropout) ----------
    def train(self):
        for l in self.layers:
            if isinstance(l, Dropout):
                l.training = True

    def eval(self):
        for l in self.layers:
            if isinstance(l, Dropout):
                l.training = False

    # ---------- forward / prédiction ----------
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out                          # y_hat, forme (n, 1)

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(float)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # ---------- backward ----------
    def backward(self, y_hat, y):
        """Rétropropagation à partir de la BCE. Remplit dW, db de chaque couche Linear.
        Doit être appelé juste après forward (les caches du forward sont utilisés)."""
        grad = bce_grad(y_hat, y)           # dF/dy_hat
        for layer in reversed(self.layers):
            grad = layer.backward(grad)     # propage dF/d(sortie) -> dF/d(entrée)

    def loss_and_grad(self, X, y):
        """Un forward + un backward. Retourne la loss ; les gradients sont dans les couches."""
        y_hat = self.forward(X)
        self.backward(y_hat, y)
        return bce(y_hat, y)

    # ---------- accès aux paramètres ----------
    def linear_layers(self):
        return [l for l in self.layers if isinstance(l, Linear)]

    def n_parameters(self):
        return sum(l.W.size + l.b.size for l in self.linear_layers())

    def get_params(self):
        """Vecteur plat de tous les paramètres (W1, b1, W2, b2, ...)."""
        return np.concatenate([np.concatenate([l.W.ravel(), l.b.ravel()])
                               for l in self.linear_layers()])

    def set_params(self, theta):
        i = 0
        for l in self.linear_layers():
            n = l.W.size
            l.W = theta[i:i + n].reshape(l.W.shape).copy(); i += n
            n = l.b.size
            l.b = theta[i:i + n].reshape(l.b.shape).copy(); i += n

    def get_grads(self):
        """Vecteur plat des gradients, même ordre que get_params."""
        return np.concatenate([np.concatenate([l.dW.ravel(), l.db.ravel()])
                               for l in self.linear_layers()])