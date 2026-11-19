"""Régularisation L2 et early stopping (partie D)."""
import numpy as np


def l2_penalty(model, lam):
    """(lambda / 2) * sum_l ||W^(l)||_F^2  (biais non pénalisés)."""
    return 0.5 * lam * sum(float(np.sum(l.W ** 2)) for l in model.linear_layers())


def add_l2_grad(model, lam):
    """Ajoute le gradient de la pénalité : d/dW [(lambda/2)||W||_F^2] = lambda * W.
    À appeler après model.loss_and_grad (les dW contiennent alors le gradient de F_lambda)."""
    if lam > 0:
        for l in model.linear_layers():
            l.dW = l.dW + lam * l.W


class EarlyStopping:
    """Early stopping avec patience P (sujet 7.4).

    À chaque époque : si la loss de validation s'améliore (de plus de min_delta), on sauvegarde les
    paramètres et on remet le compteur à 0 ; sinon on l'incrémente. On s'arrête quand compteur >= patience.
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.reset()

    def reset(self):
        self.best_loss = np.inf
        self.best_params = None
        self.best_epoch = 0
        self.counter = 0
        self.stopped_epoch = None

    def step(self, val_loss, params, epoch):
        """Retourne True s'il faut arrêter l'entraînement."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_params, self.best_epoch = val_loss, params.copy(), epoch
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False