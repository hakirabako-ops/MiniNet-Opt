"""Boucle d'entraînement par mini-batchs avec enregistrement de l'historique (C1, D)."""
import numpy as np
from src.losses import bce
from src.regularization import l2_penalty, add_l2_grad


def weight_norm(model):
    """||theta||_2 = sqrt(sum_l ||W^(l)||_F^2)  (biais exclus, convention du sujet)."""
    return float(np.sqrt(sum(np.sum(l.W ** 2) for l in model.linear_layers())))


def evaluate(model, data, l2=0.0):
    """Mesures sur les données entières, dropout désactivé :
    F_train(theta), F_val(theta), ||grad||, accuracy train/val, ||theta||_2, et l'objectif F_lambda.
    Avec l2 > 0, grad_norm est la norme du gradient de F_lambda (la fonction réellement minimisée).
    """
    model.eval()
    F_train = model.loss_and_grad(data["X_train"], data["y_train"])  # remplit aussi les gradients
    add_l2_grad(model, l2)
    grad_norm = float(np.linalg.norm(model.get_grads()))
    return {
        "loss_train": F_train,
        "objective": F_train + l2_penalty(model, l2),
        "loss_val": bce(model.forward(data["X_val"]), data["y_val"]),
        "grad_norm": grad_norm,
        "acc_train": model.accuracy(data["X_train"], data["y_train"]),
        "acc_val": model.accuracy(data["X_val"], data["y_val"]),
        "weight_norm": weight_norm(model),
    }


def train(model, optimizer, data, epochs=100, batch_size=32, seed=0, l2=0.0, early_stopping=None):
    """Entraîne `model` avec `optimizer`. Retourne l'historique (dict de listes + history["meta"]).

    l2             : coefficient lambda de la pénalité (lambda/2) sum ||W||_F^2 (0 = sans)
    early_stopping : objet EarlyStopping ou None. S'il est fourni, on s'arrête après `patience`
                     époques sans amélioration de la loss val, puis on restaure les meilleurs paramètres.
    history[k] a une entrée à l'époque 0 (avant entraînement) puis une par époque réalisée.
    """
    X, y = data["X_train"], data["y_train"]
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    optimizer.reset()
    if early_stopping is not None:
        early_stopping.reset()

    keys = ["loss_train", "objective", "loss_val", "grad_norm", "acc_train", "acc_val", "weight_norm"]
    history = {k: [] for k in keys}

    def log():
        for k, v in evaluate(model, data, l2).items():
            history[k].append(v)

    log()                                           # époque 0
    if early_stopping is not None:
        early_stopping.step(history["loss_val"][0], model.get_params(), 0)
    stopped = False
    for epoch in range(1, epochs + 1):
        model.train()                               # dropout actif pendant l'entraînement
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            model.loss_and_grad(X[idx], y[idx])     # g_k = grad F_{B_k}(theta_k)
            add_l2_grad(model, l2)                  # + lambda W  (gradient de F_lambda)
            theta = optimizer.step(model.get_params(), model.get_grads())
            model.set_params(theta)
        log()                                       # évaluation (repasse en mode eval)
        if early_stopping is not None and early_stopping.step(history["loss_val"][-1], model.get_params(), epoch):
            stopped = True
            break

    meta = {"epochs_run": len(history["loss_train"]) - 1, "stopped_early": stopped}
    if early_stopping is not None:
        model.set_params(early_stopping.best_params)   # restaure les meilleurs paramètres
        meta.update(best_epoch=early_stopping.best_epoch, best_val_loss=early_stopping.best_loss)
    model.eval()
    history["meta"] = meta
    return history