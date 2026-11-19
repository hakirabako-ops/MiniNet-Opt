"""Protocole de comparaison des optimiseurs (C2) : même architecture, même split,
mêmes graines, même budget ; seuls l'optimiseur et le learning rate changent."""
import numpy as np
from src.model import MLP
from src.optimizers import SGD, Momentum, Adam
from src.train import train


def make_optimizer(name, lr):
    if name == "SGD":
        return SGD(lr=lr)
    if name == "Momentum":
        return Momentum(lr=lr, beta=0.9)
    if name == "Adam":
        return Adam(lr=lr)
    raise ValueError(name)


def run_one(name, lr, seed, data, epochs=100, batch_size=32):
    """Un entraînement. `seed` fixe à la fois l'initialisation des poids et l'ordre des mini-batchs,
    donc deux optimiseurs lancés avec la même seed partent du même point et voient les mêmes batchs."""
    model = MLP(seed=seed)
    opt = make_optimizer(name, lr)
    return train(model, opt, data, epochs=epochs, batch_size=batch_size, seed=seed)


def summarize(h, thr=0.2):
    """Indicateurs d'une courbe d'entraînement (une seed, un lr)."""
    lt, lv = np.array(h["loss_train"]), np.array(h["loss_val"])
    below = np.where(lt < thr)[0]
    diverged = (not np.all(np.isfinite(lt))) or lt[-1] > lt[0]
    return {
        "loss_train_ep5": lt[5],                                   # vitesse initiale
        "epochs_to_thr": float(below[0]) if len(below) else np.nan,  # vitesse : 1re époque sous thr
        "loss_train_final": lt[-1],
        "loss_val_final": lv[-1],
        "loss_val_min": lv.min(),
        "epoch_val_min": int(lv.argmin()),
        "acc_val_final": h["acc_val"][-1],
        "gap_final": lv[-1] - lt[-1],                              # écart de généralisation
        "frac_up": float(np.mean(np.diff(lt) > 0)),                # stabilité : % d'époques où la loss monte
        "diverged": bool(diverged),
    }


def lr_sweep(data, grids, seeds, epochs=100, batch_size=32, verbose=True):
    """grids : {nom_optimiseur: [lr, ...]}. Retourne {(nom, lr, seed): history}."""
    out = {}
    for name, lrs in grids.items():
        for lr in lrs:
            for s in seeds:
                out[(name, lr, s)] = run_one(name, lr, s, data, epochs, batch_size)
            if verbose:
                print(f"{name:9s} lr={lr:<8g} terminé")
    return out