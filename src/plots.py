"""Courbes et visualisations."""
import numpy as np


def stack(histories, key):
    """Liste d'historiques -> tableau (n_seeds, n_epochs+1)."""
    return np.array([h[key] for h in histories])


def plot_mean_std(ax, histories, key, label=None, color=None, logy=False):
    """Moyenne sur les seeds, bande = ± 1 écart-type."""
    A = stack(histories, key)
    m, s = A.mean(axis=0), A.std(axis=0)
    x = np.arange(A.shape[1])
    line, = ax.plot(x, m, label=label, color=color)
    ax.fill_between(x, m - s, m + s, alpha=0.2, color=line.get_color())
    if logy:
        ax.set_yscale("log")