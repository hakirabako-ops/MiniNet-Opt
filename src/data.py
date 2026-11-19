"""Génération, split et standardisation des données (partie A1)."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def generate_data(n_samples=1500, noise=0.20, seed=42):
    """Génère le jeu make_moons. Retourne X (n, 2) et y (n,)."""
    return make_moons(n_samples=n_samples, noise=noise, random_state=seed)


def split_data(X, y, seed=42, val_size=0.20, test_size=0.20):
    """Split stratifié train / val / test (60/20/20 par défaut)."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (val_size + test_size),
        random_state=seed, stratify=y_tmp)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def standardize(X_train, *others):
    """Standardise avec les statistiques du train uniquement.
    Retourne (X_train_std, *others_std, mu, sigma)."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    out = [(A - mu) / sigma for A in (X_train, *others)]
    return (*out, mu, sigma)


def load_data(n_samples=1500, noise=0.20, seed=42):
    """Pipeline complet. Les labels sont mis en colonnes (n, 1)."""
    X, y = generate_data(n_samples, noise, seed)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, seed)
    X_tr, X_va, X_te, mu, sigma = standardize(X_tr, X_va, X_te)
    col = lambda v: v.reshape(-1, 1).astype(float)
    return {
        "X_train": X_tr, "y_train": col(y_tr),
        "X_val": X_va, "y_val": col(y_va),
        "X_test": X_te, "y_test": col(y_te),
        "mu": mu, "sigma": sigma,
    }