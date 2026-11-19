"""Optimisation sous contrainte ||theta||_2 <= R (partie E).

Convention du sujet : ||theta||_2^2 = sum_l ||W^(l)||_F^2, biais exclus.
Les poids sont lus via model.linear_layers() (attribut .W de chaque couche).
"""
import numpy as np


# ---------------------------------------------------------------- normes
def params_norm(parameters):
    """Norme euclidienne globale d'une liste de tableaux."""
    return float(np.sqrt(sum(float(np.sum(p ** 2)) for p in parameters)))


def model_weights(model):
    """Liste des matrices W du réseau (sans les biais)."""
    return [l.W for l in model.linear_layers()]


def weights_norm(model):
    """||theta||_2 au sens du sujet (biais exclus)."""
    return params_norm(model_weights(model))


# ---------------------------------------------------------------- projection
def project_l2_ball(parameters, R):
    """Projection sur C = {theta : ||theta||_2 <= R} (signature du sujet, E1).

    P_C(z) = z si ||z||_2 <= R, sinon R * z / ||z||_2.
    Renvoie de nouveaux tableaux, les entrées ne sont pas modifiées.
    """
    norm = params_norm(parameters)
    if norm <= R:
        return [p.copy() for p in parameters]
    return [p * (R / norm) for p in parameters]


def project_model_(model, R):
    """Projette sur place les poids du modèle. Renvoie la norme avant projection."""
    norm = weights_norm(model)
    if norm > R:
        scale = R / norm
        for l in model.linear_layers():
            l.W *= scale
    return norm


# ---------------------------------------------------------------- faisabilité
def feasibility_residual(norm, R):
    """r_feas = max(0, ||theta||_2 - R)."""
    return max(0.0, norm - R)


def is_active(norm, R, rel_tol=1e-3):
    """Contrainte active si | ||theta||_2 - R | <= eps, avec eps = rel_tol * R."""
    return abs(norm - R) <= rel_tol * R


def kkt_multiplier(model, X, y):
    """Estimation du multiplicateur mu de la contrainte en fin d'entraînement.

    Sur le bord, KKT donne grad_W F + mu * W = 0. Par moindres carrés :
    mu = -<grad_W F, W> / ||W||^2 (tronqué à 0). Renvoie aussi cos(-grad F, W),
    qui vaut 1 en un point KKT exact : c'est le contrôle de qualité de l'estimation.
    """
    model.eval()
    model.loss_and_grad(X, y)
    W = model_weights(model)
    G = [l.dW for l in model.linear_layers()]
    dot = sum(float(np.sum(g * w)) for g, w in zip(G, W))
    mu = max(0.0, -dot / params_norm(W) ** 2)
    cos = -dot / (params_norm(G) * params_norm(W) + 1e-300)
    return mu, cos


# ---------------------------------------------------------------- gradient projeté
class ProjectedOptimizer:
    """Pas de l'optimiseur de base, puis projection des poids W sur la boule.

        theta_tilde = pas de l'optimiseur de base,   theta = P_C(theta_tilde)

    Avec SGD, c'est exactement le gradient projeté du sujet. Avec Adam, on
    projette le pas d'Adam (« Adam projeté »). Les biais ne sont jamais projetés.

    S'utilise à la place de l'optimiseur dans train(...) :
        opt = ProjectedOptimizer(Adam(lr=0.003), model, R)

    Compatible avec les deux façons de mettre à jour les paramètres :
    l'optimiseur modifie directement les tableaux du modèle, ou il reçoit
    le vecteur aplati de get_params() que train.py remet ensuite dans le modèle.

    Journal : norme avant projection et projection effective, à chaque pas.
    """

    def __init__(self, base, model, R):
        self.base, self.model, self.R = base, model, R
        self.pre_norms, self.projected = [], []
        self._mask = None

    def __getattr__(self, name):
        # lr, état d'Adam, etc. restent accessibles comme sur l'optimiseur de base
        return getattr(self.base, name)

    def _weight_mask(self):
        """Repère les composantes de get_params() qui sont des poids W (et non des biais)."""
        if self._mask is None:
            saved = self.model.get_params().copy()
            self.model.set_params(np.zeros_like(saved))
            for l in self.model.linear_layers():
                l.W[...] = 1.0
            self._mask = self.model.get_params() == 1.0
            self.model.set_params(saved)
        return self._mask

    def step(self, *args, **kwargs):
        out = self.base.step(*args, **kwargs)
        n = self.model.get_params().size
        flat = [a for a in args if isinstance(a, np.ndarray) and a.ndim == 1 and a.size == n]
        if flat:
            # cas « vecteur aplati » : on projette les composantes W de ce vecteur
            v, m = flat[0], self._weight_mask()
            pre = float(np.linalg.norm(v[m]))
            if pre > self.R:
                v[m] *= self.R / pre
        else:
            pre = None
        # cas « tableaux du modèle » (sans effet si déjà dans la boule)
        pre_model = project_model_(self.model, self.R)
        pre = pre_model if pre is None else pre
        self.pre_norms.append(pre)
        self.projected.append(pre > self.R)
        return out

    def projection_rate(self):
        """Part des pas où la projection a réellement modifié les poids."""
        return float(np.mean(self.projected)) if self.projected else 0.0


def ProjectedGradient(model, lr, R):
    """Gradient projeté au sens strict du sujet : SGD puis projection."""
    from .optimizers import SGD
    return ProjectedOptimizer(SGD(lr=lr), model, R)