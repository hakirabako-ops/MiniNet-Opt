"""Optimiseurs (NumPy) travaillant sur le vecteur plat theta.

Interface commune :  theta_new = optimizer.step(theta, grad)
Toutes les opérations (carré, racine, division) sont faites composante par composante.
"""
import numpy as np


class SGD:
    """theta_{k+1} = theta_k - alpha * g_k."""

    def __init__(self, lr=0.1):
        self.lr = lr

    def reset(self):
        pass

    def step(self, theta, grad):
        return theta - self.lr * grad


class Momentum:
    """v_{k+1} = beta v_k + (1 - beta) g_k ;  theta_{k+1} = theta_k - alpha v_{k+1}."""

    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta = lr, beta
        self.v = None

    def reset(self):
        self.v = None

    def step(self, theta, grad):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.beta * self.v + (1.0 - self.beta) * grad
        return theta - self.lr * self.v


class Adam:
    """m_k = b1 m_{k-1} + (1-b1) g ;  v_k = b2 v_{k-1} + (1-b2) g^2
    m_hat = m_k / (1 - b1^k) ;  v_hat = v_k / (1 - b2^k)
    theta_{k+1} = theta_k - alpha * m_hat / (sqrt(v_hat) + eps).
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = None
        self.v = None
        self.k = 0

    def reset(self):
        self.m, self.v, self.k = None, None, 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.k += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2
        m_hat = self.m / (1.0 - self.beta1 ** self.k)
        v_hat = self.v / (1.0 - self.beta2 ** self.k)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)