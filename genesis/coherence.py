"""The projection-and-residual step of the August 2026 restoration architecture, applied to intent.

From "The Kouns Fixed-Point Medical Restoration Architecture" (MEDICAL ONLY // NO PII // RESTORATION
FOCUS), Appendix B, Minimal Restoration Algorithm:

    normalize -> project into the invariant carrier and compute the residual -> contraction
    -> emergence gate -> verify

Carried over to a decoder:

* The carrier is the standardized signature space R^d.
* The invariant subspace E keeps the fraction Omega_c = 47/125 of directions that carry the most
  coherent intent. Directions are ranked by Fisher's criterion (between-class over within-class
  variance, the rule the ALS study used to choose channels) or, unsupervised, by variance.
* P is the orthogonal projector onto E and H_perp = I - P the residual projector; x = Px + H_perp x.
* With K = I - P (K vanishes exactly on E), the contraction Gamma = I - eps K^2 satisfies
  Gamma^n = P + (1 - eps)^n (I - P), so iterating it converges to P, as Gamma = I - K^2/99144 does in
  the E47 construction. The decoder applies the limit P directly.
* Emergence gate C = Theta[ occupancy(x) - Omega_c ] with occupancy(x) = |Px|^2 / |x|^2. For a
  direction-less (isotropic) signature the expected occupancy of a k-of-d subspace is k/d, which is
  Omega_c by construction, so the gate passes exactly those trials whose signature concentrates in
  the intent subspace more than chance.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

OMEGA_C = 47 / 125


def fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature Fisher criterion: between-class variance over pooled within-class variance."""
    classes = np.unique(y)
    mu = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        between += len(Xc) * (Xc.mean(axis=0) - mu) ** 2
        within += ((Xc - Xc.mean(axis=0)) ** 2).sum(axis=0)
    return between / (within + 1e-12)


class CoherenceProjection(BaseEstimator, TransformerMixin):
    """Project standardized signatures onto the Omega_c fraction of most coherent directions.

    transform returns the coordinates of Px in an orthonormal basis of E (k = round(fraction * d)).
    occupancy / gate / residual expose the rest of the restoration step for analysis.
    """

    def __init__(self, fraction: float = OMEGA_C, criterion: str = "fisher"):
        self.fraction = fraction
        self.criterion = criterion

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        d = X.shape[1]
        k = int(np.clip(round(self.fraction * d), 1, d))
        if self.criterion == "fisher":
            if y is None:
                raise ValueError("criterion='fisher' needs labels")
            keep = np.argsort(fisher_scores(X, np.asarray(y)))[::-1][:k]
            Q = np.zeros((d, k))
            Q[keep, np.arange(k)] = 1.0
        elif self.criterion == "variance":
            Xc = X - X.mean(axis=0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            Q = Vt[:k].T
            if Q.shape[1] < k:  # fewer samples than dimensions: complete the basis
                extra = np.linalg.qr(np.column_stack([Q, np.random.default_rng(47).standard_normal((d, k - Q.shape[1]))]))[0]
                Q = extra[:, :k]
        else:
            raise ValueError(f"unknown criterion {self.criterion!r}")
        self.basis_ = Q
        self.k_, self.d_ = k, d
        return self

    def projector(self) -> np.ndarray:
        return self.basis_ @ self.basis_.T

    def transform(self, X):
        return np.asarray(X, float) @ self.basis_

    def residual(self, X):
        X = np.asarray(X, float)
        return X - (X @ self.basis_) @ self.basis_.T

    def occupancy(self, X):
        X = np.asarray(X, float)
        inside = ((X @ self.basis_) ** 2).sum(axis=1)
        return inside / np.maximum((X ** 2).sum(axis=1), 1e-300)

    def gate(self, X, threshold: float = OMEGA_C):
        return self.occupancy(X) >= threshold


def contraction_power(P: np.ndarray, eps: float, n: int) -> np.ndarray:
    """Gamma^n for Gamma = I - eps K^2 with K = I - P (K^2 = K because I - P is a projector)."""
    d = P.shape[0]
    G = np.eye(d) - eps * (np.eye(d) - P)
    return np.linalg.matrix_power(G, n)
