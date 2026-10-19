"""Band power: the size of the rhythms, which the fractal measures are nearly blind to.

Log-variance of every channel in the mu (8-13 Hz), beta (13-30 Hz) and broad bands: the same
channels and bands as the fractal intent signature. "Power" and "power + fractal shape" therefore
differ only by the shape measures, which is what the follow-up asks about.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .fractal import BANDS, _band


class BandPower(BaseEstimator, TransformerMixin):
    """EEG epochs (trials x channels x times) -> log band power (trials x bands*channels)."""

    def __init__(self, sfreq: float = 128.0, bands=tuple(BANDS)):
        self.sfreq = sfreq
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        feats = [np.log(np.var(_band(X, BANDS[b], self.sfreq), axis=-1) + 1e-20) for b in self.bands]
        return np.concatenate(feats, axis=1)
