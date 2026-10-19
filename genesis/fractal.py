"""Fractal intent signatures: the input layer of the Recursive Intelligence Engine, made measurable.

Project Genesis models motor intent as "fractal recursion signatures" of brain signals. Here the
signature of one trial is the vector of fractal measures of every EEG channel in each motor band:

    higuchi      Higuchi fractal dimension (kmax = 10)
    katz         Katz fractal dimension
    petrosian    Petrosian fractal dimension
    dfa          detrended-fluctuation exponent (alpha)
    correlation  Grassberger-Procaccia correlation dimension D2, the measure that decoded imagined
                 hand movement in five late-stage ALS patients (PMC5539553)

Bands: mu 8-13 Hz and beta 13-30 Hz (the rhythms that desynchronise during motor imagery) and the
paradigm's own broad band. The correlation dimension is computed on the broad band only.

The transform is stateless, so every trial's signature is computed once per process and cached:
the Genesis variants in one benchmark run share it.
"""
from __future__ import annotations

import hashlib

import antropy as ant
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, TransformerMixin

BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0), "broad": None}
BAND_MEASURES = ("higuchi", "katz", "petrosian", "dfa")
BROAD_ONLY = ("correlation",)
_CACHE: dict = {}


def correlation_dimension(x, m: int = 6, tau: int = 2, q=(0.05, 0.5), n_radii: int = 12) -> float:
    """Grassberger-Procaccia correlation dimension D2 of a 1-D signal.

    Delay-embed x in m dimensions with lag tau, take all pairwise distances, and fit the slope of
    log C(r) against log r, where C(r) is the fraction of pairs closer than r, over the scaling range
    between the q[0] and q[1] quantiles of the distances.
    """
    x = np.asarray(x, float)
    n = x.size - (m - 1) * tau
    if n < 16 or not np.isfinite(x).all() or x.std() == 0:
        return float("nan")
    emb = np.stack([x[i * tau:i * tau + n] for i in range(m)], axis=1)
    d = np.sort(pdist(emb))
    d = d[d > 0]
    lo, hi = np.quantile(d, q)
    r = np.geomspace(lo, hi, n_radii)
    c = np.searchsorted(d, r, side="right") / d.size
    return float(np.polyfit(np.log(r), np.log(c), 1)[0])


def _band(x, band, sfreq):
    if band is None:
        return x
    lo, hi = band
    sos = butter(4, [lo, min(hi, 0.45 * sfreq)], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _signature(trial: np.ndarray, sfreq: float) -> np.ndarray:
    """Full signature of one trial (channels x times): band measures then broad-band D2."""
    out = []
    for band in BANDS.values():
        xb = _band(trial, band, sfreq)
        xb = np.ascontiguousarray(xb, dtype=np.float64)
        for measure in BAND_MEASURES:
            if measure == "higuchi":
                vals = [ant.higuchi_fd(ch, kmax=10) for ch in xb]
            elif measure == "katz":
                vals = ant.katz_fd(xb, axis=-1)
            elif measure == "petrosian":
                vals = ant.petrosian_fd(xb, axis=-1)
            else:
                vals = [ant.detrended_fluctuation(ch) for ch in xb]
            out.extend(np.asarray(vals, float).ravel())
    out.extend(correlation_dimension(ch) for ch in trial)
    return np.asarray(out, float)


def signature_names(channels: list[str]) -> list[str]:
    names = [f"{b}:{m}:{c}" for b in BANDS for m in BAND_MEASURES for c in channels]
    return names + [f"broad:correlation:{c}" for c in channels]


class FractalIntentSignature(BaseEstimator, TransformerMixin):
    """EEG epochs (trials x channels x times) -> fractal intent signatures (trials x features).

    measures / bands select a subset of the full signature; the full signature is what is cached.
    """

    def __init__(self, sfreq: float = 128.0, measures=BAND_MEASURES + BROAD_ONLY, bands=tuple(BANDS), n_jobs: int = 1):
        self.sfreq = sfreq
        self.measures = measures
        self.bands = bands
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def _columns(self, n_channels: int) -> np.ndarray:
        cols, k = [], 0
        for b in BANDS:
            for m in BAND_MEASURES:
                if b in self.bands and m in self.measures:
                    cols.extend(range(k, k + n_channels))
                k += n_channels
        if "correlation" in self.measures and "broad" in self.bands:
            cols.extend(range(k, k + n_channels))
        if not cols:
            raise ValueError("FractalIntentSignature: no measure/band selected")
        return np.asarray(cols)

    def transform(self, X):
        X = np.asarray(X, float)
        keys = [(self.sfreq, hashlib.blake2b(np.ascontiguousarray(t).tobytes(), digest_size=16).hexdigest()) for t in X]
        todo = [i for i, k in enumerate(keys) if k not in _CACHE]
        if todo:
            sigs = Parallel(n_jobs=self.n_jobs)(delayed(_signature)(X[i], self.sfreq) for i in todo)
            for i, s in zip(todo, sigs):
                _CACHE[keys[i]] = s
        full = np.stack([_CACHE[k] for k in keys])
        F = full[:, self._columns(X.shape[1])]
        # a flat or broken channel can make a measure undefined; replace by the column median
        bad = ~np.isfinite(F)
        if bad.any():
            med = np.nanmedian(np.where(bad, np.nan, F), axis=0)
            F[bad] = np.take(np.nan_to_num(med), np.nonzero(bad)[1])
        return F
