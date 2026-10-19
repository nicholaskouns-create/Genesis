"""The decoders under test.

Genesis                      fractal intent signature -> standardize -> coherence projection at
                             Omega_c = 47/125 (Fisher) -> shrinkage LDA
Genesis, fraction 0.25/0.50  the same with another occupancy fraction            (H2)
Genesis, no projection       the same with the projection removed                 (H2)
Genesis, variance projection unsupervised coherence ranking instead of Fisher
Correlation dimension + LDA  the single measure that worked in late-stage ALS
CSP + LDA                    the classic motor-imagery decoder                    (H1 baseline)
Tangent space + LR           the Riemannian decoder, strongest classical baseline (H1 baseline)

All decoders see the same epochs: band-passed 8-32 Hz, resampled to 128 Hz.
"""
from __future__ import annotations

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .coherence import OMEGA_C, CoherenceProjection
from .fractal import BAND_MEASURES, BANDS, BROAD_ONLY, FractalIntentSignature

SFREQ = 128.0
GENESIS = "Genesis"
BASELINES = ("CSP + LDA", "Tangent space + LR")


def _lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def genesis_pipeline(fraction=OMEGA_C, criterion="fisher", measures=BAND_MEASURES + BROAD_ONLY, bands=tuple(BANDS), sfreq=SFREQ, n_jobs=1):
    steps = [FractalIntentSignature(sfreq=sfreq, measures=measures, bands=bands, n_jobs=n_jobs), StandardScaler()]
    if fraction is not None:
        steps.append(CoherenceProjection(fraction=fraction, criterion=criterion))
    steps.append(_lda())
    return make_pipeline(*steps)


def build_pipelines(sfreq=SFREQ, n_jobs=1) -> dict:
    return {
        GENESIS: genesis_pipeline(sfreq=sfreq, n_jobs=n_jobs),
        "Genesis, fraction 0.25": genesis_pipeline(0.25, sfreq=sfreq, n_jobs=n_jobs),
        "Genesis, fraction 0.50": genesis_pipeline(0.50, sfreq=sfreq, n_jobs=n_jobs),
        "Genesis, no projection": genesis_pipeline(None, sfreq=sfreq, n_jobs=n_jobs),
        "Genesis, variance projection": genesis_pipeline(criterion="variance", sfreq=sfreq, n_jobs=n_jobs),
        "Correlation dimension + LDA": genesis_pipeline(None, measures=("correlation",), bands=("broad",), sfreq=sfreq, n_jobs=n_jobs),
        "CSP + LDA": make_pipeline(CSP(n_components=8), _lda()),
        "Tangent space + LR": make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000)),
    }
