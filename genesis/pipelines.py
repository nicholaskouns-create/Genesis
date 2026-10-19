"""The decoders under test.

Pre-registered benchmark (H1-H3):

Genesis                      fractal intent signature -> standardize -> coherence projection at
                             Omega_c = 47/125 (Fisher) -> shrinkage LDA
Genesis, fraction 0.25/0.50  the same with another occupancy fraction            (H2)
Genesis, no projection       the same with the projection removed                 (H2)
Genesis, variance projection unsupervised coherence ranking instead of Fisher
Correlation dimension + LDA  the single measure that worked in late-stage ALS
CSP + LDA                    the classic motor-imagery decoder                    (H1 baseline)
Tangent space + LR           the Riemannian decoder, strongest classical baseline (H1 baseline)

Follow-up (H4-H5): does fractal shape add anything on top of power? Each pair differs only by the
fractal signature (same channels, bands, standardization and shrinkage LDA):

Band power                   log band power, mu / beta / broad, every channel
Band power + fractal shape   the same plus the fractal intent signature           (H4)
Tangent space                Riemannian tangent-space covariance features
Tangent space + fractal shape the same plus the fractal intent signature          (H5)

All decoders see the same epochs: band-passed 8-32 Hz, resampled to 128 Hz.
"""
from __future__ import annotations

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler

from .coherence import OMEGA_C, CoherenceProjection
from .fractal import BAND_MEASURES, BANDS, BROAD_ONLY, FractalIntentSignature
from .power import BandPower

SFREQ = 128.0
GENESIS = "Genesis"
BASELINES = ("CSP + LDA", "Tangent space + LR")
FOLLOWUP_PAIRS = (("Band power + fractal shape", "Band power"), ("Tangent space + fractal shape", "Tangent space"))


def _lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def _tangent_space():
    return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"))


def _tangent_space_lr():
    return make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(C=1.0, max_iter=1000))


def _standardized_lda(*features):
    union = features[0] if len(features) == 1 else make_union(*features)
    return make_pipeline(union, StandardScaler(), _lda())


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
        "Tangent space + LR": _tangent_space_lr(),
    }


def build_followup_pipelines(sfreq=SFREQ, n_jobs=1) -> dict:
    def fis():
        return FractalIntentSignature(sfreq=sfreq, n_jobs=n_jobs)
    return {
        "Band power": _standardized_lda(BandPower(sfreq=sfreq)),
        "Band power + fractal shape": _standardized_lda(BandPower(sfreq=sfreq), fis()),
        "Tangent space": _standardized_lda(_tangent_space()),
        "Tangent space + fractal shape": _standardized_lda(_tangent_space(), fis()),
        "Tangent space + LR": _tangent_space_lr(),
        GENESIS: genesis_pipeline(sfreq=sfreq, n_jobs=n_jobs),
    }
