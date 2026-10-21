"""Tests: the measures on signals whose answer is known, the projector and contraction identities,
a synthetic motor-imagery recovery, and the full MOABB benchmark on synthetic data (no download)."""
import json

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

import antropy as ant
from genesis import OMEGA_C, CoherenceProjection, FractalIntentSignature, build_pipelines, contraction_power, correlation_dimension

FS = 128.0
RNG = np.random.default_rng(47)


def test_fractal_measures_on_known_signals():
    t = np.arange(512) / FS
    sine = np.sin(2 * np.pi * 10 * t)
    noise = RNG.standard_normal(512)
    assert abs(ant.higuchi_fd(noise, kmax=10) - 2.0) < 0.1          # white noise fills the plane
    slow = np.sin(2 * np.pi * 1 * t)                                 # smooth on every Higuchi scale (k <= 10 samples)
    assert abs(ant.higuchi_fd(slow, kmax=10) - 1.0) < 0.05          # a smooth curve is one-dimensional
    assert 0.8 < correlation_dimension(sine) < 1.3                  # a limit cycle has D2 = 1
    assert correlation_dimension(noise) > 3.0                       # noise fills the embedding


def test_projector_and_residual_identities():
    X = RNG.standard_normal((60, 125))
    y = np.repeat([0, 1], 30)
    for criterion in ("fisher", "variance"):
        cp = CoherenceProjection(criterion=criterion).fit(X, y)
        P = cp.projector()
        assert cp.k_ == 47 and np.linalg.matrix_rank(P) == 47
        assert np.allclose(P @ P, P, atol=1e-12) and np.allclose(P, P.T, atol=1e-12)
        R = cp.residual(X)
        assert np.allclose(X, X @ P + R, atol=1e-12)
        assert np.abs(R @ cp.basis_).max() < 1e-10                   # the residual is orthogonal to E
        occ = cp.occupancy(X)
        assert ((occ >= 0) & (occ <= 1 + 1e-12)).all()


def test_gate_threshold_is_chance_occupancy():
    """For direction-less signatures a 47-of-125 subspace holds 47/125 of the energy on average."""
    cp = CoherenceProjection(criterion="variance").fit(RNG.standard_normal((200, 125)))
    occ = cp.occupancy(RNG.standard_normal((20000, 125)))
    assert abs(occ.mean() - OMEGA_C) < 0.003


def test_contraction_converges_to_the_projector():
    X = RNG.standard_normal((40, 30))
    P = CoherenceProjection(criterion="variance").fit(X).projector()
    for eps in (0.1, 0.5, 1.0):
        for n in (1, 5, 20):
            err = np.linalg.norm(contraction_power(P, eps, n) - P, 2)
            assert abs(err - (1 - eps) ** n) < 1e-10


def synthetic_motor_imagery(n_per_class=40, seed=0):
    """Eight channels; a 10 Hz mu rhythm over C3 and C4 desynchronises contralateral to the imagined hand."""
    rng = np.random.default_rng(seed)
    names = ["FC3", "C3", "CP3", "Cz", "FC4", "C4", "CP4", "Pz"]
    t = np.arange(int(4 * FS)) / FS
    X, y = [], []
    for label in ("left_hand", "right_hand"):
        for _ in range(n_per_class):
            pink = np.cumsum(rng.standard_normal((8, t.size)), axis=1)
            pink = (pink - pink.mean(axis=1, keepdims=True)) * 0.05 + rng.standard_normal((8, t.size))
            for ch, side in ((1, "right_hand"), (5, "left_hand")):        # C3 serves the right hand, C4 the left
                amp = 0.4 if label == side else 1.0
                pink[ch] += 2.0 * amp * np.sin(2 * np.pi * (10 + rng.normal(0, 0.5)) * t + rng.uniform(0, 2 * np.pi))
            X.append(pink)
            y.append(label)
    return np.asarray(X), np.asarray(y), names


def test_every_decoder_recovers_synthetic_motor_imagery():
    X, y, _ = synthetic_motor_imagery()
    cv = StratifiedKFold(5, shuffle=True, random_state=47)
    scores = {name: cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean() for name, pipe in build_pipelines(n_jobs=1).items()}
    print(json.dumps({k: round(v, 3) for k, v in scores.items()}, indent=1))
    for name, s in scores.items():
        assert s > 0.8, f"{name} failed to decode the synthetic mu desynchronisation (AUC {s:.3f})"


def test_signature_shape_and_cache():
    X, _, names = synthetic_motor_imagery(n_per_class=3)
    full = FractalIntentSignature(n_jobs=1).fit_transform(X)
    assert full.shape == (6, 13 * len(names))                          # 4 measures x 3 bands + D2, per channel
    only = FractalIntentSignature(measures=("correlation",), bands=("broad",), n_jobs=1).fit_transform(X)
    assert np.array_equal(only, full[:, -len(names):])


def test_benchmark_end_to_end_without_download(tmp_path):
    from benchmark.run import main
    main(["--datasets", "fake", "--out", str(tmp_path), "--n-jobs", "1"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    d = summary["datasets"]["FakeDataset"]
    assert set(d["auc"]) == set(build_pipelines(n_jobs=1))
    for key in ("H1_genesis_beats_standard_decoders", "H2_47_over_125_beats_other_fractions", "H3_gate_predicts_reliability"):
        assert key in d
    assert (tmp_path / "summary.md").read_text().startswith("# Genesis benchmark")


def test_band_power_reads_rhythm_size():
    """Log band power of a 10 Hz rhythm of amplitude A is log(A^2 / 2) in the mu band; doubling A adds log 4."""
    from genesis import BandPower
    t = np.arange(int(4 * FS)) / FS
    X = np.stack([np.stack([a * np.sin(2 * np.pi * 10 * t)]) for a in (1.0, 2.0)])   # 2 trials x 1 channel
    bp = BandPower(bands=("mu",)).fit_transform(X)
    assert bp.shape == (2, 1)
    assert abs(bp[0, 0] - np.log(0.5)) < 0.05
    assert abs((bp[1, 0] - bp[0, 0]) - np.log(4.0)) < 1e-6


def test_fractal_measures_ignore_rhythm_size():
    """Why the fractal signature misses imagery: scaling a signal leaves every fractal measure unchanged."""
    X, _, _ = synthetic_motor_imagery(n_per_class=2)
    a = FractalIntentSignature(n_jobs=1).fit_transform(X)
    b = FractalIntentSignature(n_jobs=1).fit_transform(3.0 * X)
    assert np.allclose(a, b, atol=1e-6)


def test_followup_pairs_differ_only_by_the_signature():
    from genesis import FOLLOWUP_PAIRS, build_followup_pipelines
    X, y, _ = synthetic_motor_imagery()
    pipes = build_followup_pipelines(n_jobs=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=47)
    for with_shape, without in FOLLOWUP_PAIRS:
        assert cross_val_score(pipes[with_shape], X, y, cv=cv, scoring="roc_auc").mean() > 0.8
        assert cross_val_score(pipes[without], X, y, cv=cv, scoring="roc_auc").mean() > 0.8
        n_with = pipes[with_shape][:-2].fit_transform(X, y).shape[1]
        n_without = pipes[without][:-2].fit_transform(X, y).shape[1]
        assert n_with - n_without == 13 * X.shape[1]                     # exactly the fractal signature added


def test_followup_end_to_end_without_download(tmp_path):
    from benchmark.run import main
    main(["--suite", "shape-on-power", "--datasets", "fake", "--out", str(tmp_path), "--n-jobs", "1"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    comps = summary["datasets"]["FakeDataset"]["comparisons"]
    assert [c["title"][:2] for c in comps] == ["H4", "H5"]
    assert all(len(c["ci95"]) == 2 for c in comps)
    assert (tmp_path / "summary.md").read_text().startswith("# Does fractal shape add anything")
