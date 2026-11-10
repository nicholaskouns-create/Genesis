# Moment-selector transport: three tests, three null results

Run 2026-11-10 UTC. Every rule and hypothesis below was written, timestamped and hashed before the
signals it concerns were analysed. Nothing here has been trimmed.

**Read this first.** Motor intent is in the EEG signal and can be read: in healthy subjects, existing
decoders reach 0.82 to 0.94 in these runs. In the 50 acute stroke patients of the published Genesis
benchmark (`results/summary.md`), every decoder was close to chance. That is where the problem is
still open, for all methods.

| decoder (published Genesis benchmark, ROC-AUC) | healthy, 9 subjects | acute stroke, 50 patients |
|---|---|---|
| Tangent space + LR | 0.863 | 0.550 |
| CSP + LDA | 0.834 | 0.539 |
| Genesis | 0.623 | 0.522 |

The tests below close one candidate: the E47 moment selector as the decoder. They do not measure the
goal of restoring movement, and they do not show that existing methods have solved it for patients.

**Question.** The E47 moment-selector theorem builds the selector from a carrier's own first two
moments: K = (C − (μ − σ))(C − (μ + σ)). On the spin-2 cube this gives the sectors 6 and 30 and the
fraction 47/125. The earlier Genesis benchmark carried the *number* 47/125 onto EEG and found no
support (H1–H3). These runs carry the *selector* instead: each subject's own μ and σ, with the kept
fraction left as an output.

**Answer.** Not supported, in all three tests. The selector did not read motor intent from EEG.

| test | data | hypotheses | outcome |
|---|---|---|---|
| 1 · covariance operator | 9 subjects, 22 electrodes | M1–M4, α = 0.0125 | 0 of 4 supported |
| 2 · log-covariance operator (exploratory) | same 9 subjects | M1′–M4′ | 0 of 4 supported |
| 3 · scalp-sphere Casimir | 4 human subjects, 59 electrodes, untouched data | one exact test, α = 0.05 | not supported, p = 0.974 |

## Test 1 · the selector on each subject's covariance operator

`eeg22/PREREGISTRATION.md` · 2026-11-10T05:59:18Z · sha256 `1277ec1b…cb5d5c`

BCI Competition IV 2a (BNCI2014-001), training sessions, left versus right hand, 144 trials per
subject, 5-fold cross-validation, ROC-AUC. Kept sector: eigenvectors with |K(λ)| ≤ τσ², τ = 1/2.
The rule reproduces E47 exactly on the cube for every τ < 3/4.

Carrier A is the standardized Genesis fractal signature (286 dimensions). Carrier B is the 22 channels.

| decoder | mean ROC-AUC |
|---|---|
| Tangent space + LR | 0.851 |
| CSP + LDA | 0.817 |
| 22-channel log-variance | 0.746 |
| Genesis, no projection | 0.622 |
| Genesis, Fisher 47/125 | 0.615 |
| **A · moment selector** | **0.585** |
| A · top-variance, same rank | 0.527 |
| A · random subspace, same rank | 0.525 |
| B · random subspace, same rank | 0.578 |
| B · top-variance, same rank | 0.550 |
| **B · moment selector** | **0.545** |

| hypothesis | Δ AUC | wins | p | |
|---|---|---|---|---|
| M1 · A selector > no projection | −0.037 | 4/9 | 0.850 | not supported |
| M2 · A selector > random, same rank | +0.060 | 7/9 | 0.037 | not supported at α = 0.0125 |
| M3 · B selector > random, same rank | −0.033 | 3/9 | 0.898 | not supported |
| M4 · B selector > CSP | −0.272 | 0/9 | 1.000 | not supported |

The rebuilt pipeline matches the published Genesis benchmark on the shared decoders (no projection
0.622 here, 0.624 published; Fisher 47/125 0.615 here, 0.623 published), on one session instead of two.

**Why.** On carrier A, μ = 1 and σ ≈ 4.9. So μ − σ is negative, below the spectrum of a positive
operator, and the lower sector was empty in every fold. The selector kept about 4 of 286 directions
(fraction 0.013), all near μ + σ. On carrier B no eigenvalue fell inside the τ band in any fold and
the fallback (the two nearest eigenvectors) was used every time. On the cube σ/μ = 12/18; for
SU(2) tensor powers σ/μ = √(2(n − 1)/(3n)) < 1 always.

Sensitivity (τ = 1/4 and τ = 0.74) is in `eeg22/run.log`; neither changes an outcome.

## Test 2 · the same selector on log C (exploratory)

`eeg22/PREREG_2_EXPLORATORY.md` · 2026-11-10T06:09:02Z · sha256 `0988d21b…598b77`

Written after test 1 and before this run. One change: the operator is the matrix logarithm of C,
whose spectrum is two-sided. Declared exploratory.

| | A · selector | A · random | B · selector | B · random | B · CSP |
|---|---|---|---|---|---|
| mean ROC-AUC | 0.505 | 0.573 | 0.611 | 0.676 | 0.817 |

M1′ Δ = −0.117 (p = 0.963) · M2′ Δ = −0.068 (p = 0.951) · M3′ Δ = −0.065 (p = 0.963) ·
M4′ Δ = −0.205 (p = 1.000). None supported.

On carrier B the selection became two-sided (about 5.7 low and 1.9 high eigenvectors) and the kept
fraction came out at 0.345 ± 0.060. The preregistration states that no value of the kept fraction
counts as support or failure.

## Test 3 · the Casimir on the scalp sphere

`sphere59/PREREGISTRATION_SPHERE.md` · 2026-11-10T06:41:35Z · sha256 `c8570f41…5b6453`

The one carrier where the same operator acts: rotations act on the scalp sphere, and the Casimir L²
has eigenvalue ℓ(ℓ + 1) on degree-ℓ spherical harmonics. Truncated at ℓ = 6 the spectrum is
0, 2, 6, 12, 20, 30, 42. Data: BCI Competition IV dataset 1, 59 electrodes, 200 trials per subject,
not used in any earlier run. Subjects a, b, f, g are human; c, d, e are the competition's artificial
sets and sit outside the test.

μ and σ are the mean and standard deviation of the Casimir in the subject's own state (the share
of harmonic power in each degree, training folds only). Kept sector: the degrees nearest μ − σ and
μ + σ. Primary test: is the selected sector better than a degree set chosen at random from the 28
possible sets? Exact null by enumeration over the four human subjects.

| subject | classes | μ | σ | selected ℓ | selected AUC | rank of 28 | median set | all 49 harmonics | 59 channels | CSP | tangent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | left / foot | 8.66 | 9.91 | 0, 4 | 0.621 | 25 | 0.808 | 0.908 | 0.941 | 0.904 | 0.977 |
| b | left / right | 16.27 | 13.07 | 1, 5 | 0.461 | 27 | 0.525 | 0.635 | 0.801 | 0.834 | 0.832 |
| f | left / foot | 9.36 | 10.35 | 0, 4 | 0.748 | 19 | 0.770 | 0.886 | 0.916 | 0.962 | 0.982 |
| g | left / right | 8.63 | 9.64 | 0, 4 | 0.577 | 22 | 0.626 | 0.696 | 0.948 | 0.962 | 0.980 |

**Primary test: T = 0.602, null mean 0.673, p = 0.974. Not supported.**

Human means: selected 0.602 · fixed pair ℓ = 2, 5: 0.696 · all 49 harmonics 0.781 ·
59 channels 0.902 · CSP 0.915 · tangent space 0.943.

**Why.** In three of four subjects σ > μ again, so μ − σ < 0 and the rule chose ℓ = 0, which holds
under 1% of the power after average referencing. The single degrees that decoded best were ℓ = 2
and ℓ = 3 (human means 0.695 and 0.696). The harmonic basis itself lost signal: 0.781 against 0.902
for the raw channels.

Artificial sets, outside the test: c selected ℓ = 0, 4 (AUC 0.538), d selected ℓ = 2, 5 (0.567),
e selected ℓ = 3, 6 (0.564). The preregistration states that no selected value counts.

Limits: four human subjects. Electrode geometry was reconstructed from the file's flat coordinates
with one fitted scale (RMS error 0.10 rad over 48 electrodes). The electrodes cover a cap, so the
rotation symmetry holds only approximately and degrees mix under the ridge fit.

## What the same runs established about the theorem

`algebra/` holds the checks. The census (1, 9, 25, 28, 27, 22, 13), N = 125, μ = 18, σ² = 144, the
roots 6 and 30, and dimension 47 agree in Python (exact rationals), SymPy, Node (BigInt), PARI/GP,
Maxima, GAP, SWI-Prolog, Z3 and Coq 8.18. `M.v` proves seven theorems, closed under the global
context. Lean was not run.

`carriers.py` scans other carriers exactly for μ − σ and μ + σ both landing on the spectrum:

| carrier | dimension | σ² | kept | fraction |
|---|---|---|---|---|
| hypercube Q₄ | 16 | 4 | 8 | 0.5000 |
| hypercube Q₉ | 512 | 9 | 168 | 0.3281 |
| hypercube Q₁₆ | 65 536 | 16 | 16 016 | 0.2444 |
| Hamming H(18, 3) | 3¹⁸ | 36 | 94 943 232 | 0.2451 |
| Johnson J(6, 3) | 20 | 9 | 10 | 0.5000 |
| Johnson J(12, 6) | 924 | 36 | 286 | 0.3095 |
| Johnson J(20, 10) | 184 756 | 100 | 40 052 | 0.2168 |
| S₉, transposition class sum | 362 880 | 36 | 42 912 | 0.1183 |
| SU(2) spin 2, cube | 125 | 144 | 47 | 0.3760 |
| SU(2) spin 6, cube | 2 197 | 7 056 | 314 | 0.1429 |

The selector is exact on carriers where a group acts and eigenvalues label its sectors. Each has
its own fraction.

## Not run

- The 50 acute stroke patients (Liu2024). The host was unreachable from the sandbox that ran these.
- Session-to-session stability. A Casimir sector is invariant under the group; left versus right is
  an orientation. Whether the selector finds what stays fixed in one person's signal across days is
  a separate question that BNCI2014-001's two sessions can answer. It has not been tested.

## Reproduce

The scripts are stored exactly as they ran, including sandbox paths. Replace the `sys.path` line
with the repository root and pass the data directory:

```
python results/moment-selector/eeg22/run_transport.py      <dir with A01T.npz … A09T.npz>
python results/moment-selector/eeg22/run_transport_log.py  <same>
python results/moment-selector/sphere59/run_sphere.py      <dir with BCICIV_calib_ds1a.mat … g.mat>
python results/moment-selector/algebra/carriers.py
coqc   results/moment-selector/algebra/M.v
```

Data sources used: `bregydoc/bcidatasetIV2a` (GitHub) and `robert1208/2026-HIP-Final-Project`
(GitHub LFS). Both are third-party mirrors of the public competition files. Environment: Python 3.13,
NumPy 2.5, SciPy 1.18, scikit-learn, MNE, pyRiemann, antropy.
