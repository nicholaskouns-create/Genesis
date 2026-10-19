# Genesis

**Does the Genesis intent decoder read the intention to move from the brain better than the decoders
brain–computer interfaces use today?**

This repository answers that question with an open benchmark that anyone can rerun. It uses public EEG
recorded from 50 people in the first days after a stroke, and from 9 healthy people.

It began in Flatwoods, Kentucky, when a twelve-year-old wanted to help his friend Lonnie Lane move his
paralyzed side again. Rancho Los Amigos wrote back with a report on correcting foot drop by stimulating
the peroneal nerve. That idea is now a standard device. This is the test of the next one.

## What is tested

Project Genesis proposes a Bio-Recursive Mobility Network. Its Recursive Intelligence Engine reads motor
intent as fractal recursion signatures of brain signals. The August 2026 restoration architecture adds
a projection into an invariant subspace of occupancy Ωc = 47/125, a residual, and a gate. Both steps
are made measurable here:

| step | in this benchmark |
|---|---|
| Fractal intent signature | Higuchi, Katz and Petrosian fractal dimensions and the DFA exponent of every channel, in the mu (8–13 Hz), beta (13–30 Hz) and 8–32 Hz bands, plus the Grassberger–Procaccia correlation dimension |
| Normalize | each feature standardized on the training trials |
| Project into the invariant subspace | keep the 47/125 of directions with the highest Fisher criterion. P is the orthogonal projector, x = Px + (I − P)x, and the contraction I − εK² with K = I − P converges to P |
| Decode | shrinkage LDA on Px |
| Gate | pass when the occupancy \|Px\|²/\|x\|² ≥ 47/125, which is exactly the chance occupancy of a 47-of-125 subspace |

The baselines are CSP + LDA and Riemannian tangent space + logistic regression, the standard
motor-imagery pipelines of the MOABB benchmark. The ablations test the fraction and the
projection: fractions 0.25 and 0.50, no projection, an unsupervised (variance) projection, and the
correlation dimension alone.

A literal 125-site E47 embedding is not used, because no principled map exists from EEG channels to
the sites of V₂⊗V₂⊗V₂. The benchmark tests what carries over to a decoder: the occupancy fraction, the
projection and residual, and the gate.

## Hypotheses, written down before any real recording was seen

Primary data: **Liu2024**, 50 acute stroke patients, left- versus right-hand motor imagery (29
channels, 40 trials each). Secondary: **BNCI2014_001** (BCI Competition IV 2a), 9 healthy
participants.

All decoders see the same epochs: band-passed 8–32 Hz and resampled to 128 Hz. The score is ROC-AUC
from MOABB's within-session 5-fold cross-validation per subject and session, where 0.5 is chance. The
tests are MOABB's paired one-sided tests (permutation below 20 subjects, Wilcoxon from 20), with a
Bonferroni correction.

- **H1.** The Genesis decoder scores higher than CSP + LDA and higher than tangent space + LR (α = 0.05/2 each).
- **H2.** The 47/125 fraction scores higher than fractions 0.25 and 0.50 and higher than no projection (α = 0.05/3 each).
- **H3.** Within the Genesis decoder, trials that pass the occupancy gate are classified correctly
  more often than trials that fail it (one-sided Wilcoxon across subjects, α = 0.05).

A hypothesis that fails is reported as failing. This follows the August monograph's own falsification
criterion: the architecture fails its translation if the projected state does not predict
functional performance.

## Follow-up: does fractal shape add anything on top of power?

H1–H3 failed, and the reason is visible in the measures. Every fractal measure is unchanged when a
signal is scaled (tested), while imagining a hand movement mostly changes the size of the mu and beta
rhythms. The follow-up therefore asks whether shape adds information to size. These hypotheses were
written down before the follow-up was run.

Each pair below differs only by the fractal intent signature. Both decoders in a pair use the same
channels, bands, standardization and shrinkage LDA, on the same data, evaluation and statistics as
above.

- **H4.** Band power + fractal shape scores higher than band power alone (log power, mu / beta / 8–32 Hz, every channel).
- **H5.** Tangent space + fractal shape scores higher than tangent space alone (Riemannian covariance features).

Each test is one-sided at Bonferroni α = 0.05/2. The summary also reports the mean ΔAUC with a 95%
bootstrap interval over subjects, and in how many subjects the signature helped.

## Results

GitHub Actions writes them to [`results/summary.md`](results/summary.md) (H1–H3) and
[`results/shape-on-power/summary.md`](results/shape-on-power/summary.md) (H4–H5), along with the
per-subject scores, the statistics and the gate analysis.

## Run

```
pip install -r requirements.txt
python -m pytest -q        # measures, projector and contraction identities, synthetic motor imagery, MOABB end to end
python -m benchmark.run    # downloads Liu2024 and BNCI2014_001, writes results/
python -m benchmark.run --suite shape-on-power --out results/shape-on-power
```

`python -m benchmark.run --datasets fake` checks the whole pipeline without downloading anything.
GitHub Actions runs each suite once and skips any suite whose results are already committed; delete
a suite's `summary.json` to run it again.

## Layout

```
genesis/fractal.py      fractal intent signatures (per channel and band), cached per trial
genesis/coherence.py    projection at 47/125, residual, contraction, occupancy gate
genesis/power.py        log band power, the size the fractal measures cannot see
genesis/pipelines.py    the Genesis decoder, its ablations, the standard decoders, the follow-up pairs
benchmark/run.py        MOABB evaluation, statistics, gate analysis, summary
tests/                  tests that need no download
results/                written by GitHub Actions
```

## Credits

Project Genesis and the Kouns–Killion framework: Nicholas Shane Kouns. Benchmark: Claude (Anthropic),
19 October 2026.

Data and methods:
- *An EEG motor imagery dataset for brain computer interface in acute stroke patients*, Scientific Data, 2024, doi:10.1038/s41597-023-02787-8
- BCI Competition IV dataset 2a
- MOABB: Jayaram and Barachant, *MOABB: trustworthy algorithm benchmarking for BCIs*, 2018
- Fractal dimension decoding of motor imagery in ALS, PMC5539553
