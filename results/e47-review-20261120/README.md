# Genesis benchmark and leakage audit

The preregistered comparison uses Liu2024 (50 acute-stroke participants) and BNCI2014-001 (9 healthy participants). The existing protocol is five-fold within-session trial-level ROC-AUC, with 8–32 Hz input epochs resampled to 128 Hz. Generalization to a new participant, session, or recording block requires a different split.

## Audit

The standardizer, supervised Fisher projection, LDA, CSP and tangent-space estimators fit inside their training folds. The gate analyzes held-out epochs with a fixed 47/125 threshold using the fitted projection. Its pass/fail split reports held-out accuracy conditional on the gate, alongside coverage.

The raw per-epoch fractal cache keys on the sample content and sample rate. Previously, undefined features were filled by taking a column median from the batch presented to `transform`. In a test fold this allowed the transformed value for one held-out trial to depend on other held-out trials and used different fill values from training. The corrected transformer computes the median on training features only, with zero as the fallback when a training column has no finite value. A regression test confirms that adding a held-out trial cannot change an existing held-out trial's imputation. The corrected suite passed 12 tests.

[Corrected source](https://github.com/nicholaskouns-create/Genesis/commit/4c477aba3c2b0b56d84c5212cf6faff312666078)

## Fresh corrected rerun

ROC-AUC is averaged across sessions within participants and then across participants. The counts compare each participant’s Genesis AUC to the comparator. A difference within 10⁻⁸ counts as a tie.

### BNCI2014-001 (9 participants)

Mean Genesis AUC was 0.623.

| Comparator | Mean AUC | Mean Genesis minus comparator | Genesis better / tied / worse |
| --- | ---: | ---: | ---: |
| CSP + LDA | 0.834 | -0.210 | 0 / 0 / 9 |
| Correlation dimension + LDA | 0.590 | +0.034 | 5 / 0 / 4 |
| Genesis, fraction 0.25 | 0.629 | -0.006 | 4 / 0 / 5 |
| Genesis, fraction 0.50 | 0.629 | -0.006 | 4 / 0 / 5 |
| Genesis, no projection | 0.624 | -0.001 | 6 / 0 / 3 |
| Genesis, variance projection | 0.614 | +0.010 | 4 / 0 / 5 |
| Tangent space + LR | 0.863 | -0.239 | 0 / 0 / 9 |

The gate passed 48.1% of epochs on average across participants. Mean conditional accuracy was 0.603 for passes and 0.592 for failures. Pass accuracy was higher for 4 participants and lower for 5, with 0 ties and 0 undefined differences.

### Liu2024 (50 participants)

Mean Genesis AUC was 0.522.

| Comparator | Mean AUC | Mean Genesis minus comparator | Genesis better / tied / worse |
| --- | ---: | ---: | ---: |
| CSP + LDA | 0.538 | -0.017 | 22 / 0 / 28 |
| Correlation dimension + LDA | 0.537 | -0.015 | 22 / 0 / 28 |
| Genesis, fraction 0.25 | 0.528 | -0.006 | 17 / 7 / 26 |
| Genesis, fraction 0.50 | 0.518 | +0.004 | 27 / 2 / 21 |
| Genesis, no projection | 0.513 | +0.009 | 25 / 4 / 21 |
| Genesis, variance projection | 0.507 | +0.015 | 28 / 3 / 19 |
| Tangent space + LR | 0.550 | -0.028 | 19 / 0 / 31 |

The gate passed 49.1% of epochs on average across participants. Mean conditional accuracy was 0.500 for passes and 0.537 for failures. Pass accuracy was higher for 19 participants and lower for 31, with 0 ties and 0 undefined differences.

## Interpretation

The corrected rerun reproduced 544 session-level decoder scores across 59 participants. Against the prior run, exactly five participant-average AUCs changed, all in Liu2024's variance-projection ablation, with a maximum absolute change of 0.0125. The gate table and Genesis and baseline AUCs reproduced at the stored precision. The comparisons provide no support for any of the three preregistered hypotheses. See the benchmark summary and paired statistics for tests and correction thresholds.

Reproduce from the corrected source with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmark.run --out results/e47-review-20261120` after installing the repository requirements.
