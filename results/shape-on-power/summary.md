# Does fractal shape add anything on top of power?

Run 2026-10-19 22:11 UTC · commit `125c5a6` · MOABB 1.7.2 · MNE 1.13.2 · scikit-learn 1.9.1 · Python 3.12.15

Each pair differs only by the fractal intent signature: same channels, bands, standardization and shrinkage LDA. Score: within-session ROC-AUC, 5-fold cross-validation per subject and session (0.5 is chance).

## BNCI2014-001

9 healthy participants, BCI Competition IV 2a, left- versus right-hand motor imagery · 9 subjects

| decoder | ROC-AUC (mean ± sd over subjects) |
|---|---|
| Tangent space + LR | 0.863 ± 0.138 |
| Tangent space | 0.855 ± 0.130 |
| Tangent space + fractal shape | 0.825 ± 0.159 |
| Band power | 0.752 ± 0.182 |
| Band power + fractal shape | 0.664 ± 0.198 |
| Genesis | 0.623 ± 0.194 |

**H4 · fractal shape adds to band power**: **not supported** (one-sided, Bonferroni α = 0.025)

- Band power + fractal shape minus Band power: ΔAUC = -0.088 (95% CI -0.121 to -0.057), p = 0.998, standardized effect = -1.68, better in 0 of 9 subjects

**H5 · fractal shape adds to tangent space**: **not supported** (one-sided, Bonferroni α = 0.025)

- Tangent space + fractal shape minus Tangent space: ΔAUC = -0.030 (95% CI -0.051 to -0.011), p = 0.9863, standardized effect = -0.90, better in 3 of 9 subjects

## Liu2024

50 acute stroke patients, left- versus right-hand motor imagery (Liu et al., Scientific Data 2024) · 50 subjects

| decoder | ROC-AUC (mean ± sd over subjects) |
|---|---|
| Tangent space | 0.564 ± 0.166 |
| Tangent space + fractal shape | 0.553 ± 0.180 |
| Band power | 0.552 ± 0.163 |
| Tangent space + LR | 0.550 ± 0.178 |
| Genesis | 0.522 ± 0.174 |
| Band power + fractal shape | 0.517 ± 0.164 |

**H4 · fractal shape adds to band power**: **not supported** (one-sided, Bonferroni α = 0.025)

- Band power + fractal shape minus Band power: ΔAUC = -0.036 (95% CI -0.075 to +0.004), p = 0.9265, standardized effect = -0.25, better in 19 of 50 subjects

**H5 · fractal shape adds to tangent space**: **not supported** (one-sided, Bonferroni α = 0.025)

- Tangent space + fractal shape minus Tangent space: ΔAUC = -0.012 (95% CI -0.038 to +0.014), p = 0.6437, standardized effect = -0.12, better in 25 of 50 subjects

