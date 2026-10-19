# Genesis benchmark: results

Run 2026-10-19 19:59 UTC · commit `18d7c98` · MOABB 1.7.2 · MNE 1.13.2 · scikit-learn 1.9.1 · Python 3.12.15

Score: within-session ROC-AUC, 5-fold cross-validation per subject and session (0.5 is chance).

## BNCI2014-001

BNCI2014-001 · 9 subjects

| decoder | ROC-AUC (mean ± sd over subjects) |
|---|---|
| Tangent space + LR | 0.863 ± 0.138 |
| CSP + LDA | 0.834 ± 0.155 |
| Genesis, fraction 0.50 | 0.629 ± 0.190 |
| Genesis, fraction 0.25 | 0.629 ± 0.189 |
| Genesis, no projection | 0.624 ± 0.188 |
| **Genesis** | 0.623 ± 0.194 |
| Genesis, variance projection | 0.614 ± 0.161 |
| Correlation dimension + LDA | 0.590 ± 0.112 |

**H1 · Genesis beats the standard decoders**: **not supported** (one-sided, Bonferroni α = 0.0250)

- versus CSP + LDA: ΔAUC = -0.210, p = 0.998, standardized effect = -1.72
- versus Tangent space + LR: ΔAUC = -0.239, p = 0.998, standardized effect = -1.93

**H2 · the 47/125 fraction beats the other fractions**: **not supported** (one-sided, Bonferroni α = 0.0167)

- versus Genesis, fraction 0.25: ΔAUC = -0.006, p = 0.791, standardized effect = -0.26
- versus Genesis, fraction 0.50: ΔAUC = -0.006, p = 0.8477, standardized effect = -0.36
- versus Genesis, no projection: ΔAUC = -0.001, p = 0.541, standardized effect = -0.04

**H3 · trials that pass the occupancy gate decode better**: **not supported**

- accuracy when the gate passes 0.603, when it fails 0.592; 48% of trials pass; 9 subjects; p = 0.2852

## Liu2024

50 acute stroke patients, left- versus right-hand motor imagery (Liu et al., Scientific Data 2024) · 50 subjects

| decoder | ROC-AUC (mean ± sd over subjects) |
|---|---|
| Tangent space + LR | 0.550 ± 0.178 |
| CSP + LDA | 0.539 ± 0.143 |
| Correlation dimension + LDA | 0.537 ± 0.137 |
| Genesis, fraction 0.25 | 0.528 ± 0.178 |
| **Genesis** | 0.522 ± 0.174 |
| Genesis, fraction 0.50 | 0.518 ± 0.177 |
| Genesis, no projection | 0.513 ± 0.164 |
| Genesis, variance projection | 0.507 ± 0.149 |

**H1 · Genesis beats the standard decoders**: **not supported** (one-sided, Bonferroni α = 0.0250)

- versus CSP + LDA: ΔAUC = -0.017, p = 0.7459, standardized effect = -0.09
- versus Tangent space + LR: ΔAUC = -0.028, p = 0.8062, standardized effect = -0.14

**H2 · the 47/125 fraction beats the other fractions**: **not supported** (one-sided, Bonferroni α = 0.0167)

- versus Genesis, fraction 0.25: ΔAUC = -0.006, p = 0.8683, standardized effect = -0.15
- versus Genesis, fraction 0.50: ΔAUC = +0.004, p = 0.1404, standardized effect = +0.11
- versus Genesis, no projection: ΔAUC = +0.008, p = 0.2014, standardized effect = +0.15

**H3 · trials that pass the occupancy gate decode better**: **not supported**

- accuracy when the gate passes 0.500, when it fails 0.537; 49% of trials pass; 50 subjects; p = 0.958

