# Moment-selector transport test — written before any outcome was computed

Rule under test (from E47_Moment_Selector_Theorem): on a carrier with self-adjoint operator C,
mu = tr(C)/d, sigma^2 = tr(C^2)/d - mu^2 under the normalized trace, K(l) = (l-(mu-sigma))(l-(mu+sigma)) = (l-mu)^2 - sigma^2.
On the spin-2 cube ker K is exact. On a carrier with continuous spectrum the kept sector is
    E_tau = span{ eigenvectors of C with |K(lambda)| <= tau * sigma^2 }.
PRIMARY tau = 1/2. The rule is cube-faithful for every tau < 3/4 (smallest unselected |K|/sigma^2 on the cube is 108/144).
Sensitivity: tau = 1/4 and tau = 0.74. If E_tau is empty: the two eigenvectors nearest mu-sigma and mu+sigma (count reported).
The kept fraction k/d is an OUTPUT. No value of it counts as support or failure.

Data: BCI Competition IV 2a (= BNCI2014-001), 9 healthy subjects, training sessions, left vs right hand,
from a public GitHub mirror (bregydoc/bcidatasetIV2a). Epoch = cue onset + [0,4] s, 22 EEG channels,
band-pass 8-32 Hz, resample 128 Hz (Genesis settings). Score: ROC-AUC, 5-fold stratified CV, seed 47.
All operators are fitted on training folds only.

Carrier A (Genesis carrier): standardized fractal intent signature (genesis.fractal, unchanged). C = OAS-shrunk covariance of training signatures.
   Features = coordinates in E_tau -> shrinkage LDA.
Carrier B (signal carrier): R^22. C = mean trace-normalized spatial covariance of training epochs.
   Features = log-variance of each epoch along each kept eigenvector -> shrinkage LDA.

Hypotheses (one-sided Wilcoxon signed-rank over 9 subjects, Bonferroni alpha = 0.05/4 = 0.0125):
 M1  A: moment selector > no projection
 M2  A: moment selector > random subspace of the same rank (mean of 20 draws)
 M3  B: moment selector > random subspace of the same rank (mean of 20 draws)
 M4  B: moment selector > CSP(8) + LDA
Reported, not tested: A vs top-k variance (same rank), A vs original Genesis (Fisher, 47/125), B vs top-k variance, tangent space + LR, kept fractions.
Not runnable here: the 50 stroke patients (Liu2024); the host is unreachable from this sandbox.
