# Scalp-sphere Casimir test — written before any signal was analysed

Carrier: the scalp potential on the unit sphere. Group: SO(3). Operator: the Casimir L^2, eigenvalue l(l+1) on degree-l
spherical harmonics. Truncation L = 6, so the spectrum is 0, 2, 6, 12, 20, 30, 42 (49 real harmonics).

Data: BCI Competition IV dataset 1, calibration files, 59 electrodes, 100 Hz, 200 trials per subject (GitHub LFS mirror
robert1208/2026-HIP-Final-Project). Never used in any earlier run. Subjects a, b, f, g are human recordings; c, d, e are the
competition's artificially generated sets and are reported separately, outside the test.
Only file structure, channel names and class names were read before this was written.

Geometry: electrode positions from the file's own xpos/ypos, read as an azimuthal-equidistant projection about the vertex;
the single scale factor is fitted to the 49 electrodes that have standard 10-05 positions. No EEG enters this step.
Signal: common average reference, band-pass 8-32 Hz, epoch = cue + [0,4] s.
Harmonic coefficients: ridge least squares, a = (Y'Y + r I)^-1 Y' v, r = 0.01 * tr(Y'Y)/49.

State and selector (training folds only, no labels): p_l = share of total coefficient variance in degree l.
mu = sum p_l l(l+1); sigma^2 = sum p_l (l(l+1))^2 - mu^2.
l_lo = degree whose l(l+1) is nearest mu - sigma; l_hi = nearest mu + sigma. Kept sector = those degrees (one degree if they coincide).
Features: log-variance of each kept coefficient over the epoch -> shrinkage LDA. ROC-AUC, 5-fold stratified CV repeated 5 times (seeds 47-51).

PRIMARY TEST (one test, alpha = 0.05): the same pipeline is scored for all 28 degree sets {a,b}, a <= b.
T = mean over the four human subjects of the selected sector's AUC. Null: a selector with no information picks one of the 28 sets
uniformly for each subject; exact distribution by enumeration (28^4). p = P(T_null >= T).
Reported, not tested (four subjects cannot carry a second test): all 49 harmonics, 59-channel log-variance, CSP(8)+LDA,
tangent space + LR, the fixed pair {2,5}, the degrees selected, the kept fraction, and subjects c, d, e.
No value of the selected degrees or of the kept fraction counts as support or failure.
