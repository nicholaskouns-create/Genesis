"""Run the Genesis benchmark and write the results.

    python -m benchmark.run                                   # the pre-registered run (Liu2024, BNCI2014_001)
    python -m benchmark.run --datasets fake --subjects 3      # plumbing check, no download

Outputs in --out (default results/):
    results.csv   MOABB within-session ROC-AUC, one row per subject x session x decoder
    stats.csv     MOABB paired statistics per dataset (one-sided p for pipe1 > pipe2, standardized effect)
    gate.csv      per-subject decoding accuracy of trials that pass / fail the occupancy gate
    summary.json  the verdict on H1-H3 with every number behind it
    summary.md    the same in words
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mne
import moabb
import numpy as np
import pandas as pd
import sklearn
from moabb.analysis.meta_analysis import compute_dataset_statistics
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from genesis import BASELINES, GENESIS, OMEGA_C, SFREQ, build_pipelines

ALPHA = 0.05
PRIMARY = "Liu2024"
FRACTION_RIVALS = ("Genesis, fraction 0.25", "Genesis, fraction 0.50", "Genesis, no projection")
DESCRIPTIONS = {
    "Liu2024": "50 acute stroke patients, left- versus right-hand motor imagery (Liu et al., Scientific Data 2024)",
    "BNCI2014_001": "9 healthy participants, BCI Competition IV 2a, left- versus right-hand motor imagery",
    "FakeDataset": "synthetic noise (plumbing check only)",
}


def make_dataset(name: str):
    import moabb.datasets as mds
    if name == "fake":
        from moabb.datasets.fake import FakeDataset
        return FakeDataset(event_list=("left_hand", "right_hand"), n_sessions=1, n_runs=1, n_subjects=3,
                           channels=("FC3", "C3", "CP3", "Cz", "FC4", "C4", "CP4", "Pz"), sfreq=128, n_events=40, seed=47)
    return getattr(mds, name)()


def gate_analysis(paradigm, datasets, n_jobs=1) -> pd.DataFrame:
    """H3: within each session, 5-fold CV of the Genesis decoder; split test trials by the gate."""
    rows = []
    template = build_pipelines(n_jobs=n_jobs)[GENESIS]
    for ds in datasets:
        for subject in ds.subject_list:
            X, y, meta = paradigm.get_data(ds, subjects=[subject])
            for session in meta.session.unique():
                idx = np.flatnonzero(meta.session.values == session)
                Xs, ys = X[idx], y[idx]
                n_splits = min(5, np.bincount(pd.factorize(ys)[0]).min())
                if n_splits < 2:
                    continue
                correct, passed, occ = [], [], []
                for train, test in StratifiedKFold(n_splits, shuffle=True, random_state=47).split(Xs, ys):
                    pipe = clone(template).fit(Xs[train], ys[train])
                    F = pipe[:2].transform(Xs[test])                  # signature, standardized
                    proj = pipe.named_steps["coherenceprojection"]
                    o = proj.occupancy(F)
                    correct.extend(pipe.predict(Xs[test]) == ys[test])
                    passed.extend(o >= OMEGA_C)
                    occ.extend(o)
                correct, passed, occ = map(np.asarray, (correct, passed, occ))
                rows.append({
                    "dataset": ds.code, "subject": subject, "session": session, "n_trials": correct.size,
                    "pass_fraction": passed.mean(), "mean_occupancy": occ.mean(),
                    "acc_pass": correct[passed].mean() if passed.any() else np.nan,
                    "acc_fail": correct[~passed].mean() if (~passed).any() else np.nan,
                    "acc_all": correct.mean(),
                })
    return pd.DataFrame(rows)


def verdicts(results: pd.DataFrame, stats: pd.DataFrame, gate: pd.DataFrame) -> dict:
    out = {"datasets": {}}
    per_subject = results.groupby(["dataset", "pipeline", "subject"]).score.mean().reset_index()
    for code, g in per_subject.groupby("dataset"):
        ds = "FakeDataset" if str(code).startswith("FakeDataset") else str(code)
        table = g.groupby("pipeline").score.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
        st = stats[stats.dataset == code]

        def comp(a, b):
            row = st[(st.pipe1 == a) & (st.pipe2 == b)]
            wide = g.pivot(index="subject", columns="pipeline", values="score")
            diff = (wide[a] - wide[b]).mean() if a in wide and b in wide else np.nan
            return {"versus": b, "mean_auc_difference": float(diff),
                    "p_one_sided": float(row.p.iloc[0]) if len(row) else np.nan,
                    "standardized_effect": float(row.smd.iloc[0]) if len(row) else np.nan}

        h1 = [comp(GENESIS, b) for b in BASELINES if b in table.index]
        h2 = [comp(GENESIS, r) for r in FRACTION_RIVALS if r in table.index]
        bonf1, bonf2 = ALPHA / max(len(h1), 1), ALPHA / max(len(h2), 1)
        gd = gate[(gate.dataset == code)].groupby("subject")[["acc_pass", "acc_fail", "pass_fraction"]].mean().dropna()
        if len(gd) >= 5 and (gd.acc_pass - gd.acc_fail).abs().sum() > 0:
            w = wilcoxon(gd.acc_pass, gd.acc_fail, alternative="greater")
            h3 = {"subjects_with_both": int(len(gd)), "mean_acc_pass": float(gd.acc_pass.mean()),
                  "mean_acc_fail": float(gd.acc_fail.mean()), "mean_pass_fraction": float(gd.pass_fraction.mean()),
                  "p_one_sided": float(w.pvalue), "supported": bool(w.pvalue < ALPHA and gd.acc_pass.mean() > gd.acc_fail.mean())}
        else:
            h3 = {"subjects_with_both": int(len(gd)), "supported": None, "note": "too few subjects with trials on both sides of the gate"}
        out["datasets"][ds] = {
            "description": DESCRIPTIONS.get(ds, ds),
            "n_subjects": int(g.subject.nunique()),
            "auc": {p: {"mean": float(r["mean"]), "sd": float(r["std"]) if np.isfinite(r["std"]) else None} for p, r in table.iterrows()},
            "H1_genesis_beats_standard_decoders": {"comparisons": h1, "alpha_bonferroni": bonf1,
                                                   "supported": bool(h1) and all(c["p_one_sided"] < bonf1 and c["mean_auc_difference"] > 0 for c in h1)},
            "H2_47_over_125_beats_other_fractions": {"comparisons": h2, "alpha_bonferroni": bonf2,
                                                     "supported": bool(h2) and all(c["p_one_sided"] < bonf2 and c["mean_auc_difference"] > 0 for c in h2)},
            "H3_gate_predicts_reliability": h3,
        }
    return out


def write_summary(v: dict, meta: dict, path: Path):
    def yesno(s):
        return "not tested" if s is None else ("**supported**" if s else "**not supported**")
    lines = ["# Genesis benchmark: results", "",
             f"Run {meta['finished']} · commit `{meta['commit']}` · MOABB {meta['moabb']} · MNE {meta['mne']} · scikit-learn {meta['sklearn']} · Python {meta['python']}",
             "", "Score: within-session ROC-AUC, 5-fold cross-validation per subject and session (0.5 is chance).", ""]
    for ds, d in v["datasets"].items():
        lines += [f"## {ds}", "", d["description"] + f" · {d['n_subjects']} subjects", "", "| decoder | ROC-AUC (mean ± sd over subjects) |", "|---|---|"]
        for p, s in d["auc"].items():
            sd = f" ± {s['sd']:.3f}" if s["sd"] is not None else ""
            lines.append(f"| {'**' + p + '**' if p == GENESIS else p} | {s['mean']:.3f}{sd} |")
        for key, title in (("H1_genesis_beats_standard_decoders", "H1 · Genesis beats the standard decoders"),
                           ("H2_47_over_125_beats_other_fractions", "H2 · the 47/125 fraction beats the other fractions")):
            h = d[key]
            lines += ["", f"**{title}**: {yesno(h['supported'])} (one-sided, Bonferroni α = {h['alpha_bonferroni']:.4f})", ""]
            for c in h["comparisons"]:
                lines.append(f"- versus {c['versus']}: ΔAUC = {c['mean_auc_difference']:+.3f}, p = {c['p_one_sided']:.4g}, standardized effect = {c['standardized_effect']:+.2f}")
        h3 = d["H3_gate_predicts_reliability"]
        lines += ["", f"**H3 · trials that pass the occupancy gate decode better**: {yesno(h3.get('supported'))}", ""]
        if h3.get("supported") is not None:
            lines.append(f"- accuracy when the gate passes {h3['mean_acc_pass']:.3f}, when it fails {h3['mean_acc_fail']:.3f}; "
                         f"{100 * h3['mean_pass_fraction']:.0f}% of trials pass; {h3['subjects_with_both']} subjects; p = {h3['p_one_sided']:.4g}")
        else:
            lines.append(f"- {h3.get('note', '')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="Liu2024,BNCI2014_001")
    ap.add_argument("--subjects", type=int, default=None, help="use only the first N subjects of each dataset")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args(argv)
    mne.set_log_level("ERROR")
    moabb.set_log_level("error")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    datasets = [make_dataset(n.strip()) for n in args.datasets.split(",") if n.strip()]
    if args.subjects:
        for ds in datasets:
            ds.subject_list = ds.subject_list[: args.subjects]
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=SFREQ)
    evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True, random_state=47,
                                         hdf5_path=str(out / "moabb-cache"))
    results = evaluation.process(build_pipelines(n_jobs=args.n_jobs))
    results.to_csv(out / "results.csv", index=False)
    stats = compute_dataset_statistics(results)
    stats.to_csv(out / "stats.csv", index=False)
    gate = gate_analysis(paradigm, datasets, n_jobs=args.n_jobs)
    gate.to_csv(out / "gate.csv", index=False)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {"finished": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "seconds": round(time.time() - t0, 1), "commit": commit,
            "moabb": moabb.__version__, "mne": mne.__version__, "sklearn": sklearn.__version__, "python": platform.python_version(),
            "paradigm": "LeftRightImagery 8-32 Hz, resampled to 128 Hz", "evaluation": "WithinSessionEvaluation, 5-fold, random_state 47",
            "omega_c": OMEGA_C, "alpha": ALPHA, "primary_dataset": PRIMARY}
    v = verdicts(results, stats, gate)
    v["meta"] = meta
    (out / "summary.json").write_text(json.dumps(v, indent=1) + "\n")
    write_summary(v, meta, out / "summary.md")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
