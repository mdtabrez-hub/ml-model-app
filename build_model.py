"""
train_export_model.py
=====================
Trains and exports the deployment artefact for the crop recommendation system,
following the protocol described in the manuscript:

  * Sec. 2.2  min-max scaling fitted inside the pipeline (never on the full data)
  * Sec. 2.3  repeated stratified 10-fold CV (3 repeats), 80/20 held-out partition
  * Table 3   selected Random Forest configuration
  * Sec. 4.5  out-of-fold predictions over all 2,200 records
  * Sec. 4.6  ablation-derived 5-feature reduced model (humidity, N, rainfall, K, P)
  * Sec. 4.7  isotonic recalibration -> the variant served in production
  * Sec. 5    single serialised artefact consumed by the Streamlit / Flutter tiers

Usage
-----
    python train_export_model.py --data Crop_recommendation.csv --outdir models

Outputs
-------
    models/crop_model_v2.pkl           full 7-feature calibrated model
    models/crop_model_v2_reduced.pkl   5-feature fallback model (optional)
    models/training_report.json           the numbers to quote in the paper
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = "2.0"
RANDOM_STATE = 42

FULL_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
# Sec. 4.6 / Table 10: top five by permutation importance, 99.09% CV accuracy.
REDUCED_FEATURES = ["N", "P", "K", "humidity", "rainfall"]
TARGET = "label"

# Table 3: selected configuration for the deployed classifier.
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Sec. 4.5: rice and jute separate almost entirely on rainfall, so the system
# should defer inside the overlap band rather than assert a single answer.
AMBIGUITY_RULES = [
    {
        "classes": ["rice", "jute"],
        "feature": "rainfall",
        "low": 180.0,
        "high": 210.0,
        "note": (
            "Rice and jute have near-identical nutrient signatures in this dataset "
            "and separate mainly on rainfall. In this band both are presented as "
            "joint candidates rather than a single recommendation."
        ),
    }
]


def expected_calibration_error(y_true, proba, classes, n_bins: int = 10) -> float:
    """Standard equal-width ECE on the top-1 confidence."""
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def make_pipeline() -> Pipeline:
    """Min-max scaler + Random Forest, exactly as evaluated in the paper.

    The scaler lives *inside* the pipeline so that it is re-fitted on the
    training folds only (Sec. 2.3, leakage control) and is carried into the
    serialised artefact, so inference reproduces training-time preprocessing.
    """
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("rf", RandomForestClassifier(**RF_PARAMS)),
        ]
    )


def feature_stats(df: pd.DataFrame, features: list[str]) -> dict:
    """Table 4 statistics, used by the app for input validation and defaults."""
    stats = {}
    for f in features:
        s = df[f].astype(float)
        stats[f] = {
            "min": float(s.min()),
            "q1": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q3": float(s.quantile(0.75)),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "sd": float(s.std()),
        }
    return stats


def evaluate(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Repeated stratified CV + held-out + out-of-fold, as reported in Sec. 4."""
    print(f"\n--- Evaluating {label} model ({X.shape[1]} features) ---")

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)
    scores = cross_val_score(make_pipeline(), X, y, cv=rskf, scoring="accuracy", n_jobs=-1)
    mean, sd = scores.mean() * 100, scores.std(ddof=1) * 100
    half = 1.96 * sd / np.sqrt(len(scores))
    print(f"repeated 10x3 CV accuracy : {mean:.2f}% (SD {sd:.2f} pp, 95% CI "
          f"[{mean - half:.2f}, {mean + half:.2f}])")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    holdout_pipe = make_pipeline().fit(X_tr, y_tr)
    y_hat = holdout_pipe.predict(X_te)
    proba = holdout_pipe.predict_proba(X_te)
    n_err = int((y_hat != y_te).sum())
    print(f"held-out accuracy         : {accuracy_score(y_te, y_hat) * 100:.2f}% "
          f"({n_err} errors of {len(y_te)})")

    cal_holdout = CalibratedClassifierCV(
        estimator=make_pipeline(), method="isotonic", cv=5, ensemble=False
    ).fit(X_tr, y_tr)
    cal_proba = cal_holdout.predict_proba(X_te)
    ece_raw = expected_calibration_error(y_te, proba, holdout_pipe.classes_)
    ece_cal = expected_calibration_error(y_te, cal_proba, cal_holdout.classes_)
    print(f"ECE uncalibrated / isotonic: {ece_raw:.4f} / {ece_cal:.4f}")

    oof = cross_val_predict(
        make_pipeline(), X, y,
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    oof_acc = accuracy_score(y, oof) * 100
    print(f"out-of-fold accuracy      : {oof_acc:.2f}% ({int((oof != y).sum())} errors of {len(y)})")

    return {
        "cv_accuracy_mean": round(mean, 3),
        "cv_accuracy_sd": round(sd, 3),
        "cv_accuracy_ci95": [round(mean - half, 3), round(mean + half, 3)],
        "holdout_accuracy": round(accuracy_score(y_te, y_hat) * 100, 3),
        "holdout_errors": n_err,
        "holdout_precision_macro": round(precision_score(y_te, y_hat, average="macro") * 100, 3),
        "holdout_recall_macro": round(recall_score(y_te, y_hat, average="macro") * 100, 3),
        "holdout_f1_macro": round(f1_score(y_te, y_hat, average="macro") * 100, 3),
        "holdout_log_loss_uncalibrated": round(float(log_loss(y_te, proba, labels=list(holdout_pipe.classes_))), 4),
        "holdout_log_loss_calibrated": round(float(log_loss(y_te, cal_proba, labels=list(cal_holdout.classes_))), 4),
        "holdout_ece_uncalibrated": round(ece_raw, 4),
        "holdout_ece_calibrated": round(ece_cal, 4),
        "oof_accuracy": round(oof_acc, 3),
        "oof_errors": int((oof != y).sum()),
        "oof_classification_report": classification_report(y, oof, output_dict=True, zero_division=0),
    }


def build_artifact(df: pd.DataFrame, features: list[str], metrics: dict, variant: str) -> dict:
    """Fit the deployment models and package everything the app needs."""
    X, y = df[features], df[TARGET]

    # (a) uncalibrated pipeline -> used by TreeSHAP, which needs the raw forest
    pipeline = make_pipeline().fit(X, y)

    # (b) isotonic-calibrated pipeline -> the variant served in production (Sec. 4.7).
    #     ensemble=False refits a single base estimator on all the data instead of
    #     keeping one forest per fold, which keeps the artefact at the footprint
    #     reported in Sec. 5 rather than multiplying it by cv.
    calibrated = CalibratedClassifierCV(
        estimator=make_pipeline(), method="isotonic", cv=5, ensemble=False
    ).fit(X, y)

    # The forest refitted inside CalibratedClassifierCV is trained on the same rows
    # with the same seed as (a), so the two are identical. Rebinding the reference
    # makes joblib serialise the forest once instead of twice; the assertion below
    # refuses the optimisation if the two ever diverge.
    try:
        probe = X.iloc[:50]
        before = calibrated.predict_proba(probe)
        original = calibrated.calibrated_classifiers_[0].estimator
        calibrated.calibrated_classifiers_[0].estimator = pipeline
        if not np.allclose(before, calibrated.predict_proba(probe), atol=1e-12):
            calibrated.calibrated_classifiers_[0].estimator = original
            print("note: base estimator not shared (predictions differed); artefact will be larger")
    except (AttributeError, IndexError):
        print("note: could not share base estimator on this scikit-learn version")

    return {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "python": platform.python_version(),
        },
        "feature_names": features,
        "feature_stats": feature_stats(df, features),
        "class_names": list(pipeline.classes_),          # crop names, not integers
        "n_training_records": int(len(df)),
        "model_params": RF_PARAMS,
        "pipeline": pipeline,                             # uncalibrated: label + SHAP
        "calibrated": calibrated,                         # served: probabilities
        "ambiguity_rules": AMBIGUITY_RULES if variant == "full" else [],
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="Crop_recommendation.csv", help="path to the dataset CSV")
    ap.add_argument("--outdir", default="models", help="directory for the artefacts")
    ap.add_argument("--skip-reduced", action="store_true", help="do not export the 5-feature model")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    missing = [c for c in FULL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"dataset is missing required columns: {missing}")

    # Labels stay as crop-name strings all the way through. Nothing downstream
    # ever has to map an integer back to a crop, which is where the previous
    # version went wrong.
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    print(f"records: {len(df)} | classes: {df[TARGET].nunique()} | "
          f"missing values: {int(df.isna().sum().sum())} | "
          f"duplicate rows: {int(df.duplicated().sum())} | "
          f"duplicate feature vectors: {int(df.duplicated(subset=FULL_FEATURES).sum())}")

    report = {}

    metrics_full = evaluate(df[FULL_FEATURES], df[TARGET], "full")
    report["full"] = metrics_full
    art_full = build_artifact(df, FULL_FEATURES, metrics_full, "full")
    path_full = outdir / "crop_model_v2.pkl"
    joblib.dump(art_full, path_full, compress=0)
    print(f"\nwrote {path_full} ({path_full.stat().st_size / 1e6:.2f} MB)")

    if not args.skip_reduced:
        metrics_red = evaluate(df[REDUCED_FEATURES], df[TARGET], "reduced")
        report["reduced"] = metrics_red
        art_red = build_artifact(df, REDUCED_FEATURES, metrics_red, "reduced")
        path_red = outdir / "crop_model_v2_reduced.pkl"
        joblib.dump(art_red, path_red, compress=0)
        print(f"wrote {path_red} ({path_red.stat().st_size / 1e6:.2f} MB)")

    (outdir / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {outdir / 'training_report.json'}")

    # --- reload and smoke-test exactly as the app will ------------------------
    art = joblib.load(path_full)
    probe = pd.DataFrame(
        [[79.9, 47.6, 39.9, 23.7, 82.3, 6.4, 236.0],   # rice centroid, Sec. 4.5
         [78.4, 46.9, 40.0, 24.9, 79.6, 6.7, 175.0]],  # jute centroid, Sec. 4.5
        columns=FULL_FEATURES,
    )
    proba = art["calibrated"].predict_proba(probe)
    for row, p in zip(("rice centroid", "jute centroid"), proba):
        top = np.argsort(p)[::-1][:2]
        print(f"check {row:14} -> " + ", ".join(
            f"{art['class_names'][i]} {p[i] * 100:.1f}%" for i in top))


if __name__ == "__main__":
    main()
