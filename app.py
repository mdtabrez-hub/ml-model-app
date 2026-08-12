"""
Crop Recommendation System — Streamlit application
==================================================
Consumes crop_model_v2.pkl, the artefact described in the manuscript:

    min-max scaler fitted on training folds only   (Sec. 2.2, Step 7)
    Random Forest, 100 trees, max_depth 10         (Table 3)
    isotonic-recalibrated probabilities            (Sec. 4.7)
    exact TreeSHAP per-recommendation explanation  (Sec. 4.6)
    rice/jute deferral inside the rainfall overlap (Sec. 4.5)
    input rejection outside the trained ranges     (Sec. 5, Tier 2)

Class labels inside the artefact are crop-name strings, so this file contains no
integer-to-name mapping. That mapping is what made the previous version display
the wrong crop for every prediction.

Repository layout:  app.py, crop_model_v2.pkl, requirements.txt
Run locally with:   streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:  # optional: falls back to global importances if unavailable
    import shap

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

MODEL_FILENAME = "crop_model_v2.pkl"

# Display metadata only; the artefact is the authority on features and ranges.
FEATURE_META = {
    "N": {"label": "Nitrogen (N)", "unit": "kg ha⁻¹", "step": 1.0, "fmt": "%.0f"},
    "P": {"label": "Phosphorus (P)", "unit": "kg ha⁻¹", "step": 1.0, "fmt": "%.0f"},
    "K": {"label": "Potassium (K)", "unit": "kg ha⁻¹", "step": 1.0, "fmt": "%.0f"},
    "temperature": {"label": "Temperature", "unit": "°C", "step": 0.1, "fmt": "%.2f"},
    "humidity": {"label": "Relative humidity", "unit": "%", "step": 0.1, "fmt": "%.2f"},
    "ph": {"label": "Soil pH", "unit": "", "step": 0.01, "fmt": "%.2f"},
    "rainfall": {"label": "Rainfall", "unit": "mm", "step": 1.0, "fmt": "%.2f"},
}

CROP_NOTES = {
    "apple": "Temperate; needs winter chill and well-drained loam.",
    "banana": "Warm and humid throughout; heavy potassium feeder.",
    "blackgram": "Short-duration pulse; warm conditions and fertile loamy soil.",
    "chickpea": "Cool, dry finish; sensitive to waterlogging.",
    "coconut": "Coastal humid tropics; sandy soils with steady moisture.",
    "coffee": "Shaded highlands; well-distributed rainfall, acidic soil.",
    "cotton": "Long warm season; deep soils, high nitrogen demand.",
    "grapes": "Dry ripening period; very high P and K requirement.",
    "jute": "Warm and humid with moderate rainfall; alluvial soils.",
    "kidneybeans": "Moderate temperature; avoid waterlogged fields.",
    "lentil": "Cool season pulse; low input, drought tolerant.",
    "maize": "Wide adaptability; responsive to nitrogen.",
    "mango": "Dry flowering period followed by warm humid growth.",
    "mothbeans": "Arid and semi-arid; highly drought tolerant.",
    "mungbean": "Short duration; warm season, low water need.",
    "muskmelon": "Hot dry weather; sandy loam with irrigation.",
    "orange": "Subtropical; well-drained soil, moderate rainfall.",
    "papaya": "Continuous warmth; frost and waterlogging intolerant.",
    "pigeonpeas": "Deep-rooted, drought hardy; long duration.",
    "pomegranate": "Semi-arid; tolerates poor soils, dislikes humidity at ripening.",
    "rice": "High rainfall or assured irrigation; puddled fields.",
    "watermelon": "Hot dry season; sandy loam and steady irrigation.",
}


# --------------------------------------------------------------------------- #
# Artefact loading                                                            #
# --------------------------------------------------------------------------- #


def find_model() -> Path | None:
    """Locate the artefact regardless of the host's working directory.

    Streamlit Community Cloud does not guarantee that the working directory is
    the one holding this file, so a bare relative path can fail on the server
    while working on a laptop.
    """
    here = Path(__file__).resolve().parent
    for directory in (here, here / "models", Path.cwd(), Path.cwd() / "models"):
        candidate = directory / MODEL_FILENAME
        if candidate.is_file():
            return candidate
    return None


@st.cache_resource(show_spinner="Loading model…")
def load_artifact(path_str: str) -> dict[str, Any]:
    """Unpickle the artefact and attach a TreeSHAP explainer for the forest."""
    artifact = joblib.load(Path(path_str))
    artifact["_explainer"] = None
    if SHAP_AVAILABLE:
        try:
            artifact["_explainer"] = shap.TreeExplainer(artifact["pipeline"][-1])
        except Exception:
            artifact["_explainer"] = None
    return artifact


def version_note(artifact: dict[str, Any]) -> str | None:
    """Pickled estimators are version-sensitive; surface a mismatch plainly."""
    try:
        import sklearn

        trained = artifact["library_versions"]["scikit-learn"]
        if trained != sklearn.__version__:
            return (
                f"This model was built with scikit-learn {trained}, but the "
                f"environment has {sklearn.__version__}. Pin "
                f"`scikit-learn=={trained}` in requirements.txt — predictions are "
                "not guaranteed to be identical across versions."
            )
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #


def validate(values: dict[str, float], stats: dict[str, dict]) -> list[str]:
    """Reject inputs outside the trained ranges (Table 4; Sec. 5, Tier 2)."""
    problems = []
    for name, value in values.items():
        info = stats[name]
        label = FEATURE_META.get(name, {}).get("label", name)
        unit = FEATURE_META.get(name, {}).get("unit", "")
        if value is None or not np.isfinite(value):
            problems.append(f"{label}: enter a number.")
        elif value < info["min"] or value > info["max"]:
            problems.append(
                f"{label} = {value:g} is outside the trained range "
                f"{info['min']:g}–{info['max']:g} {unit}."
            )
    return problems


def top_contributions(artifact: dict[str, Any], frame: pd.DataFrame,
                      class_index: int, k: int = 3):
    """Top-k exact TreeSHAP contributions for this decision (Sec. 4.6).

    Attribution runs on the forest inside the pipeline, so the inputs must be
    pushed through the fitted scaler first.
    """
    explainer = artifact.get("_explainer")
    if explainer is None:
        return None
    features = artifact["feature_names"]
    try:
        scaled = artifact["pipeline"][:-1].transform(frame[features])
        raw = explainer.shap_values(scaled, check_additivity=False)
        if isinstance(raw, list):  # older shap: one array per class
            contrib = np.asarray(raw[class_index])[0]
        else:
            arr = np.asarray(raw)
            contrib = arr[0, :, class_index] if arr.ndim == 3 else arr[0]
    except Exception:
        return None
    order = np.argsort(np.abs(contrib))[::-1][:k]
    return [
        {
            "feature": FEATURE_META.get(features[i], {}).get("label", features[i]),
            "value": float(frame.iloc[0][features[i]]),
            "contribution": float(contrib[i]),
        }
        for i in order
    ]


def triggered_rule(artifact: dict[str, Any], values: dict[str, float],
                   classes: np.ndarray, crop: str):
    """The deferral rule this input triggers, if any (Sec. 4.5)."""
    for rule in artifact.get("ambiguity_rules", []):
        feature = rule["feature"]
        if crop in rule["classes"] and feature in values:
            if rule["low"] <= values[feature] <= rule["high"]:
                if all(name in classes for name in rule["classes"]):
                    return rule
    return None


# --------------------------------------------------------------------------- #
# Interface                                                                   #
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="wide")

st.title("🌱 Crop Recommendation System")
st.caption(
    "Enter soil and climate measurements to get a crop recommendation, a calibrated "
    "confidence and the reasons behind it. Advisory only — check against local "
    "agronomic knowledge."
)

model_path = find_model()
if model_path is None:
    here = Path(__file__).resolve().parent
    st.error(
        f"**`{MODEL_FILENAME}` not found.** Commit it to the repository root, next "
        "to `app.py`, then reboot the app."
    )
    try:
        visible = sorted(p.name for p in here.iterdir())
        st.caption(f"Files the app can see in `{here}`: {', '.join(visible) or '(none)'}")
        st.caption("File names are case-sensitive on the server.")
    except OSError:
        pass
    st.stop()

artifact = load_artifact(str(model_path))
features = artifact["feature_names"]
stats = artifact["feature_stats"]
classes = np.asarray(artifact["class_names"])
metrics = artifact.get("metrics", {})

note = version_note(artifact)
if note:
    st.warning(note, icon="⚠️")

with st.sidebar:
    st.header("Model")
    st.metric("Cross-validated accuracy", f"{metrics.get('cv_accuracy_mean', float('nan')):.2f}%")
    ci = metrics.get("cv_accuracy_ci95")
    if ci:
        st.caption(
            f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}] · SD {metrics.get('cv_accuracy_sd', 0):.2f} pp"
        )
    st.caption("Repeated stratified 10-fold cross-validation, three repeats (30 folds)")
    st.caption(
        f"Out-of-fold: {metrics.get('oof_accuracy', float('nan')):.2f}% over "
        f"{artifact.get('n_training_records', '—')} records "
        f"({metrics.get('oof_errors', '—')} errors)"
    )
    st.caption(
        f"Calibration ECE {metrics.get('holdout_ece_calibrated', float('nan')):.4f} "
        f"(before recalibration {metrics.get('holdout_ece_uncalibrated', float('nan')):.4f})"
    )
    st.divider()
    params = artifact.get("model_params", {})
    st.caption(
        f"Random Forest · {params.get('n_estimators', '?')} trees · "
        f"max_depth {params.get('max_depth', '?')} · isotonic-calibrated"
    )
    st.caption(f"Built {str(artifact.get('created_utc', ''))[:10]} · "
               f"scikit-learn {artifact['library_versions']['scikit-learn']}")
    st.divider()
    show_why = st.checkbox("Explain the recommendation", value=True)
    n_alternatives = st.slider("Alternatives to show", 3, 10, 5)
    if not SHAP_AVAILABLE:
        st.info("`shap` is not installed, so explanations use global importances.")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Field measurements")
    st.caption("Each range below is the range covered by the training data.")
    values: dict[str, float] = {}
    for name in features:
        meta = FEATURE_META.get(name, {"label": name, "unit": "", "step": 0.1, "fmt": "%.2f"})
        info = stats[name]
        title = f"{meta['label']} ({meta['unit']})" if meta["unit"] else meta["label"]
        values[name] = st.number_input(
            title,
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["median"]),
            step=float(meta["step"]),
            format=meta["fmt"],
            help=(f"Trained range {info['min']:g}–{info['max']:g}; "
                  f"typical values {info['q1']:g}–{info['q3']:g}"),
        )

    problems = validate(values, stats)
    for problem in problems:
        st.error(problem)

    go = st.button("Recommend crop", type="primary", disabled=bool(problems))

with right:
    st.subheader("Recommendation")

    if not go:
        st.info("Enter the measurements, then select **Recommend crop**.")
    else:
        frame = pd.DataFrame([[values[f] for f in features]], columns=features)

        started = time.perf_counter()
        probabilities = artifact["calibrated"].predict_proba(frame)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        best = int(np.argmax(probabilities))
        crop = str(classes[best])
        rule = triggered_rule(artifact, values, classes, crop)

        if rule is not None:
            pair = rule["classes"]
            st.warning("### " + " or ".join(name.title() for name in pair))
            columns = st.columns(len(pair))
            for column, name in zip(columns, pair):
                index = int(np.where(classes == name)[0][0])
                column.metric(name.title(), f"{probabilities[index] * 100:.1f}%")
            st.caption(rule["note"])
        else:
            st.success(f"### {crop.title()}")
            # Isotonic regression can saturate at exactly 1.0; never claim certainty.
            st.metric("Calibrated confidence", f"{min(float(probabilities[best]), 0.999) * 100:.1f}%")
            st.caption(CROP_NOTES.get(crop, ""))

        if show_why:
            st.markdown("**Why this crop**")
            contributions = top_contributions(artifact, frame, best)
            if contributions:
                for item in contributions:
                    positive = item["contribution"] >= 0
                    st.markdown(
                        f"{'▲' if positive else '▼'} **{item['feature']}** = "
                        f"{item['value']:g} — "
                        f"{'supports' if positive else 'argues against'} this crop "
                        f"({item['contribution']:+.4f})"
                    )
                st.caption("Exact TreeSHAP contributions for this specific prediction.")
            else:
                importances = pd.DataFrame(
                    {"Importance": artifact["pipeline"][-1].feature_importances_},
                    index=[FEATURE_META.get(f, {}).get("label", f) for f in features],
                ).sort_values("Importance", ascending=False)
                st.bar_chart(importances)
                st.caption("Global feature importances — not specific to this prediction.")

        st.markdown("**Other candidates**")
        order = np.argsort(probabilities)[::-1][:n_alternatives]
        st.dataframe(
            pd.DataFrame(
                {
                    "Crop": [str(classes[i]).title() for i in order],
                    "Probability": [float(probabilities[i]) for i in order],
                }
            ),
            hide_index=True,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="%.3f"
                )
            },
        )
        st.caption(f"Answered in {elapsed_ms:.1f} ms.")

st.divider()
st.caption(
    "The model was validated on a single public benchmark and has not been tested "
    "against independent field observations. Seven variables omit soil texture, "
    "organic carbon, irrigation availability, rotation history, pest pressure and "
    "economics, so the output is agronomic advice rather than a farm-management "
    "decision."
)
