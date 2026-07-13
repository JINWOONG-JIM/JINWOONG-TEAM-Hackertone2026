"""Step 2 — Accent estimation via nearest-centroid heuristic.

PRD section 4.2 explicitly warns against overconfident conclusions. Centroids
below are rough approximations from general phonetics references, not fitted
to a labeled accent corpus — this is a demo-grade heuristic, not a validated
classifier. Short or mostly-unvoiced clips return "insufficient_data" instead
of forcing a guess.

Feature vector per accent centroid: (mean_f0, f0_std, F1, F2, F3), all in Hz.
"""

import math

from .audio_features import FeatureSet

MIN_DURATION_SEC = 3.0
MIN_VOICED_RATIO = 0.2
HIGH_CONFIDENCE_DURATION = 6.0

CENTROIDS = {
    "American English": (150.0, 25.0, 550.0, 1600.0, 2600.0),
    "British English": (155.0, 22.0, 500.0, 1700.0, 2700.0),
    "Korean-accented English": (190.0, 15.0, 480.0, 1500.0, 2500.0),
    "Indian-accented English": (165.0, 20.0, 530.0, 1750.0, 2650.0),
    "Chinese-accented English": (175.0, 18.0, 470.0, 1550.0, 2550.0),
}

FEATURE_SCALE = (50.0, 15.0, 100.0, 200.0, 200.0)
TEMPERATURE = 1.4


def estimate_accent(features: FeatureSet) -> dict:
    if (
        features.duration_sec < MIN_DURATION_SEC
        or features.voiced_ratio < MIN_VOICED_RATIO
        or all(f == 0 for f in features.mean_formants)
    ):
        return {
            "status": "insufficient_data",
            "distribution": [],
            "confidence_level": "insufficient_data",
            "voiced_duration_sec": round(features.duration_sec * features.voiced_ratio, 2),
        }

    vector = (features.mean_f0, features.f0_std, *features.mean_formants)

    distances = {}
    for label, centroid in CENTROIDS.items():
        d_sq = sum(
            ((v - c) / s) ** 2
            for v, c, s in zip(vector, centroid, FEATURE_SCALE)
        )
        distances[label] = math.sqrt(d_sq)

    neg_scaled = {label: -d / TEMPERATURE for label, d in distances.items()}
    max_val = max(neg_scaled.values())
    exp_vals = {label: math.exp(v - max_val) for label, v in neg_scaled.items()}
    total = sum(exp_vals.values())
    probs = {label: v / total for label, v in exp_vals.items()}

    other_prob = max(0.0, 1.0 - sum(sorted(probs.values(), reverse=True)[:3]) * 0.6)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

    distribution = [
        {"label": label, "probability": round(p * 100, 1)} for label, p in ranked
    ]
    distribution.append({"label": "Other", "probability": round(other_prob * 100, 1)})
    total_pct = sum(d["probability"] for d in distribution)
    for d in distribution:
        d["probability"] = round(d["probability"] / total_pct * 100, 1)
    distribution.sort(key=lambda d: d["probability"], reverse=True)

    if features.duration_sec >= HIGH_CONFIDENCE_DURATION and features.voiced_ratio >= 0.5:
        confidence_level = "high"
    else:
        confidence_level = "medium"

    return {
        "status": "ok",
        "distribution": distribution,
        "confidence_level": confidence_level,
        "voiced_duration_sec": round(features.duration_sec * features.voiced_ratio, 2),
    }
