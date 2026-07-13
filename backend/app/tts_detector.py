"""Step 1 — Human vs Synthetic heuristic classifier.

This is a rule-based scorer over signal-processing features (PRD section 4.1),
not a trained model. Thresholds are reasonable approximations from speech
science literature, not calibrated against a labeled dataset — treat this as
a first-pass screening signal, not a forensic verdict.
"""

from .audio_features import FeatureSet

JITTER_LOW = 0.0025
SHIMMER_LOW = 0.02
SILENCE_STD_LOW = 0.0015
HF_STD_LOW = 0.01

WEIGHTS = {
    "jitter": 0.30,
    "shimmer": 0.25,
    "silence": 0.25,
    "hf_artifact": 0.20,
}


def detect_synthetic(features: FeatureSet) -> dict:
    reasons = []
    score = 0.0

    if features.voiced_frame_count < 3:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "reasons": ["유효 음성 구간이 너무 짧아 판별할 수 없음"],
            "metrics": _metrics(features),
        }

    if features.jitter < JITTER_LOW:
        score += WEIGHTS["jitter"]
        reasons.append("pitch jitter가 비정상적으로 낮음 (자연스러운 성대 떨림 부족)")

    if features.shimmer < SHIMMER_LOW:
        score += WEIGHTS["shimmer"]
        reasons.append("amplitude shimmer가 비정상적으로 낮음")

    if features.silence_rms_std < SILENCE_STD_LOW:
        score += WEIGHTS["silence"]
        reasons.append("배경 노이즈가 비정상적으로 규칙적/깨끗함")

    if features.hf_energy_std < HF_STD_LOW:
        score += WEIGHTS["hf_artifact"]
        reasons.append("고주파 대역 에너지 패턴이 인위적으로 균일함")

    is_synthetic = score >= 0.5
    confidence = round((score if is_synthetic else 1 - score) * 100, 1)

    if not reasons:
        reasons.append("자연스러운 성대/배경 노이즈 변동이 관측됨")

    return {
        "label": "synthetic" if is_synthetic else "human",
        "confidence": confidence,
        "reasons": reasons,
        "metrics": _metrics(features),
    }


def _metrics(features: FeatureSet) -> dict:
    return {
        "jitter": round(features.jitter, 5),
        "shimmer": round(features.shimmer, 5),
        "silence_rms_std": round(features.silence_rms_std, 5),
        "hf_energy_std": round(features.hf_energy_std, 5),
        "voiced_ratio": round(features.voiced_ratio, 3),
    }
