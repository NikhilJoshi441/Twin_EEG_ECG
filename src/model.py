"""Placeholder anomaly detector / digital twin stub.

This module contains a simple rule-based detector to simulate outputs.
Replace with a ML model (PyTorch/TensorFlow) later.
"""
from typing import Dict, List


def detect_anomalies(ecg_features: Dict[str, float], eeg_bandpower: float) -> Dict[str, float]:
    """Return a small dict with anomaly scores between 0 and 1.

    Simple heuristics:
    - low HR (avg_hr_bpm < 50) -> bradycardia risk
    - high HR (avg_hr_bpm > 100) -> tachycardia risk
    - low eeg_bandpower -> low-alpha indicator (placeholder)
    """
    hr = ecg_features.get("avg_hr_bpm", 0.0)
    score = 0.0
    if hr > 100:
        score = min(1.0, (hr - 100) / 40.0)
    elif hr < 50 and hr > 0:
        score = min(1.0, (50 - hr) / 50.0)

    eeg_alert = 1.0 if eeg_bandpower < 0.1 else 0.0

    return {"cardiac_anomaly_score": round(score, 3), "eeg_low_bandpower": eeg_alert}
