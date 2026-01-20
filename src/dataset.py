"""Synthetic dataset generator and PyTorch Dataset for training the hybrid model."""
from typing import Tuple
import random
import numpy as np


def synthesize_pair(duration_s: float = 5.0, ecg_fs: int = 250, eeg_fs: int = 128, anomaly: bool = False):
    """Return (ecg, eeg, label) where ecg/eeg are numpy arrays.

    If anomaly is True, inject a simple cardiac anomaly (tachycardia) or EEG seizure burst.
    """
    from .simulator import generate_ecg, generate_eeg

    ecg = np.array(generate_ecg(duration_s, ecg_fs), dtype=float)
    eeg = np.array(generate_eeg(duration_s, eeg_fs), dtype=float)

    label = 0
    if anomaly:
        label = 1
        # inject tachycardia: compress beat period by 40% for half the recording
        n = len(ecg)
        half = n // 2
        ecg[:half] *= 1.2
        # inject EEG burst (add high-amplitude beta)
        t = np.arange(len(eeg)) / eeg_fs
        eeg += 5.0 * np.sin(2 * np.pi * 20.0 * t) * (np.exp(-((t - duration_s/4) ** 2) / (0.1 ** 2)))

    return ecg.astype(np.float32), eeg.astype(np.float32), int(label)


if __name__ == "__main__":
    ecg, eeg, lbl = synthesize_pair(5.0, anomaly=True)
    print(ecg.shape, eeg.shape, lbl)
