"""Synthetic ECG and EEG signal generators (very small, dependency-free)"""
import math
import random
from typing import List

def generate_ecg(duration_s: float = 1.0, fs: int = 250) -> List[float]:
    """Generate a synthetic ECG-like waveform for duration seconds sampled at fs.

    This is a toy signal (sum of sinusoids and sharp peaks) for pipeline testing.
    """
    n = int(duration_s * fs)
    ecg = []
    for i in range(n):
        t = i / fs
        # baseline wander (low freq)
        baseline = 0.1 * math.sin(2 * math.pi * 0.33 * t)
        # QRS-like sharp peak approximated by a narrow gaussian-like pulse
        beat_period = 60.0 / 60.0  # 60 bpm baseline
        # place beats every beat_period seconds
        phase = (t % beat_period) / beat_period
        qrs = 0.0
        if phase < 0.05:
            # narrow pulse
            qrs = 1.0 * math.exp(-((phase - 0.025) ** 2) / (2 * (0.01 ** 2)))
        # low amplitude P and T waves
        p_t = 0.02 * math.sin(2 * math.pi * 5 * t) + 0.03 * math.sin(2 * math.pi * 1.2 * t)
        noise = 0.02 * (random.random() - 0.5)
        ecg.append(baseline + qrs + p_t + noise)
    return ecg


def generate_eeg(duration_s: float = 1.0, fs: int = 128) -> List[float]:
    """Generate a synthetic EEG-like signal as mixture of band-limited sinusoids.

    No external dependencies; purely illustrative.
    """
    n = int(duration_s * fs)
    eeg = []
    for i in range(n):
        t = i / fs
        # alpha (8-12 Hz), beta (13-30 Hz), delta slow
        alpha = 20.0 * math.sin(2 * math.pi * 10 * t) * 0.5
        beta = 5.0 * math.sin(2 * math.pi * 20 * t) * 0.3
        delta = 10.0 * math.sin(2 * math.pi * 1.5 * t) * 0.2
        noise = 1.0 * (random.random() - 0.5)
        eeg.append(alpha + beta + delta + noise)
    return eeg


if __name__ == "__main__":
    # quick smoke
    ecg = generate_ecg(1.0, 250)
    eeg = generate_eeg(1.0, 128)
    print(f"Generated ECG samples: {len(ecg)}, EEG samples: {len(eeg)}")
