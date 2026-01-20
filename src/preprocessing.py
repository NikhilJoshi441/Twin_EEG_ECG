"""Preprocessing helpers using NumPy / SciPy.

This module provides ECG and EEG preprocessing utilities used across the
project: bandpass, notch, baseline wander removal, moving-average denoising
and optional wavelet denoising (if `pywt` is installed).

Functions are written to accept lists or 1D numpy arrays and return numpy
arrays (for easier downstream processing).
"""
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy import signal


def bandpass_filter(x: ArrayLike, low: float = 0.5, high: float = 40.0, fs: int = 250, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        x: 1D signal (list or array)
        low: low cutoff (Hz)
        high: high cutoff (Hz)
        fs: sampling frequency (Hz)
        order: filter order

    Returns:
        filtered signal as numpy array
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    nyq = 0.5 * fs
    low_n = max(1e-6, low / nyq)
    high_n = min(1 - 1e-6, high / nyq)
    if low_n >= high_n:
        # invalid band, return copy
        return arr.copy()
    sos = signal.butter(order, [low_n, high_n], btype="band", output="sos")
    return signal.sosfiltfilt(sos, arr)


def notch_filter(x: ArrayLike, fs: int = 250, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """IIR notch (bandstop) filter around `freq` in Hz.

    Uses `scipy.signal.iirnotch` and `filtfilt` for zero-phase.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    w0 = freq / (0.5 * fs)
    if w0 <= 0 or w0 >= 1:
        return arr.copy()
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, arr)


def detrend(x: ArrayLike) -> np.ndarray:
    """Remove linear trend from signal using `scipy.signal.detrend`."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    return signal.detrend(arr)


def baseline_wander_removal(x: ArrayLike, fs: int = 250, cutoff: float = 0.5, order: int = 2) -> np.ndarray:
    """Remove baseline wander by subtracting a lowpass estimate.

    Implementation: lowpass filter the signal at `cutoff` Hz and subtract the
    result from the original signal to remove very-low-frequency baseline.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    nyq = 0.5 * fs
    wn = min(max(cutoff / nyq, 1e-6), 1 - 1e-6)
    sos = signal.butter(order, wn, btype="low", output="sos")
    baseline = signal.sosfiltfilt(sos, arr)
    return arr - baseline


def moving_average(x: ArrayLike, window_ms: float = 50.0, fs: int = 250) -> np.ndarray:
    """Simple moving-average denoising.

    Args:
        window_ms: window length in milliseconds
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    win_samples = max(1, int(round(window_ms * fs / 1000.0)))
    # use convolution with a normalized window
    window = np.ones(win_samples) / float(win_samples)
    return np.convolve(arr, window, mode="same")


def wavelet_denoise(x: ArrayLike, wavelet: str = "db4", level: Optional[int] = None) -> np.ndarray:
    """Wavelet denoising (soft-threshold). Requires `pywt`.

    Falls back to moving-average if `pywt` is not installed.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return arr
    try:
        import importlib
        pywt = importlib.import_module('pywt')

        # choose level if not provided
        maxlev = pywt.dwt_max_level(arr.size, pywt.Wavelet(wavelet).dec_len)
        lev = level if (level is not None and level > 0) else (maxlev // 2 or 1)
        coeffs = pywt.wavedec(arr, wavelet, level=lev)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(arr.size))
        denoised_coeffs = [pywt.threshold(c, value=uthresh, mode="soft") if i > 0 else c for i, c in enumerate(coeffs)]
        return pywt.waverec(denoised_coeffs, wavelet)[: arr.size]
    except Exception:
        # pywt not installed or failed — fallback
        return moving_average(arr, window_ms=50.0)


# Compatibility wrappers
def simple_bandpass(x: ArrayLike, low: float = 0.5, high: float = 40.0, fs: int = 250) -> np.ndarray:
    """Compatibility wrapper -> `bandpass_filter` returning numpy array."""
    return bandpass_filter(x, low=low, high=high, fs=fs)


def simple_notch(x: ArrayLike, fs: int = 250, notch_freq: float = 50.0) -> np.ndarray:
    """Compatibility wrapper -> `notch_filter`."""
    return notch_filter(x, fs=fs, freq=notch_freq)

