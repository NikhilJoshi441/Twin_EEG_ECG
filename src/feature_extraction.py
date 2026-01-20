"""Feature extraction using NumPy/SciPy.

Implements Pan–Tompkins-like QRS detection and Welch PSD for EEG bandpower.
"""
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy import signal
from scipy.integrate import trapezoid
from scipy import stats


def pan_tompkins_detector(ecg: ArrayLike, fs: int = 250) -> List[int]:
    """Pan–Tompkins inspired detector returning R-peak indices.

    Steps: bandpass (5-15Hz), derivative, squaring, moving window integration, peak search.
    This is a simplification suitable for well-behaved signals and testing.
    """
    x = np.asarray(ecg, dtype=float)
    if x.size == 0:
        return []
    # bandpass 5-15 Hz
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    sos = signal.butter(1, [low, high], btype="band", output="sos")
    y = signal.sosfiltfilt(sos, x)

    # derivative (simple difference)
    dydt = np.ediff1d(y, to_end=0)
    # squaring
    squared = dydt ** 2
    # moving window integration (window ~150ms)
    win_len = max(1, int(0.150 * fs))
    window = np.ones(win_len) / win_len
    integrated = np.convolve(squared, window, mode="same")

    # find peaks on integrated signal
    distance = int(0.2 * fs)  # minimum 200 ms between peaks
    peaks, _ = signal.find_peaks(integrated, distance=distance)

    # refine peaks by finding local maxima in original filtered signal near integrated peaks
    r_peaks = []
    search_radius = int(0.05 * fs)
    for p in peaks:
        start = max(0, p - search_radius)
        end = min(len(y) - 1, p + search_radius)
        local_max = start + int(np.argmax(y[start:end + 1]))
        r_peaks.append(int(local_max))

    # unique and sorted
    r_peaks = sorted(list(dict.fromkeys(r_peaks)))
    return r_peaks


def compute_rr_intervals(peaks: List[int], fs: int = 250) -> List[float]:
    return [float(peaks[i] - peaks[i - 1]) / fs for i in range(1, len(peaks))]


def hrv_time_domain(rr_intervals_s: ArrayLike) -> Dict[str, float]:
    """Compute common time-domain HRV metrics from RR intervals in seconds.

    Returns SDNN, RMSSD, pNN50 (percentage of adjacent RR differences >50ms).
    """
    arr = np.asarray(rr_intervals_s, dtype=float)
    if arr.size == 0:
        return {"SDNN": 0.0, "RMSSD": 0.0, "pNN50": 0.0}
    diff = np.diff(arr)
    sdnn = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if diff.size > 0 else 0.0
    nn50 = np.sum(np.abs(diff) > 0.05)
    pnn50 = float(100.0 * nn50 / diff.size) if diff.size > 0 else 0.0
    return {"SDNN": sdnn, "RMSSD": rmssd, "pNN50": pnn50}


def hrv_frequency_domain(rr_intervals_s: ArrayLike, fs_interp: float = 4.0) -> Dict[str, float]:
    """Estimate LF/HF power and LF/HF ratio from RR intervals.

    Steps:
    - convert RR events to tachogram (RR values at R-peak times)
    - interpolate to even sampling at `fs_interp` Hz
    - compute PSD via Welch and integrate LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz)
    """
    rr = np.asarray(rr_intervals_s, dtype=float)
    if rr.size < 2:
        return {"LF": 0.0, "HF": 0.0, "LF_HF": 0.0}
    # create time vector for RR events (cumulative)
    times = np.cumsum(rr)
    # interpolate using numpy.interp (avoids scipy interp1d type issues)
    if times.size < 2:
        return {"LF": 0.0, "HF": 0.0, "LF_HF": 0.0}
    t_interp = np.arange(times[0], times[-1], 1.0 / fs_interp)
    rr_interp = np.interp(t_interp, times, rr)
    # detrend
    rr_interp = signal.detrend(rr_interp)
    fvals, Pxx = signal.welch(rr_interp, fs=fs_interp, nperseg=min(256, rr_interp.size))
    total = float(trapezoid(Pxx, fvals))
    lf_mask = (fvals >= 0.04) & (fvals < 0.15)
    hf_mask = (fvals >= 0.15) & (fvals <= 0.4)
    lf = float(trapezoid(Pxx[lf_mask], fvals[lf_mask])) if np.any(lf_mask) else 0.0
    hf = float(trapezoid(Pxx[hf_mask], fvals[hf_mask])) if np.any(hf_mask) else 0.0
    lf_hf = float(lf / hf) if hf > 0 else float('inf') if lf > 0 else 0.0
    return {"LF": lf, "HF": hf, "LF_HF": lf_hf, "total_power": total}


def compute_hr_from_rr(rr_intervals_s: ArrayLike) -> float:
    """Return average heart rate (bpm) from RR intervals in seconds."""
    arr = np.asarray(rr_intervals_s, dtype=float)
    if arr.size == 0:
        return 0.0
    mean_rr = float(np.mean(arr))
    return 60.0 / mean_rr if mean_rr > 0 else 0.0


def spectral_entropy(x: ArrayLike, fs: int = 128, nperseg: int = 256) -> float:
    """Compute spectral entropy of a signal using normalized PSD."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return 0.0
    f, Pxx = signal.welch(arr, fs=fs, nperseg=min(nperseg, arr.size))
    psd = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else Pxx
    # small epsilon for numerical stability
    psd = psd + 1e-12
    return float(stats.entropy(psd, base=2))


def compute_bandpowers(eeg: ArrayLike, fs: int = 128, bands: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
    """Return relative band powers for standard EEG bands.

    bands: dict mapping band name to (low, high)
    """
    if bands is None:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (12.0, 30.0),
            "gamma": (30.0, 45.0),
        }
    arr = np.asarray(eeg, dtype=float)
    if arr.size == 0:
        return {k: 0.0 for k in bands}
    f, Pxx = signal.welch(arr, fs=fs, nperseg=min(256, arr.size))
    total = float(trapezoid(Pxx, f))
    out = {}
    for name, (low, high) in bands.items():
        mask = (f >= low) & (f <= high)
        p = float(trapezoid(Pxx[mask], f[mask])) if np.any(mask) else 0.0
        out[name] = float(p / total) if total > 0 else 0.0
    out["alpha_beta_ratio"] = float(out.get("alpha", 0.0) / out.get("beta", 1e-12))
    return out


def coherence_between_signals(x: ArrayLike, y: ArrayLike, fs: int = 128, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Compute magnitude-squared coherence between two signals.

    Returns (frequencies, coherence_values)
    """
    arrx = np.asarray(x, dtype=float)
    arry = np.asarray(y, dtype=float)
    if arrx.size == 0 or arry.size == 0:
        return np.array([]), np.array([])
    f, Cxy = signal.coherence(arrx, arry, fs=fs, nperseg=min(nperseg, arrx.size))
    return f, Cxy


def spectrogram_for_cnn(x: ArrayLike, fs: int = 128, nperseg: int = 128, noverlap: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spectrogram (frequencies, times, Sxx) suitable for CNN inputs."""
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.array([]), np.array([]), np.array([[]])
    f, t, Sxx = signal.spectrogram(arr, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # convert to dB
    Sdb = 10.0 * np.log10(Sxx + 1e-12)
    return f, t, Sdb


def template_match_beats(ecg: ArrayLike, peaks: List[int], fs: int = 250, window_ms: int = 600) -> Dict[str, Any]:
    """Extract beat templates around `peaks` and compute correlation to template.

    Returns average template and per-beat correlation coefficients.
    """
    arr = np.asarray(ecg, dtype=float)
    if arr.size == 0 or len(peaks) == 0:
        return {"template": np.array([]), "correlations": np.array([])}
    half_w = int(window_ms * fs / 2000)
    beats = []
    for p in peaks:
        s = max(0, p - half_w)
        e = min(len(arr), p + half_w)
        beat = arr[s:e]
        # pad to same length
        beats.append(np.pad(beat, (0, 2 * half_w - len(beat)), mode="constant"))
    beats = np.vstack(beats)
    template = np.mean(beats, axis=0)
    # normalize
    tpl = (template - np.mean(template)) / (np.std(template) + 1e-12)
    corrs = []
    for b in beats:
        b_norm = (b - np.mean(b)) / (np.std(b) + 1e-12)
        corrs.append(float(np.correlate(b_norm, tpl).ravel()[0] / len(tpl)))
    return {"template": template, "correlations": np.array(corrs)}


def compute_qt_qtc(ecg: ArrayLike, peaks: List[int], fs: int = 250) -> Dict[str, float]:
    """Heuristic QT and QTc estimation.

    Notes: This is a heuristic implementation: it searches for Q as local
    minima before R within 60 ms and T as local maxima after R within 300 ms.
    QTc uses Bazett formula. Results should be validated on annotated data.
    """
    out = {"QT_s": float('nan'), "QTc_s": float('nan')}
    arr = np.asarray(ecg, dtype=float)
    if arr.size == 0 or len(peaks) == 0:
        return out
    r = peaks[0]
    # Q: search back up to 60ms
    q_search = int(0.06 * fs)
    t_search = int(0.4 * fs)
    s_q = max(0, r - q_search)
    q_idx = s_q + int(np.argmin(arr[s_q:r + 1]))
    # T: search forward 80-400 ms
    s_t = r + int(0.08 * fs)
    e_t = min(len(arr) - 1, r + t_search)
    if s_t >= e_t:
        return out
    t_idx = s_t + int(np.argmax(arr[s_t:e_t + 1]))
    qt = float((t_idx - q_idx) / fs)
    # RR ~ use previous RR if available else use mean RR
    rr_s = None
    if len(peaks) > 1:
        rr_s = float((peaks[1] - peaks[0]) / fs)
    else:
        rr_s = np.mean(np.diff(peaks)) / fs if len(peaks) > 1 else 1.0
    qtc = qt / (np.sqrt(rr_s) if rr_s > 0 else 1.0)
    out["QT_s"] = qt
    out["QTc_s"] = qtc
    return out


def qrs_duration(ecg: ArrayLike, peak_idx: int, fs: int = 250) -> float:
    """Estimate QRS duration (seconds) around a given R-peak index.

    Heuristic: find onset and offset as crossings of a fraction of peak amplitude.
    """
    arr = np.asarray(ecg, dtype=float)
    if arr.size == 0:
        return float('nan')
    r = peak_idx
    peak_amp = arr[r]
    thresh = peak_amp * 0.3
    # onset
    onset = r
    while onset > 0 and arr[onset] > thresh:
        onset -= 1
    offset = r
    while offset < len(arr) - 1 and arr[offset] > thresh:
        offset += 1
    return float((offset - onset) / fs)


def st_segment_deviation(ecg: ArrayLike, peak_idx: int, fs: int = 250, baseline_ms: int = 20, st_ms: int = 80) -> float:
    """Estimate ST-segment deviation in mV relative to baseline.

    Baseline is measured `baseline_ms` after the J-point (~at R+20ms), ST point at R+st_ms.
    """
    arr = np.asarray(ecg, dtype=float)
    if arr.size == 0:
        return float('nan')
    j_idx = min(len(arr) - 1, peak_idx + int(0.02 * fs))
    baseline_idx = min(len(arr) - 1, j_idx + int(baseline_ms * fs / 1000.0))
    st_idx = min(len(arr) - 1, peak_idx + int(st_ms * fs / 1000.0))
    baseline = arr[baseline_idx]
    st = arr[st_idx]
    return float(st - baseline)


def summary_ecg_features(ecg: ArrayLike, fs: int = 250) -> Dict[str, float]:
    peaks = pan_tompkins_detector(ecg, fs)
    rr = compute_rr_intervals(peaks, fs)
    avg_hr = 0.0
    mean_rr = 0.0
    if rr:
        mean_rr = float(sum(rr) / len(rr))
        avg_hr = 60.0 / mean_rr if mean_rr > 0 else 0.0
    return {"n_peaks": len(peaks), "avg_hr_bpm": float(avg_hr), "mean_rr_s": float(mean_rr)}


def welch_bandpower(eeg: ArrayLike, band: Tuple[float, float] = (8.0, 12.0), fs: int = 128, nperseg: int = 256) -> float:
    """Compute band power using Welch PSD.

    Returns relative power in the requested band (band power / total power).
    """
    arr = np.asarray(eeg, dtype=float)
    if arr.size == 0:
        return 0.0
    f, Pxx = signal.welch(arr, fs=fs, nperseg=min(nperseg, arr.size))
    from scipy.integrate import trapezoid
    total_power = float(trapezoid(Pxx, f))
    # band mask
    low, high = band
    mask = (f >= low) & (f <= high)
    band_power = float(trapezoid(Pxx[mask], f[mask])) if np.any(mask) else 0.0
    return float(band_power / total_power) if total_power > 0 else 0.0


# Backwards-compatible alias for earlier API
def simple_bandpower(eeg: ArrayLike, band: Tuple[float, float] = (8, 12), fs: int = 128) -> float:
    """Compatibility wrapper: returns relative bandpower using Welch."""
    return welch_bandpower(eeg, band=band, fs=fs)

