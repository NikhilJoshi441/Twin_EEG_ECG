#!/usr/bin/env python3
"""Train a toy RandomForest on synthetic features and save to `src/models/rf.pkl`.

This creates a quick demo model so the server can show ML-powered alerts.
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE, 'src', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
OUT_PATH = os.path.join(MODEL_DIR, 'rf.pkl')

FEATURE_NAMES = ['LF_HF','SDNN','RMSSD','spectral_entropy','alpha_power','beta_power','alpha_beta','avg_hr_bpm','n_peaks']

def generate_sample():
    # normal ranges based on plausible values
    lf_hf = np.random.normal(1.0, 0.5)
    sdnn = max(0.01, np.random.normal(0.05, 0.02))
    rmssd = max(0.01, np.random.normal(0.03, 0.01))
    entropy = np.random.normal(3.5, 0.5)
    alpha = abs(np.random.normal(0.15, 0.08))
    beta = abs(np.random.normal(0.12, 0.06))
    alpha_beta = alpha / (beta + 1e-12)
    avg_hr = np.random.normal(70.0, 8.0)
    n_peaks = int(max(0, np.random.poisson(1)))
    return [lf_hf, sdnn, rmssd, entropy, alpha, beta, alpha_beta, avg_hr, n_peaks]

def label_from_features(x):
    # Simple heuristic: high lf/hf, low sdnn/rmssd, high entropy => anomaly
    lf_hf, sdnn, rmssd, entropy, alpha, beta, alpha_beta, avg_hr, n_peaks = x
    score = 0.0
    score += max(0.0, (lf_hf - 1.5))
    score += max(0.0, (0.06 - sdnn) * 5.0)
    score += max(0.0, (0.02 - rmssd) * 10.0)
    score += max(0.0, (entropy - 4.0))
    score += max(0.0, (alpha_beta - 1.5))
    return 1 if score > 0.8 else 0

def main(n=2000):
    X = []
    y = []
    for _ in range(n):
        v = generate_sample()
        X.append(v)
        y.append(label_from_features(v))
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Saved toy RF model to {OUT_PATH}")

if __name__ == '__main__':
    main()
