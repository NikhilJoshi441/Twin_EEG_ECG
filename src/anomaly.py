"""Anomaly detection helper module.

Provides simple RandomForest baseline (sklearn) and LSTM baseline (PyTorch)
training wrappers. Imports sklearn lazily to avoid hard dependency at import time.
"""
from typing import Any, Tuple
import numpy as np
import os
import pickle


def train_rf_baseline(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, random_state: int = 42) -> Any:
    """Train a RandomForest classifier as a quick baseline.

    Returns the fitted model. Requires scikit-learn.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError('scikit-learn is required for RF baseline') from e
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)
    return clf


def predict_rf(model: Any, X: np.ndarray) -> np.ndarray:
    try:
        # prefer predict_proba when available
        probs = model.predict_proba(X)
        probs = np.asarray(probs)
        # common shapes: (N,2) -> return prob of positive class at index 1
        if probs.ndim == 2:
            if probs.shape[1] >= 2:
                return probs[:, 1]
            if probs.shape[1] == 1:
                return probs[:, 0]
        # 1D array of probabilities
        if probs.ndim == 1:
            return probs.reshape(-1)
    except Exception:
        # fallback: some estimators only implement predict() -> return label as float
        try:
            preds = model.predict(X)
            preds = np.asarray(preds)
            # If preds are integer class labels, cast to float probabilities (0.0/1.0)
            if preds.ndim == 1:
                return preds.astype(float).reshape(-1)
        except Exception:
            pass
    # ultimate fallback: zeros
    return np.zeros((X.shape[0],), dtype=float)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class LSTMBaseline(nn.Module):
    def __init__(self, input_size=1, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return torch.sigmoid(self.fc(last)).squeeze(-1)


def train_lstm_baseline(X: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 32) -> Tuple[nn.Module, dict]:
    """Train a tiny LSTM for sequence anomaly detection. X shape: (N, T, F)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMBaseline(input_size=X.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

    dl = DataLoader(SimpleDataset(X, y), batch_size=batch_size, shuffle=True)
    history = {'loss': []}
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        history['loss'].append(tot / len(dl))
    return model, history


def save_rf_model(model: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_rf_model(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_rf_feature_importances(model: Any, feature_names: list) -> dict:
    try:
        imps = getattr(model, 'feature_importances_', None)
        if imps is None:
            return {}
        imps = list(map(float, imps))
        total = sum(imps) if sum(imps) > 0 else 1.0
        return {n: imps[i] / total for i, n in enumerate(feature_names)}
    except Exception:
        return {}


def save_lstm_model(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_lstm_model(path: str, input_size: int = 1, hidden: int = 64, num_layers: int = 1):
    if not os.path.exists(path):
        return None
    mdl = LSTMBaseline(input_size=input_size, hidden=hidden, num_layers=num_layers)
    mdl.load_state_dict(torch.load(path, map_location='cpu'))
    mdl.eval()
    return mdl


def predict_lstm(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predict probabilities for input X of shape (N, T, F) using the LSTM model."""
    if model is None:
        return np.zeros((X.shape[0],), dtype=float)
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')
    with torch.no_grad():
        t_in = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(t_in)
        if isinstance(preds, torch.Tensor):
            out = preds.cpu().numpy()
            return out.reshape(-1)
    return np.zeros((X.shape[0],), dtype=float)


def load_fusion_model(path: str, device: str = 'cpu'):
    try:
        if not os.path.exists(path):
            return None
        # import here to avoid circular imports
        from src.models.fusion_model import FusionModel
        mdl = FusionModel()
        mdl.load_state_dict(torch.load(path, map_location=device))
        mdl.to(device)
        mdl.eval()
        return mdl
    except Exception:
        return None


def load_fusion_ts(path: str, device: str = 'cpu'):
    try:
        if not os.path.exists(path):
            return None
        mdl = torch.jit.load(path, map_location=device)
        mdl.eval()
        return mdl
    except Exception:
        return None


def predict_fusion(model, ecg_np, eeg_np):
    """Predict probability from fusion model. ecg_np: (N, ECG_LEN), eeg_np: (N, EEG_LEN)"""
    if model is None:
        return np.zeros((ecg_np.shape[0],), dtype=float)
    try:
        device = next(model.parameters()).device if hasattr(model, 'parameters') and any(True for _ in model.parameters()) else torch.device('cpu')
    except Exception:
        device = torch.device('cpu')
    with torch.no_grad():
        ecg_t = torch.tensor(ecg_np, dtype=torch.float32).to(device)
        eeg_t = torch.tensor(eeg_np, dtype=torch.float32).to(device)
        try:
            out = model(ecg_t, eeg_t)
        except Exception:
            # try swapped order
            out = model(eeg_t, ecg_t)
        # out may be:
        # - scalar logits per sample (N,) or (N,1): treat as binary logit -> sigmoid
        # - vector logits per sample (N, C): treat as multiclass logits -> softmax
        try:
            if isinstance(out, torch.Tensor):
                if out.dim() == 1:
                    # shape (N,) -> sigmoid
                    prob = torch.sigmoid(out).cpu().numpy()
                    return prob.reshape(-1)
                if out.dim() == 2 and out.size(1) == 1:
                    prob = torch.sigmoid(out).squeeze(-1).cpu().numpy()
                    return prob.reshape(-1)
                if out.dim() == 2 and out.size(1) > 1:
                    # multiclass logits -> softmax probabilities
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    return probs
        except Exception:
            pass
        # fallback: try converting to numpy
        try:
            arr = np.asarray(out)
            if arr.ndim == 1:
                return arr.reshape(-1)
            return arr
        except Exception:
            return np.zeros((ecg_np.shape[0],), dtype=float)
