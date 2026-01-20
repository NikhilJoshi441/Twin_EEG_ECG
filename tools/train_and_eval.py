"""Train and evaluate the fusion model on a stratified train/validation split.

Produces a saved model checkpoint and evaluation report (JSON + ROC plot).

Usage: python tools/train_and_eval.py --data data/fusion_augmented.jsonl --out-model src/models/fusion_balanced_final.pth
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt


EEG_LEN = 250
ECG_LEN = 250


class FusionDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        ecg = np.asarray(r['ecg'], dtype=np.float32)
        eeg = np.asarray(r['eeg'], dtype=np.float32)
        label = float(r.get('label', 1))
        # ensure lengths
        if ecg.size != ECG_LEN:
            ecg = np.resize(ecg, ECG_LEN)
        if eeg.size != EEG_LEN:
            eeg = np.resize(eeg, EEG_LEN)
        return torch.from_numpy(ecg), torch.from_numpy(eeg), torch.tensor(label)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32*(EEG_LEN//4), 128), nn.ReLU()
        )
        self.ecg_lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.ecg_fc = nn.Linear(64, 128)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, ecg, eeg):
        # ecg: (B, L), eeg: (B, L)
        x_eeg = self.eeg(eeg.unsqueeze(1))
        out, _ = self.ecg_lstm(ecg.unsqueeze(-1))
        x_ecg = torch.relu(self.ecg_fc(out[:, -1, :]))
        x = torch.cat([x_ecg, x_eeg], dim=1)
        return self.head(x).squeeze(1)


def load_jsonl(path):
    recs = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            recs.append(json.loads(line))
    return recs


def evaluate_model(model, dl, device):
    model.eval()
    ys = []
    probs = []
    with torch.no_grad():
        for ecg, eeg, y in dl:
            ecg = ecg.to(device).float()
            eeg = eeg.to(device).float()
            out = model(ecg, eeg)
            prob = torch.sigmoid(out)
            probs.extend(prob.cpu().numpy().tolist())
            ys.extend(y.numpy().tolist())
    return ys, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out-model', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    recs = load_jsonl(args.data)
    labels = [int(r.get('label', 1)) for r in recs]
    train_idx, val_idx = train_test_split(list(range(len(recs))), test_size=0.2, stratify=labels, random_state=42)
    train_recs = [recs[i] for i in train_idx]
    val_recs = [recs[i] for i in val_idx]

    train_ds = FusionDataset(train_recs)
    val_ds = FusionDataset(val_recs)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    device = torch.device(args.device)
    model = FusionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for ecg, eeg, y in train_dl:
            ecg = ecg.to(device).float()
            eeg = eeg.to(device).float()
            y = y.to(device).float()
            opt.zero_grad()
            out = model(ecg, eeg)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            count += y.size(0)
        train_loss = running / max(1, count)

        # val
        model.eval()
        running = 0.0
        count = 0
        with torch.no_grad():
            for ecg, eeg, y in val_dl:
                ecg = ecg.to(device).float()
                eeg = eeg.to(device).float()
                y = y.to(device).float()
                out = model(ecg, eeg)
                loss = loss_fn(out, y)
                running += loss.item() * y.size(0)
                count += y.size(0)
        val_loss = running / max(1, count)
        print(f"Epoch {ep}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.out_model)

    # load best
    model.load_state_dict(torch.load(args.out_model, map_location=device))
    ys, probs = evaluate_model(model, val_dl, device)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    report = {
        'n_val': len(ys),
        'accuracy': accuracy_score(ys, preds),
        'precision': precision_score(ys, preds, zero_division=0),
        'recall': recall_score(ys, preds, zero_division=0),
        'f1': f1_score(ys, preds, zero_division=0)
    }
    try:
        report['auc'] = roc_auc_score(ys, probs)
    except Exception:
        report['auc'] = None

    out_report = Path('reports')
    out_report.mkdir(exist_ok=True)
    with open(out_report / 'fusion_balanced_eval.json', 'w', encoding='utf-8') as fh:
        json.dump(report, fh)

    # ROC
    try:
        fpr, tpr, _ = roc_curve(ys, probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.savefig(out_report / 'roc_balanced.png')
        plt.close()
    except Exception:
        pass

    print('Saved evaluation report to reports/fusion_balanced_eval.json')


if __name__ == '__main__':
    main()
