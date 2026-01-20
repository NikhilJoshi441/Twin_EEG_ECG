"""Train fusion model using Stratified K-Fold cross-validation and save per-fold metrics.

Usage: python tools/train_kfold.py --data data/fusion_augmented_v2.jsonl --out reports/kfold_summary.json --folds 5 --epochs 20
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


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
        if ecg.size != ECG_LEN:
            ecg = np.resize(ecg, ECG_LEN)
        if eeg.size != EEG_LEN:
            eeg = np.resize(eeg, EEG_LEN)
        return torch.from_numpy(ecg), torch.from_numpy(eeg), torch.tensor(float(r.get('label',1)))


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


def run_fold(train_idx, val_idx, recs, device, epochs, batch):
    train_recs = [recs[i] for i in train_idx]
    val_recs = [recs[i] for i in val_idx]
    train_ds = FusionDataset(train_recs)
    val_ds = FusionDataset(val_recs)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)
    model = FusionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        for ecg, eeg, y in train_dl:
            ecg=ecg.to(device).float(); eeg=eeg.to(device).float(); y=y.to(device).float()
            opt.zero_grad(); out=model(ecg,eeg); loss=loss_fn(out,y); loss.backward(); opt.step()
    # eval
    ys=[]; probs=[]
    model.eval()
    with torch.no_grad():
        for ecg,eeg,y in val_dl:
            ecg=ecg.to(device).float(); eeg=eeg.to(device).float()
            out=model(ecg,eeg); p=torch.sigmoid(out)
            probs.extend(p.cpu().numpy().tolist()); ys.extend(y.numpy().tolist())
    preds=[1 if p>=0.5 else 0 for p in probs]
    report={
        'n_val': len(ys),
        'accuracy': accuracy_score(ys,preds),
        'precision': precision_score(ys,preds, zero_division=0),
        'recall': recall_score(ys,preds, zero_division=0),
        'f1': f1_score(ys,preds, zero_division=0)
    }
    try:
        report['auc']=roc_auc_score(ys,probs)
    except Exception:
        report['auc']=None
    return report, model.state_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    recs = load_jsonl(args.data)
    labels = [int(r.get('label',1)) for r in recs]
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    device = torch.device(args.device)
    summaries = []
    fold = 0
    for train_idx, val_idx in skf.split(recs, labels):
        fold += 1
        print('Running fold', fold)
        report, state = run_fold(train_idx, val_idx, recs, device, args.epochs, args.batch)
        report['fold'] = fold
        summaries.append(report)
    Path('reports').mkdir(exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump({'folds': summaries}, fh, indent=2)
    print('Saved kfold report to', args.out)


if __name__ == '__main__':
    main()
