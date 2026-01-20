"""Evaluation script for the fusion model on a synthetic validation split.

Loads checkpoint and runs evaluation, printing metrics.
"""
import argparse
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from train import SyntheticDataset, collate_fn
from src.models.nn_model import FusionModel


def evaluate(args):
    ds = SyntheticDataset(n_samples=args.samples, duration_s=args.duration)
    val_count = max(1, int(0.1 * len(ds)))
    train_count = len(ds) - val_count
    _, val_ds = torch.utils.data.random_split(ds, [train_count, val_count])
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, target_ecg_len=int(args.duration*250), target_eeg_len=int(args.duration*128)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for ecg, eeg, lbl in val_dl:
            ecg = ecg.to(device)
            eeg = eeg.to(device)
            out = model(ecg, eeg)
            preds = out.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(lbl.numpy().tolist())

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    cli()
