"""Training script for the hybrid NeuroCardiac model using synthetic data."""
import argparse
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.dataset import synthesize_pair
from src.data.physionet_loader import PhysioNetECGLoader
from src.simulator import generate_eeg
from src.models.nn_model import FusionModel
from src.feature_extraction import summary_ecg_features, welch_bandpower


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=200, duration_s=5.0, ecg_fs=250, eeg_fs=128, anomaly_prob=0.3):
        self.samples = []
        for _ in range(n_samples):
            lab = random.random() < anomaly_prob
            ecg, eeg, lbl = synthesize_pair(duration_s, ecg_fs, eeg_fs, anomaly=lab)
            # keep as numpy arrays for later resampling in collate
            self.samples.append((ecg, eeg, int(lbl)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PhysioNetDataset(Dataset):
    """Dataset that reads ECG segments from a local PhysioNet WFDB folder and pairs with synthetic EEG.

    It requires wfdb to be installed and a local copy of records in WFDB format.
    """
    def __init__(self, root_dir: str, segment_s: float = 5.0, max_segments: int = 200, target_fs: int = 250):
        self.loader = PhysioNetECGLoader(root_dir, target_fs=target_fs)
        self.segments = []
        # infer record names by listing files with .dat
        files = [f for f in os.listdir(root_dir) if f.endswith('.dat')]
        record_names = [os.path.splitext(f)[0] for f in files]
        for rec in record_names:
            for seg, idx in self.loader.iter_segments(rec, segment_s=segment_s, overlap=0.0):
                # seg is 1D numpy array (ecg)
                eeg = generate_eeg(segment_s, fs=128)
                # store as numpy arrays
                self.segments.append((seg.astype('float32'), np.array(eeg, dtype='float32'), 0))
                if len(self.segments) >= max_segments:
                    break
            if len(self.segments) >= max_segments:
                break

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]


def collate_fn(batch, target_ecg_len=1250, target_eeg_len=640):
    """Resample and pack batch to fixed lengths for training.

    target_ecg_len: number of samples for ECG (e.g., 5s at 250Hz = 1250)
    target_eeg_len: number of samples for EEG (e.g., 5s at 128Hz = 640)
    """
    ecg_list = []
    eeg_list = []
    labels = []
    from scipy import signal

    for ecg_np, eeg_np, lbl in batch:
        # resample arrays to target lengths
        ecg_res = signal.resample(ecg_np, target_ecg_len)
        eeg_res = signal.resample(eeg_np, target_eeg_len)
        ecg_list.append(torch.from_numpy(ecg_res).unsqueeze(1))
        eeg_list.append(torch.from_numpy(eeg_res).unsqueeze(0))
        labels.append(torch.tensor(lbl, dtype=torch.long))

    ecg_packed = torch.stack(ecg_list).float()
    eeg_packed = torch.stack(eeg_list).float()
    lbl_tensor = torch.stack(labels)
    return ecg_packed, eeg_packed, lbl_tensor


def train(args):
    if getattr(args, 'use_physionet', False):
        if not args.physionet_root:
            raise ValueError('When --use-physionet is set, --physionet-root must be provided')
        ds = PhysioNetDataset(root_dir=args.physionet_root, segment_s=args.duration, max_segments=args.samples, target_fs=250)
    else:
        ds = SyntheticDataset(n_samples=args.samples, duration_s=args.duration)
    # split train/val
    val_count = max(1, int(0.1 * len(ds)))
    train_count = len(ds) - val_count
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_count, val_count])

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, target_ecg_len=int(args.duration*250), target_eeg_len=int(args.duration*128)))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, target_ecg_len=int(args.duration*250), target_eeg_len=int(args.duration*128)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for ecg, eeg, lbl in dl:
            ecg = ecg.to(device)
            eeg = eeg.to(device)
            lbl = lbl.to(device)
            optim.zero_grad()
            out = model(ecg, eeg)
            loss = criterion(out, lbl)
            loss.backward()
            optim.step()
            total_loss += loss.item() * lbl.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == lbl).sum().item()
            count += lbl.size(0)
        train_acc = correct / max(1, count)

        # validation
        model.eval()
        vcorrect = 0
        vcount = 0
        with torch.no_grad():
            for ecg, eeg, lbl in val_dl:
                ecg = ecg.to(device)
                eeg = eeg.to(device)
                lbl = lbl.to(device)
                out = model(ecg, eeg)
                preds = out.argmax(dim=1)
                vcorrect += (preds == lbl).sum().item()
                vcount += lbl.size(0)
        val_acc = vcorrect / max(1, vcount)

        print(f"Epoch {epoch+1}/{args.epochs} train_loss={total_loss/max(1,count):.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        # checkpoint if improved
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, f"fusion_model_best.pt"))

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(args.outdir, "fusion_model_final.pt"))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--use-physionet", action="store_true", help="Use local PhysioNet WFDB records for ECG")
    p.add_argument("--physionet-root", type=str, default=None, help="Local folder containing WFDB records (.dat/.hea)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    cli()
