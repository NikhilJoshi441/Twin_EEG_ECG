"""Training script for fusion model (EEG CNN + ECG LSTM).

This is a minimal training loop using synthetic data from `src.simulator`.
It demonstrates dataset, dataloader, training steps and checkpointing.
"""
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from src.models.fusion_model import FusionModel
from src.simulator import generate_ecg, generate_eeg
import numpy as np


class SyntheticFusionDataset(Dataset):
    def __init__(self, n_samples=1000, ecg_len=250, eeg_len=128):
        self.n = n_samples
        self.ecg_len = ecg_len
        self.eeg_len = eeg_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # generate synthetic windows and a random label
        ecg = np.asarray(generate_ecg(self.ecg_len / 250.0, fs=250), dtype=np.float32)
        eeg = np.asarray(generate_eeg(self.eeg_len / 128.0, fs=128), dtype=np.float32)
        # prepare shapes: eeg -> (channels, seq_len) ; ecg -> (seq_len, features)
        eeg = eeg.reshape(1, -1)
        ecg = ecg.reshape(-1, 1)
        label = np.random.randint(0, 2)
        return eeg, ecg, label


def collate_fn(batch):
    eegs = [torch.tensor(b[0]) for b in batch]
    ecgs = [torch.tensor(b[1]) for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    eegs = torch.stack(eegs, dim=0)
    ecgs = torch.stack(ecgs, dim=0)
    return eegs, ecgs, labels


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionModel(eeg_in_ch=1, ecg_input_size=1, hidden=64, n_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = SyntheticFusionDataset(n_samples=200 if args.quick else 2000)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        model.train()
        tot_loss = 0.0
        for eeg, ecg, lbl in dl:
            eeg = eeg.to(device).float()
            # eeg shape: (batch, ch, seq)
            ecg = ecg.to(device).float()
            lbl = lbl.to(device)
            # ensure shapes
            if eeg.dim() == 2:
                eeg = eeg.unsqueeze(1)
            if ecg.dim() == 3 and ecg.shape[-1] == 1:
                pass
            logits = model(eeg, ecg)
            loss = F.cross_entropy(logits, lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
        print(f"Epoch {epoch+1}/{args.epochs} loss={tot_loss/len(dl):.4f}")
    torch.save(model.state_dict(), args.checkpoint)
    print('Saved checkpoint to', args.checkpoint)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--checkpoint', type=str, default='fusion_model.pt')
    args = p.parse_args()
    train(args)
