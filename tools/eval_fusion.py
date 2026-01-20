import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/fusion_raw.jsonl')
    p.add_argument('--model', default='src/models/fusion.pth')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='reports/fusion_eval.json')
    p.add_argument('--plots', default='reports')
    return p.parse_args()


args = parse_args()
DATA_FILE = args.data
MODEL_FILE = args.model
DEVICE = torch.device(args.device)

ECG_LEN = 500
EEG_LEN = 256


class EvalDataset(Dataset):
    def __init__(self, path):
        self.items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    ecg = obj.get('ecg') or []
                    eeg = obj.get('eeg') or []
                    # prefer precomputed fusion_label if present
                    if 'fusion_label' in obj:
                        try:
                            label = int(obj.get('fusion_label') or 0)
                        except Exception:
                            label = 0
                    else:
                        label = 1 if (obj.get('explanation') and (obj.get('explanation').get('prob') or 0) >= 0.6) else 0
                    self.items.append((ecg, eeg, label))
                except Exception:
                    continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ecg, eeg, label = self.items[idx]
        ecg = ecg[:ECG_LEN] + [0.0] * max(0, ECG_LEN - len(ecg))
        eeg = eeg[:EEG_LEN] + [0.0] * max(0, EEG_LEN - len(eeg))
        return torch.tensor(ecg, dtype=torch.float32), torch.tensor(eeg, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class EEGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32*(EEG_LEN//4),128), nn.ReLU()
        )
    def forward(self,x):
        return self.net(x.unsqueeze(1))


class ECGLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64,128)
    def forward(self,x):
        out,_ = self.lstm(x.unsqueeze(-1))
        return torch.relu(self.fc(out[:, -1, :]))


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg = EEGCNN()
        self.ecg = ECGLSTM()
        self.head = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, ecg, eeg):
        e1 = self.ecg(ecg)
        e2 = self.eeg(eeg)
        x = torch.cat([e1,e2], dim=1)
        return self.head(x).squeeze(1)


def main():
    if not os.path.exists(DATA_FILE):
        print('Data not found:', DATA_FILE); return
    ds = EvalDataset(DATA_FILE)
    if len(ds)==0:
        print('No items in dataset'); return
    n = len(ds)
    val_n = int(n*0.2)
    train_n = n - val_n
    gen = torch.Generator()
    gen.manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_n, val_n], generator=gen)

    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = FusionModel().to(DEVICE)
    state = torch.load(MODEL_FILE, map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except Exception as e:
        print('Failed to load state_dict:', e); return

    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for ecg,eeg,label in val_loader:
            ecg,eeg = ecg.to(DEVICE), eeg.to(DEVICE)
            out = model(ecg,eeg)
            prob = torch.sigmoid(out).cpu().numpy()
            ps.extend(prob.tolist())
            ys.extend(label.numpy().tolist())

    try:
        from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score
        auc = roc_auc_score(ys, ps)
        preds = [1 if p>=0.5 else 0 for p in ps]
        acc = accuracy_score(ys, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average='binary', zero_division=0)
        cm = confusion_matrix(ys, preds).tolist()
    except Exception as e:
        print('sklearn metrics failed:', e)
        auc=acc=prec=rec=f1=0.0; cm=[[0,0],[0,0]]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report = {
        'n_val': len(val_ds), 'auc': float(auc), 'accuracy': float(acc),
        'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'confusion_matrix': cm
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Saved evaluation report to', args.out)

    # optional plots
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fpr,tpr,_ = roc_curve(ys, ps)
        os.makedirs(args.plots, exist_ok=True)
        plt.figure()
        plt.plot(fpr,tpr,label=f'AUC={auc:.3f}')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend()
        plt.savefig(os.path.join(args.plots,'roc.png'))
        print('Saved ROC plot to', os.path.join(args.plots,'roc.png'))
    except Exception as e:
        print('Plotting skipped:', e)


if __name__=='__main__':
    main()
