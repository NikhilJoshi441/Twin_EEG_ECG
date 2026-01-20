import argparse
import json, os, math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description='Train fusion model')
    p.add_argument('--data', default='data/fusion_raw.jsonl')
    p.add_argument('--out', default='src/models/fusion.pth')
    p.add_argument('--n_classes', type=int, default=2, help='number of output classes (>=2)')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ecg_len', type=int, default=500)
    p.add_argument('--eeg_len', type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


args = parse_args()
DATA_FILE = args.data
OUT_MODEL = args.out
DEVICE = torch.device(args.device)

# hyperparams (defaults can be overridden by CLI args)
ECG_LEN = args.ecg_len
EEG_LEN = args.eeg_len
BATCH = args.batch
EPOCHS = args.epochs
LR = args.lr
N_CLASSES = max(2, args.n_classes)

class FusionDataset(Dataset):
    def __init__(self, path):
        self.items = []
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                try:
                    obj=json.loads(line)
                    ecg = obj.get('ecg') or []
                    eeg = obj.get('eeg') or []
                    score = (obj.get('metrics') or {}).get('avg_hr_bpm',0)
                    # use threat score if present
                    threat = obj.get('metrics') and obj.get('metrics').get('avg_hr_bpm')
                    # label precedence: if a precomputed `fusion_label` exists in the record, use it;
                    # otherwise fall back to explanation.prob heuristic (legacy behavior)
                    if 'fusion_label' in obj:
                        try:
                            label = int(obj.get('fusion_label'))
                        except Exception:
                            label = 0
                    else:
                        # fallback: binary heuristic -> map to class 1 if above threshold
                        label = 1 if (obj.get('explanation') and (obj.get('explanation').get('prob') or 0) >= 0.6) else 0
                    self.items.append((ecg,eeg,label))
                except Exception:
                    continue
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ecg,eeg,label = self.items[idx]
        # pad/truncate
        ecg = ecg[:ECG_LEN] + [0.0]*max(0, ECG_LEN - len(ecg))
        eeg = eeg[:EEG_LEN] + [0.0]*max(0, EEG_LEN - len(eeg))
        lbl = torch.tensor(label, dtype=torch.long) if N_CLASSES>1 else torch.tensor(label, dtype=torch.float32)
        return torch.tensor(ecg, dtype=torch.float32), torch.tensor(eeg, dtype=torch.float32), lbl

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
    def __init__(self, n_classes=2):
        super().__init__()
        self.eeg = EEGCNN()
        self.ecg = ECGLSTM()
        self.head = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,n_classes))
        self.n_classes = n_classes
    def forward(self, ecg, eeg):
        e1 = self.ecg(ecg)
        e2 = self.eeg(eeg)
        x = torch.cat([e1,e2], dim=1)
        out = self.head(x)
        if self.n_classes==1:
            return out.squeeze(1)
        return out

# load data
if not os.path.exists(DATA_FILE):
    print('Data file not found:', DATA_FILE); raise SystemExit(1)

dset = FusionDataset(DATA_FILE)
if len(dset)==0:
    print('No items in dataset'); raise SystemExit(1)

# split
n = len(dset)
train_n = int(n*0.8)
train_ds, val_ds = torch.utils.data.random_split(dset, [train_n, n-train_n])
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)

model = FusionModel(n_classes=N_CLASSES).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
if N_CLASSES <= 1:
    crit = nn.BCEWithLogitsLoss()
else:
    crit = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    tot_loss=0.0
    for ecg,eeg,label in train_loader:
        ecg, eeg = ecg.to(DEVICE), eeg.to(DEVICE)
        label = label.to(DEVICE)
        opt.zero_grad()
        out = model(ecg,eeg)
        # CrossEntropyLoss expects raw logits with shape (N, C) and labels as (N,) long
        if N_CLASSES>1:
            loss = crit(out, label)
        else:
            loss = crit(out, label.float())
        loss.backward(); opt.step()
        tot_loss += loss.item()*ecg.size(0)
    print(f'Epoch {epoch+1}/{EPOCHS} train_loss={tot_loss/len(train_ds):.4f}')
    # val
    model.eval()
    with torch.no_grad():
        tot=0; lossv=0.0
        for ecg,eeg,label in val_loader:
            ecg,eeg = ecg.to(DEVICE), eeg.to(DEVICE)
            label = label.to(DEVICE)
            out=model(ecg,eeg)
            if N_CLASSES>1:
                loss=crit(out,label)
            else:
                loss=crit(out,label.float())
            lossv += loss.item()*ecg.size(0)
            tot += ecg.size(0)
        print(f' val_loss={lossv/max(1,tot):.4f}')

# save
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
torch.save(model.state_dict(), OUT_MODEL)
print('Saved fusion model to', OUT_MODEL)
