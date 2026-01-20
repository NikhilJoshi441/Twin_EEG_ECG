import os, json
import torch
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

DATA='data/fusion_improved.jsonl'
MODEL='src/models/fusion_improved.pth'
OUT='reports/saliency.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ECG_LEN=500
EEG_LEN=256

class EvalDS(torch.utils.data.Dataset):
    def __init__(self,path):
        self.items=[]
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                try:
                    obj=json.loads(line)
                    ecg=obj.get('ecg') or []
                    eeg=obj.get('eeg') or []
                    label=int(obj.get('fusion_label') or 0)
                    self.items.append((ecg,eeg,label))
                except Exception:
                    continue
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        ecg,eeg,label=self.items[idx]
        ecg = ecg[:ECG_LEN] + [0.0]*max(0,ECG_LEN-len(ecg))
        eeg = eeg[:EEG_LEN] + [0.0]*max(0,EEG_LEN-len(eeg))
        return torch.tensor(ecg,dtype=torch.float32), torch.tensor(eeg,dtype=torch.float32), label

def load_model(path):
    # Define the same model classes used by tools/train_fusion.py (matching state_dict keys)
    class EEGCNN_local(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1,16,7,padding=3), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16,32,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Flatten(), nn.Linear(32*(EEG_LEN//4),128), nn.ReLU()
            )
        def forward(self,x):
            return self.net(x.unsqueeze(1))

    class ECGLSTM_local(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
            self.fc = nn.Linear(64,128)
        def forward(self,x):
            out,_ = self.lstm(x.unsqueeze(-1))
            return torch.relu(self.fc(out[:, -1, :]))

    class FusionModel_local(nn.Module):
        def __init__(self):
            super().__init__()
            self.eeg = EEGCNN_local()
            self.ecg = ECGLSTM_local()
            self.head = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
        def forward(self, ecg, eeg):
            e1 = self.ecg(ecg)
            e2 = self.eeg(eeg)
            x = torch.cat([e1,e2], dim=1)
            return self.head(x).squeeze(1)

    m = FusionModel_local()
    m.load_state_dict(torch.load(path,map_location='cpu'))
    m.eval()
    return m

def main():
    if not os.path.exists(DATA) or not os.path.exists(MODEL):
        print('missing data or model'); return
    ds=EvalDS(DATA)
    if len(ds)==0:
        print('no data'); return
    n=len(ds)
    val_n=int(n*0.2)
    train_n=n-val_n
    gen=torch.Generator(); gen.manual_seed(42)
    train, val = torch.utils.data.random_split(ds,[train_n,val_n], generator=gen)
    vl = DataLoader(val, batch_size=1)
    model=load_model(MODEL)
    saliency_list=[]
    for i,(ecg,eeg,label) in enumerate(vl):
        if i>=20: break
        ecg = ecg.clone().requires_grad_(True)
        eeg = eeg.clone().requires_grad_(True)
        out = model(ecg,eeg)
        prob = torch.sigmoid(out)
        prob.backward()
        g_ecg = ecg.grad.abs().mean().item()
        g_eeg = eeg.grad.abs().mean().item()
        saliency_list.append({'index': i, 'label': int(label), 'prob': float(prob.item()), 'ecg_sal_mean': g_ecg, 'eeg_sal_mean': g_eeg})
    with open(OUT,'w',encoding='utf-8') as f:
        json.dump(saliency_list,f,indent=2)
    print('Saved saliency to', OUT)

if __name__=='__main__':
    main()
