import torch, os
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL='src/models/fusion_improved.pth'
OUT='src/models/fusion_improved_ts.pt'
ECG_LEN=500
EEG_LEN=256

def build_model(n_classes=2):
    # prefer importing the canonical FusionModel when available so tracing matches saved weights
    try:
        from src.models.fusion_model import FusionModel
        return FusionModel(n_classes=n_classes)
    except Exception:
        # fallback: simple local architecture compatible with prior exports
        import torch.nn as nn
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
            def __init__(self, n_classes=2):
                super().__init__()
                self.eeg = EEGCNN_local()
                self.ecg = ECGLSTM_local()
                self.head = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,n_classes))
            def forward(self, ecg, eeg):
                e1 = self.ecg(ecg)
                e2 = self.eeg(eeg)
                x = torch.cat([e1,e2], dim=1)
                out = self.head(x)
                # if single output, match prior behavior
                if out.shape[1] == 1:
                    return out.squeeze(1)
                return out

        return FusionModel_local(n_classes=n_classes)

def build_local_model(n_classes=2):
    import torch.nn as nn
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
        def __init__(self, n_classes=2):
            super().__init__()
            self.eeg = EEGCNN_local()
            self.ecg = ECGLSTM_local()
            self.head = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,n_classes))
        def forward(self, ecg, eeg):
            e1 = self.ecg(ecg)
            e2 = self.eeg(eeg)
            x = torch.cat([e1,e2], dim=1)
            out = self.head(x)
            if out.shape[1] == 1:
                return out.squeeze(1)
            return out

    return FusionModel_local(n_classes=n_classes)

def main():
    if not os.path.exists(MODEL):
        print('model missing', MODEL); return
    # detect number of classes from env THREAT_CLASSES or fallback to 2
    n_classes = 2
    tc = os.environ.get('THREAT_CLASSES')
    if tc:
        try:
            n_classes = max(2, len([c.strip() for c in tc.split(',') if c.strip()]))
        except Exception:
            n_classes = 2
    # load state dict first to decide which model variant to instantiate
    state = torch.load(MODEL, map_location='cpu')
    sd = state
    if isinstance(state, dict) and 'model_state_dict' in state:
        sd = state['model_state_dict']
    # if keys look like the older local-export layout, prefer local fallback
    use_local = False
    try:
        keys = list(sd.keys()) if isinstance(sd, dict) else []
        if any(k.startswith('eeg.net.') for k in keys):
            use_local = True
    except Exception:
        use_local = False

    if use_local:
        # build local architecture compatible with the saved state
        print('Detected local-style state dict; using local local_architecture')
        m = build_local_model(n_classes=n_classes)
    else:
        # try canonical FusionModel first
        try:
            from src.models.fusion_model import FusionModel
            m = FusionModel(n_classes=n_classes)
        except Exception:
            m = build_model(n_classes=n_classes)

    # attempt to load weights; prefer strict, fall back to lax loading
    try:
        if isinstance(sd, dict):
            m.load_state_dict(sd)
        else:
            m.load_state_dict(state)
    except Exception:
        try:
            if isinstance(state, dict) and 'model_state_dict' in state:
                m.load_state_dict(state['model_state_dict'])
            else:
                m.load_state_dict(sd, strict=False)
        except Exception as e:
            print('Warning: failed load_state_dict:', e)
    m.eval()
    print('Model built. n_classes=', n_classes)
    try:
        import inspect
        print('Model summary (modules):')
        for name, mod in m.named_modules():
            print('  ', name, type(mod))
    except Exception:
        pass
    # example inputs
    ecg = torch.zeros((1, ECG_LEN), dtype=torch.float32)
    eeg = torch.zeros((1, EEG_LEN), dtype=torch.float32)
    try:
        traced = torch.jit.trace(m, (ecg, eeg))
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        traced.save(OUT)
        print('Saved TorchScript to', OUT)
    except Exception as e:
        print('TorchScript export failed', e)

if __name__=='__main__':
    main()
