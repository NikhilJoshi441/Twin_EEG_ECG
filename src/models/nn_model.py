"""PyTorch hybrid model: EEG-CNN + ECG-LSTM fusion classifier."""
import torch
import torch.nn as nn


class EEG_CNN(nn.Module):
    def __init__(self, in_channels=1, out_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, out_features)

    def forward(self, x):
        # x: (batch, channels=1, seq_len)
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class ECG_LSTM(nn.Module):
    def __init__(self, in_size=1, hidden_size=64, num_layers=1, out_features=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, (hn, cn) = self.lstm(x)
        # take last hidden
        h = hn[-1]
        return self.fc(h)


class FusionModel(nn.Module):
    def __init__(self, eeg_channels=1, ecg_in_size=1, hidden_features=64):
        super().__init__()
        self.eeg_net = EEG_CNN(in_channels=eeg_channels, out_features=hidden_features)
        self.ecg_net = ECG_LSTM(in_size=ecg_in_size, out_features=hidden_features)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_features * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, ecg, eeg):
        # ecg: (batch, seq_len, 1)
        # eeg: (batch, channels=1, seq_len)
        ecg_feat = self.ecg_net(ecg)
        eeg_feat = self.eeg_net(eeg)
        x = torch.cat([ecg_feat, eeg_feat], dim=1)
        return self.classifier(x)
