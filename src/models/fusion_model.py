import torch
import torch.nn as nn


class EEG_CNN(nn.Module):
    """Simple 1D CNN encoder for EEG spectrograms or raw channels.

    Expects input shape (batch, channels, seq_len) or (batch, 1, freq_bins, time)
    For spectrogram inputs use preprocessing to produce a single channel time-frequency map
    and reshape accordingly when passing through this module.
    """
    def __init__(self, in_channels=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden * 2),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        h = self.net(x)
        return h.view(h.size(0), -1)


class ECG_LSTM(nn.Module):
    """LSTM encoder for ECG RR-series or raw ECG windows."""
    def __init__(self, input_size=1, hidden=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # take last hidden state
        last = out[:, -1, :]
        return last


class FusionModel(nn.Module):
    """Dual-stream fusion model: EEG encoder (CNN) + ECG encoder (LSTM) -> classifier/regressor.

    Supports early-fusion by concatenating features before dense layers.
    """
    def __init__(self, eeg_in_ch=1, ecg_input_size=1, hidden=64, n_classes=2):
        super().__init__()
        self.eeg_enc = EEG_CNN(in_channels=eeg_in_ch, hidden=hidden)
        self.ecg_enc = ECG_LSTM(input_size=ecg_input_size, hidden=hidden)
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden * 2 + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, eeg_x, ecg_x):
        # eeg_x: (batch, channels, seq_len)
        # ecg_x: (batch, seq_len, features)
        eeg_feat = self.eeg_enc(eeg_x)
        ecg_feat = self.ecg_enc(ecg_x)
        fused = torch.cat([eeg_feat, ecg_feat], dim=1)
        return self.fusion_fc(fused)