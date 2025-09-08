import collections
import math

import numpy as np
import torch
import safetensors.torch
import librosa

def positional_encoding(length, d_model):
    pos = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    enc = torch.zeros(length, d_model)
    enc[:, 0::2] = torch.sin(pos * div)
    enc[:, 1::2] = torch.cos(pos * div)
    return enc

def mel(waveform):
    n_fft = 400
    hop_length = 160
    n_mels = 80
    samplerate = 16000
    epsiron = 1e-12

    stft = torch.stft(waveform, n_fft, hop_length,
                      window=torch.hann_window(n_fft),
                      return_complex=True)

    magnitudes = torch.pow(stft[..., :-1].abs(), 2)

    mel_filters = torch.from_numpy(
        librosa.filters.mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels)
    )

    return mel_filters @ magnitudes

class ResidualAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(128, 4, batch_first=True)
        self.attn_ln = torch.nn.LayerNorm(128)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 128*4),
            torch.nn.GELU(),
            torch.nn.Linear(128*4, 128)
        )
        self.mlp_ln = torch.nn.LayerNorm(128)

    def forward(self, x):
        x1 = self.attn_ln(x)
        x = x + self.attn(x1, x1, x1)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv1d(80, 128, 3, 1, 1)
        self.conv1 = torch.nn.Conv1d(128, 128, 3, 2, 1)
        self.register_buffer('positional_encoding', positional_encoding(150, 128))

        self.attn_list  = torch.nn.ModuleList(
            [ResidualAttention() for _ in range(4)]
        )

    def forward(self, x):
        x = torch.nn.functional.gelu(self.conv0(x))
        x = torch.nn.functional.gelu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = x + self.positional_encoding
        for attn in self.attn_list:
            x = attn(x)
        return x

class LSTMDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(384, 384)
        self.se = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("dropout", torch.nn.Dropout(0.1)),
                    ("relu",    torch.nn.ReLU()),
                    ("conv",    torch.nn.Conv1d(384, 1, 1, 1, 0)),
                    ("sigmoid", torch.nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x, h):
        # Reorder Tensor dimension because PyTorch's LSTM expects
        # [seq, batch, features]
        x = x.permute(2,0,1)
        x, h = self.lstm(x, h)
        x = x.permute(1,2,0)
        x = self.se(x)
        return x.squeeze(), h

class AttentionVAD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AudioEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x, h):
        x = self.encoder(x)
        x, h = self.decoder(x, h)
        return x, h
