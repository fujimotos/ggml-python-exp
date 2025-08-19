import collections

import numpy as np
import torch
import safetensors.torch

class STFT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        n_fft = 256
        hop_length = 128
        win_length = 256
        pad_length = (0, 64)

        # Rows up to <offset> represent real value, and
        # rows after that represent imaginary value.
        self.offset = (n_fft // 2 + 1)
        out_channels = self.offset * 2

        self.conv = torch.nn.Conv1d(1, out_channels,
                                    kernel_size=win_length,
                                    stride=hop_length,
                                    bias=False)

        self.padding = torch.nn.ReflectionPad1d(pad_length)

    def forward(self, x):
        # Transform Tensor<N, L> into Tensor<N, 1, L> to match
        # the expectation of torch.nn.Conv1d.
        x = torch.unsqueeze(x, 1)
        x = self.padding(x)
        x = self.conv(x)
        return self.magnitude(x)

    def magnitude(self, x):
        real = x[:, :self.offset, :]
        imag = x[:, self.offset:, :]
        return torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))

class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.se = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("conv0", torch.nn.Conv1d(129, 128, 3, 1, 1)),
                    ("relu0", torch.nn.ReLU()),
                    ("conv1", torch.nn.Conv1d(128,  64, 3, 2, 1)),
                    ("relu1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv1d( 64,  64, 3, 2, 1)),
                    ("relu2", torch.nn.ReLU()),
                    ("conv3", torch.nn.Conv1d( 64, 128, 3, 1, 1)),
                    ("relu3", torch.nn.ReLU()),
                ]
            )
        )

    def forward(self, x):
        return self.se(x)

class LSTMDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h = None
        self.lstm = torch.nn.LSTM(128, 128)
        self.se = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("dropout", torch.nn.Dropout(0.1)),
                    ("relu",    torch.nn.ReLU()),
                    ("conv",    torch.nn.Conv1d(128, 1, 1, 1, 0)),
                    ("sigmoid", torch.nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        # Reorder Tensor dimension because PyTorch's LSTM expects
        # [seq, batch, features]
        x = x.permute(2,0,1)
        x, self.h = self.lstm(x, self.h)
        x = x.permute(1,2,0)
        return self.se(x)

class SileroVAD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = STFT()
        self.encoder = AudioEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x):
        x = self.stft(x)
        x = self.encoder(x)
        return self.decoder(x)

def test():
    torch.set_printoptions(sci_mode=False)

    model = SileroVAD()

    model.load_state_dict(
        safetensors.torch.load_file('data/silero_vad.safetensors')
    )
    model.eval()

    with open('data/jfk.raw', 'rb') as fp:
        audio = np.frombuffer(fp.read(), dtype=np.float32)
        audio = torch.tensor(audio)

    x = audio[48000:48576]
    x = torch.unsqueeze(x, 0)

    with torch.no_grad():
        x = model(x)

    for prob in x:
        print(f"P = {prob.item():.4f}")

if __name__ == '__main__':
    test()
