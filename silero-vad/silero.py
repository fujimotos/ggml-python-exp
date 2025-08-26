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

    def forward(self, x, h):
        # Reorder Tensor dimension because PyTorch's LSTM expects
        # [seq, batch, features]
        x = x.permute(2,0,1)
        x, h = self.lstm(x, h)
        x = x.permute(1,2,0)
        x = self.se(x)
        return x.squeeze(), h

class SileroVAD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = STFT()
        self.encoder = AudioEncoder()
        self.decoder = LSTMDecoder()

    def forward(self, x, h):
        x = self.stft(x)
        x = self.encoder(x)
        return self.decoder(x, h)

# The example code below shows how to apply SileroVAD() class
# to audio waveform. You can execute it like this:
#
#   $ python3 silero.vad
#   Saved: jfk-00005120-00034816.raw
#   Saved: jfk-00053760-00055808.raw
#   Saved: jfk-00057856-00059392.raw
#   ...
#
# Note that each file contains a speech segment (from <start> samples
# to <end> samples).

def apply_vad(model, waveform):
    nsamples = len(waveform)
    threshold = 0.5
    window_size = 512
    context_size = 64

    context = torch.zeros(context_size)
    h = None
    start = None

    for idx in range(0, nsamples, window_size):
        window = waveform[idx:idx+window_size]
        x = torch.cat((context, window), 0).unsqueeze(0)

        with torch.no_grad():
            y, h = model(x, h)

        context = window[-64:]

        if start is None:
            if y.item() > threshold:
                start = idx
        else:
            if y.item() < threshold:
                yield (start, idx)
                start = None

    if start is not None:
        yield (start, nsamples)

def main():
    model = SileroVAD()

    model.load_state_dict(
        safetensors.torch.load_file('data/silero_vad.safetensors')
    )
    model.eval()

    with open('data/jfk.raw', 'rb') as fp:
        waveform = np.frombuffer(fp.read(), dtype=np.float32)
        waveform = torch.from_numpy(waveform.copy())

    for start, end in apply_vad(model, waveform):
        name = f"jfk-{start:08d}-{end:08d}.raw"
        data = waveform[start:end].detach().numpy().tobytes()

        with open(name, "wb") as fp:
            fp.write(data)

        print(f"Saved: {name}")

if __name__ == '__main__':
    main()
