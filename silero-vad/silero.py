import numpy as np
import torch
import safetensors.torch

class STFT(torch.nn.Module):
    def __init__(self, n_fft=256, hop_length=128, win_length=256, pad_length=(0, 64)):
        super().__init__()

        self.cutoff = (n_fft // 2 + 1)
        out_channels = self.cutoff * 2

        self.conv1d = torch.nn.Conv1d(in_channels=1,
                                      out_channels=out_channels,
                                      kernel_size=win_length,
                                      stride=hop_length,
                                      bias=False)

        self.padding = torch.nn.ReflectionPad1d(pad_length)

    def forward(self, x):
        # Transform Tensor<N, L> into Tensor<N, 1, L> to match
        # the expectation of torch.nn.Conv1d.
        x = torch.unsqueeze(x, 1)
        x = self.padding(x)
        x = self.conv1d(x)
        return self.magnitude(x)

    def magnitude(self, x):
        real = x[:, :self.cutoff, :]
        imag = x[:, self.cutoff:, :]
        return torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))

def test():
    torch.set_printoptions(sci_mode=False)

    stft = STFT()

    tensors = safetensors.torch.load_file('data/silero_vad.safetensors')
    with torch.no_grad():
        stft.conv1d.weight.data = tensors['stft.conv1d.weight']

    with open('data/jfk.raw', 'rb') as fp:
        audio = np.frombuffer(fp.read(), dtype=np.float32)
        audio = torch.tensor(audio)

    x = audio[48000:48576]
    x = torch.unsqueeze(x, 0)

    with torch.no_grad():
        y = stft(x)

    print(y)

if __name__ == '__main__':
    test()
