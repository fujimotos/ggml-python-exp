import time

import librosa
import numpy as np
import torch
import yaml

from silero import SileroVAD

JSUTBOOK_TRAIN_KEYS = [
    "AineKuraineNoNikki",
]

SAMPLERATE = 16000
DURATION = 12 * SAMPLERATE

def get_timestamps_from_yaml(fp):
    dic = yaml.safe_load(fp)

    for chap in dic:
        for para in dic[chap]:
            for style in dic[chap][para]:
                for text in dic[chap][para][style]['texts']:
                    yield text['time']

def load_audio_data(key):
    waveform = librosa.load(f'third_party/jsut-book_ver1/wav/{key}.wav', sr=SAMPLERATE)[0]

    # Ensure that the waveform length is dividable by DURATION.
    remainder = len(waveform) % DURATION
    if remainder:
        waveform = np.pad(waveform, (0, DURATION - remainder))

    # Mark tensor indices with voice activity
    label = np.zeros(len(waveform), dtype=np.float32)

    with open(f'third_party/jsut-book_ver1/txt/{key}.yaml') as fp:
        for timestamp in get_timestamps_from_yaml(fp):
            t1 = int(SAMPLERATE * timestamp[0])
            t2 = int(SAMPLERATE * timestamp[1]) + 1
            label[t1:t2] = 1.0

    waveform = torch.from_numpy(waveform.reshape(-1, DURATION))
    label = torch.from_numpy(label.reshape(-1, DURATION))

    return waveform, label

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        waveforms = []
        labels = []

        for key in JSUTBOOK_TRAIN_KEYS:
            waveform, label = load_audio_data(key)
            waveforms.append(waveform)
            labels.append(label)

        self.waveforms = torch.concat(waveforms)
        self.labels = torch.concat(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.waveforms[idx], self.labels[idx]

def train():
    epochs = 20
    batch_size = 32
    window_size = 512
    context_size = 64
    leaering_rate = 1e-2

    dataset = AudioDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SileroVAD()
    for param in model.stft.parameters():
        param.requires_grad = False

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=leaering_rate)

    print(f"Train SileroVAD for {epochs} epochs")
    for epoch in range(epochs):
        loss_all = []
        for X, y in loader:
            hidden = None
            context = torch.zeros(X.shape[0], context_size)

            for idx in range(0, X.shape[1], window_size):
                window_x = X[:, idx:idx + window_size]
                window_x = torch.cat((context, window_x), 1)

                # This is needed because PyTorch's autograd engine
                # would complain if we re-use the hidden tensor.
                if hidden is not None:
                    hidden = tuple([t.detach() for t in hidden])

                prediction, hidden = model(window_x, hidden)

                window_y = y[:, idx:idx + window_size]
                window_y = torch.mean(window_y, 1)

                loss = loss_fn(window_y, prediction)
                loss.backward()

                # Take the context for the next iteration
                context = window_x[:, -64:]
                loss_all.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()

        print(f"{time.ctime()} epoch={epoch} loss={np.mean(loss_all):.3f}")

    torch.save(model.state_dict(), f"{int(time.time())}.pkl")

if __name__ == '__main__':
    train()
