import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2,100)
        self.linear2 = torch.nn.Linear(100,1)

    def forward(self,x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return torch.nn.functional.sigmoid(x)

class DataSet(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
        self.data = []
        for i in range(n):
            a = np.random.random(2)
            b = np.sum((a - 0.5) ** 2) < 0.18
            x = torch.tensor(a, dtype=torch.float32)
            y = torch.tensor([b], dtype=torch.float32)
            self.data.append((x, y))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    dataset = torch.utils.data.DataLoader(DataSet(4096), batch_size=64, shuffle=1)

    m = Model()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-1)

    print("Training:")

    for e in range(20):
        for X, y in dataset:
            pred = m(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'{e:3d} {loss.item():.2f}')

    print("Result:")

    print("     ", end="")
    for x2 in np.linspace(0, 1, 11):
        print(f"{x2:5.2f}", end="")
    print()

    with torch.no_grad():
        for x1 in np.linspace(0, 1, 11):
            print("%5.2f" % x1, end="")
            for x2 in np.linspace(0, 1, 11):
                x = torch.tensor([x1, x2], dtype=torch.float32)
                y = m(x)
                print("%5.2f" % y, end="")
            print()

if __name__ == '__main__':
    main()
