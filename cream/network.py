import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

import cream.simple_1d as s1d


class CSVDataset(Dataset):
    """Wrapper over the kind of CSV used as training data for the neural network"""

    data: pd.DataFrame

    def __init__(self, data) -> None:
        self.data = pd.read_csv(data)

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return torch.FloatTensor([[
            row["mass"],
            s1d.K_(
                row["drag_coefficient"],
                row["fluid_density"],
                row["cross_sectional_area"],
            ),
            row["gravitational_acceleration"],
            row["time"],
        ]]), torch.FloatTensor([[row["relative_position"]]])


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.stack(x)


def train(data, model, loss_fn, optimizer, batch_size):
    size = len(data.dataset)

    model.train()
    for batch, (X, y) in enumerate(data):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(data, model, loss_fn):
    size = len(data.dataset)
    num_batches = len(data)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in data:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"test - acc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}")
