import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class LinearRegressionDataset(Dataset):
    def __init__(self, X, Y):
        super(LinearRegressionDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]



class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    X = torch.rand(100, 1)
    Y = 2 * X + 3 * torch.rand(100, 1)

    dataset = LinearRegressionDataset(X, Y)
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000

    for epoch in range(epochs):
        for batch_X, batch_Y in dataLoader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1} | loss {loss}")