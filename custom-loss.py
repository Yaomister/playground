import numpy
import math
import torch, torch.nn as nn, torch.functional as functional, torch.optim as optim



class HuberLoss(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, prediction, y):
        error = abs(y - prediction)
        loss = torch.where(error < self.delta, 0.5 * torch.square(y - prediction), self.delta * (error - 0.5 * self.delta))
        return loss.mean()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    X = torch.rand(100, 1) * 10
    y = 2 * X + 3 + torch.randn(100, 1)

    model = LinearRegressionModel()
    criterion = HuberLoss(delta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        predictions = model(X)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch} | loss {loss}")