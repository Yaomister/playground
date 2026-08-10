import torch
from torch import nn, optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def activation(self, x):
        return torch.tanh(x) + x

    def forward(self, x):
        x = self.activation(self.linear(x))
        return x


if __name__ == "__main__":


    model = Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    epochs = 1000

    X = torch.rand(100, 1) * 10
    y = 2 * X + 3 * torch.rand(100, 1)

    for epoch in range(epochs):
        predictions = model(X)

        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1} : loss {loss}")






