
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)



if __name__ == "__main__":

    writer = SummaryWriter()

    X = torch.rand(100, 1) * 10
    y = 2 * X + 3 * torch.rand(1, 1)

    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        predictions = model(X)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss, epoch)

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1} | loss {loss}")

    writer.flush()
    writer.close()