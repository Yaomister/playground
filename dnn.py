import torch, torch.nn as nn, torch.optim as optim


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":

    X = torch.rand(100, 2) * 10
    Y = (X[:, 0] + 2 * X[:, 1]).squeeze(0) + torch.rand(100, 1)

    model = DNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1000):
        predictions = model(X)
        loss = criterion(predictions, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"epoch {epoch + 1} | loss {loss}")

    X_test = torch.tensor([[4.0, 3.0], [7.0, 8.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")



    