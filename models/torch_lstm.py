import torch
import torch.nn as nn
import torch.optim as optim


class CustomLSTM(nn.Module):
    def __init__(self, num_hidden, num_channel, weight_decay=0.0):
        super(CustomLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=num_channel,
                            hidden_size=num_hidden,
                            batch_first=True)

        self.linear = nn.Linear(num_hidden, 1)
        self.weight_decay = weight_decay

    def forward(self, x):
        out, _ = self.lstm(x)              # (batch, T, hidden)
        out = out[:, -1, :]                # last time step
        out = self.linear(out)             # (batch, 1)
        return out

    def fit(self, x, y, batch_size=32, epochs=100, lr=1e-3, device="cuda"):

        self.to(device)
        self.train()

        x = x.to(device)
        y = y.to(device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        N = x.shape[0]

        for epoch in range(epochs):

            perm = torch.randperm(N, device=device)

            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]

                xb = x[idx]
                yb = y[idx]

                optimizer.zero_grad(set_to_none=True)

                pred = self(xb)
                loss = loss_fn(pred, yb)

                # L1 regularization (to match your Keras version)
                if self.weight_decay > 0:
                    l1_penalty = sum(p.abs().sum() for p in self.parameters())
                    loss = loss + self.weight_decay * l1_penalty

                loss.backward()
                optimizer.step()

        return self

    def predict(self, x, device="cuda"):
        self.to(device)
        self.eval()

        x = x.to(device)

        with torch.no_grad():
            pred = self(x)

        return pred