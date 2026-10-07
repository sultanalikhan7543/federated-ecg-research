import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class FLClient:
    def __init__(self, client_id, X, y, batch_size=64, lr=0.001, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.lr = lr

    def train(self, model, epochs=3, proximal_mu=0.0, global_weights=None, control_variate=None, server_cv=None):
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        model.train()

        for epoch in range(epochs):
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                if proximal_mu > 0 and global_weights is not None:
                    prox_term = 0.0
                    for p, gp in zip(model.parameters(), global_weights):
                        prox_term += ((p - gp) ** 2).sum()
                    loss += (proximal_mu / 2) * prox_term

                loss.backward()
                optimizer.step()

        return model.state_dict()

    def evaluate(self, model):
        model.to(self.device)
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        return correct / total, all_preds, all_labels