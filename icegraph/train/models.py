# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
from torch.optim import Adam
import torch.nn.functional as F

from icegraph.console import Console
from icegraph.data import DatasetRegistry

__all__ = ["Trainer"]


class Trainer:
    def __init__(self, dataset_registry: DatasetRegistry, model: torch.nn.Module, device: str = "cuda"):
        self.datasets = dataset_registry
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

    def train(self, num_epochs: int = 10):
        Console.banner("Trainer")
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_rmse = 0.0
            total = 0

            Console.out(f"[Train] Epoch {epoch + 1}/{num_epochs}")
            for batch in Console.progress_bar(self.datasets.train_dataloader):
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.batch)  # Assuming GravNet: no edge_index
                target = batch.y.view(-1, 1)
                loss = self.loss_fn(out, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch.y.size(0)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

            avg_loss = total_loss / total
            rmse = total_rmse / total
            Console.out(f"  → MSE: {avg_loss:.4f} | RMSE: {rmse:.4f}")

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        total = 0

        Console.out("[Validation]")
        with torch.no_grad():
            for batch in Console.progress_bar(self.datasets.val_dataloader):
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.batch)
                target = batch.y.view(-1, 1)
                loss = self.loss_fn(out, target)

                total_loss += loss.item() * batch.y.size(0)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

        avg_loss = total_loss / total
        rmse = total_rmse / total
        Console.out(f"  → MSE: {avg_loss:.4f} | RMSE: {rmse:.4f}")

    def test(self):
        self.model.eval()
        total_rmse = 0.0
        total = 0

        Console.out("[Test]")
        with torch.no_grad():
            for batch in Console.progress_bar(self.datasets.test_dataloader):
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.batch)
                target = batch.y.view(-1, 1)
                total_rmse += F.mse_loss(out, target, reduction="sum").sqrt().item()
                total += batch.y.size(0)

        rmse = total_rmse / total
        Console.out(f"  → RMSE: {rmse:.4f}")