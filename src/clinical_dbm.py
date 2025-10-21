from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rbm import RBM


class ClinicalDBM(nn.Module):
    def __init__(
            self,
            n_visible: int,
            n_hidden: List[int],
            k: int = 1,
            num_classes: int = 1) -> None:
        super(ClinicalDBM, self).__init__()
        self.rbms = nn.ModuleList()
        in_dim = n_visible

        for h_dim in n_hidden:
            self.rbms.append(RBM(in_dim, h_dim, k))
            in_dim = h_dim

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden[-1], 128),
            nn.GELU(approximate='tanh'),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def pretrain(self, data_loader, device: torch.Device, epochs_per_layer: int = 15) -> None:
        print("--- Starting Unsupervised Pre-Training ---")
        current_data_loader = data_loader

        for i, rbm in enumerate(self.rbms):
            print(f"Training RBM Layer {i + 1}/{len(self.rbms)}...")
            rbm_optimizer = optim.SGD(rbm.parameters(), lr=0.05, momentum=0.9)

            for epoch in range(epochs_per_layer):
                epoch_loss = 0.0

                for batch, _ in tqdm(current_data_loader):
                    batch = (batch.to(device) - batch.min()) / (batch.max() - batch.min() + 1e-8)
                    rbm_optimizer.zero_grad()
                    loss = rbm(batch)
                    loss.backward()
                    rbm_optimizer.step()

                    epoch_loss += loss.item()
                print(f"   Epoch {epoch + 1}/{epochs_per_layer}, Loss: {epoch_loss / len(current_data_loader):.4f}")
            print("   Creating features for the next layer...")

            new_features = []
            new_labels = []

            with torch.no_grad():
                for batch, labels in tqdm(current_data_loader):
                    batch = (batch.to(device) - batch.min()) / (batch.max() - batch.min() + 1e-8)
                    new_features.append(rbm.pass_through(batch))
                    new_labels.append(labels)

            current_data_loader = DataLoader(TensorDataset(torch.cat(new_features), torch.cat(new_labels)),
                                             batch_size=data_loader.batch_size)
        print("--- Unsupervised Pre-Training Complete! ---")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for rbm in self.rbms:
            x = rbm.pass_through(x)
        return self.classifier(x)
