import torch
import torch.nn as nn

from capsule_loss import CapsuleLoss

def calculate_accuracy(output, labels, criterion):
    if isinstance(criterion, CapsuleLoss):
        v_norm = torch.sqrt(torch.sum(torch.pow(output, 2), dim=2, keepdim=True) + 1e-8).squeeze()
        _, max_indices = torch.max(v_norm, 1)
        _, label_indices = torch.max(labels, 1)
        correct = (max_indices == label_indices).sum().item()
    elif isinstance(criterion, nn.BCEWithLogitsLoss):
        preds = torch.round(torch.sigmoid(output))
        correct = (preds == labels.unsqueeze(1)).sum().item()
    else:
        _, preds = torch.max(output, 1)
        correct = (preds == labels).sum().item()
    return correct