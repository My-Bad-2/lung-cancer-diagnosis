import torch
from torch.utils.data import DataLoader
from pathlib import Path

import numpy as np
import json
import os


def save_checkpoint(path, epoch, model, optimizer, scheduler, history):
    """
    Saves model, optimizer state, epoch, scheduler state, and training history.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'learning_rate': optimizer.param_groups['lr']
        }, path)
    print(f'Checkpoint saved to {path} at Epoch {epoch}.')


def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Loads checkpoint and returns start epoch and history.
    """
    start_epoch = 1
    history = {}

    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        lr_loaded = checkpoint.get('learning_rate', optimizer.param_groups['lr'])
        print(f'Resuming training from Epoch {start_epoch}. Loaded LR: {lr_loaded:.6f}.')
        return start_epoch, history

    print('No checkpoint found. Starting training from Epoch 1.')
    return start_epoch, history
