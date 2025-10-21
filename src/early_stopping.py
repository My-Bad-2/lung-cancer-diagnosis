from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.
    Saves the best model weights automatically.
    """

    def __init__(self,
                 patience: int = 10,
                 verbose: bool = False,
                 delta: int = 0,
                 monitor: str = 'val_loss',
                 path: Path = Path('checkpoint.pth'),
                 mode='min') -> None:
        if mode not in ['min', 'max']:
            raise ValueError(f"{mode} is not a valid mode ('min', 'max').")

        self.patience: int = patience
        self.verbose: bool = verbose
        self.delta: int = delta
        self.monitor: str = monitor
        self.path: Path = path
        self.counter: int = 0
        self.mode: str = mode
        self.early_stop: bool = False
        self.best_score: float = np.inf if self.mode == 'min' else -np.inf

    def __call__(self, score: float, model: nn.Module) -> None:
        is_improvement = False

        if self.mode == 'min' and score < (self.best_score - self.delta):
            is_improvement = True
        elif self.mode == 'max' and score > (self.best_score + self.delta):
            is_improvement = True

        if is_improvement:
            if self.verbose:
                print(f"{self.monitor} improved ({self.best_score:.6f} --> {score:.6f}). Saving model...")
                self.best_score = score
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model: nn.Module) -> None:
        """
        Saves model weights when monitored metric improves.
        """
        torch.save(model.state_dict(), self.path)
