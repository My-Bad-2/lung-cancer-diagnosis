import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split

from train import train_model
from vit_capsnet import SGViTCapsNet, CapsuleHybridLoss
from dataset import LungHistDataset

from utils import load_checkpoint

import os
from pathlib import Path

def main(data_root_path):
    CSV_FILE = 'data.csv'
    IMAGE_DIR = 'images'
    NUM_CLASSES = 7
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    CHECKPOINT_FILE = 'checkpoint.pth'

    csv_path = Path(os.path.join(data_root_path, CSV_FILE))
    image_path = Path(os.path.join(data_root_path, IMAGE_DIR))

    if not os.path.exists(csv_path):
        print(f'Error: Metadata file not found at {csv_path}. Please check path.')
        return

    full_dataset = LungHistDataset(csv_path, image_path)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(67)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f'Total samples: {len(full_dataset)}. Training samples: {train_size}, Validation samples: {val_size}')

    model = SGViTCapsNet(num_classes=NUM_CLASSES).to(DEVICE)

    for param in model.vit.parameters():
        param.requires_grad = True

    criterion = CapsuleHybridLoss(lambda_reg=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

    start_epoch, history = load_checkpoint(CHECKPOINT_FILE, model, optimizer, scheduler, DEVICE)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, start_epoch, DEVICE, history, CHECKPOINT_FILE)
    print("\n--- Final Training History ---")
    for record in history:
        print(
            f"E{record['epoch']:02d} | Lr: {record['lr']:.6f} | T Loss: {record['train_loss']:.4f} | V Acc: {record['val_acc']:.2f}%")

if __name__ == '__main__':
    USER_DATA_PATH = Path('')

    if USER_DATA_PATH.exists():
        main(USER_DATA_PATH)