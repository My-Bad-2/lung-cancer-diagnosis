import torch
import torch.optim as optim

from early_stopping import EarlyStopping
from accuracy import calculate_accuracy
from capsule_loss import CapsuleLoss
from tqdm import tqdm
from pathlib import Path
import os
import json

def train_loop(model, train_loader, val_loader, criterion, epochs, model_name, device):
    print(f"\n--- Initializing Training Pipeline for {model_name} ---")

    history_path = Path(f'{model_name}_history')
    checkpoint_path = Path(f'{model_name}_checkpoint.pth')
    best_model_path = Path(f'{model_name}_best.pth')

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2)
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': [],
    }

    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"\n--- Loading Checkpoint {checkpoint_path}. Resuming Training ---")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        print(f"Resuming from epoch {start_epoch}.")
    else:
        print(f"No Checkpoint found for {model_name}. Starting from scratch.")

    early_stopping = EarlyStopping(patience=7, verbose=True, path=best_model_path, monitor='valid_loss', mode='min')

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if isinstance(criterion, CapsuleLoss):
                labels = torch.eye(model.class_caps.num_caps, device=device)[labels]

            optimizer.zero_grad()
            outputs = model(inputs)
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, labels) + 1e-5 * l1_norm
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += calculate_accuracy(outputs, labels, criterion=criterion)
            total_train += labels.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if isinstance(criterion, CapsuleLoss):
                    labels = torch.eye(model.class_caps.num_caps, device=device)[labels]

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += calculate_accuracy(outputs, labels, criterion=criterion)
                total_val += labels.size(0)

        # Calculate and Log Metrics
        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val
        avg_val_acc = val_correct / total_val
        avg_train_acc = train_correct / total_train

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['valid_loss'].append(avg_val_loss)
        history['valid_acc'].append(avg_val_acc)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }, checkpoint_path)

        scheduler.step()
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nTraining finished for {model_name}. Loading best model weights from {best_model_path}.")
    model.load_state_dict(torch.load(best_model_path))
    return model