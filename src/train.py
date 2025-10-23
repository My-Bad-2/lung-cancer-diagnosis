import torch
from tqdm import tqdm
from vit_capsnet import SGViTCapsNet
from utils import save_checkpoint

def validate_model(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            probs, digit_caps = model(data, saliency_attention_weights=None)
            loss = criterion(probs, digit_caps, labels)

            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(probs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = val_loss / total_samples
    avg_accuracy = correct_predictions / total_samples

    return avg_loss, avg_accuracy


def train_model(model: SGViTCapsNet, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs, start_epoch, device, history, checkpoint_file):
    """
    Manages the training loop with SGAR dual pass, Grad-CAM++, and checkpointing
    """

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        training_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        current_lr = optimizer.param_groups['lr']

        for data, labels in tqdm(train_dataloader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            h_g = model.target_layer.register_hook(model._save_gradient)
            probs_pass1, digit_caps_pass1 = model(data, saliency_attention_weights=None)
            loss_pass1 = criterion(probs_pass1, digit_caps_pass1, labels)
            loss_pass1.backward(retain_graph=True)
            h_g.remove()

            saliency_weights = model._calculate_saliency_attention()

            optimizer.zero_grad()
            probs_pass2, digit_caps_pass2 = model(data, saliency_attention_weights=saliency_weights.detach())
            loss_final = criterion(probs_pass2, digit_caps_pass2, labels)

            loss_final.backward()
            optimizer.step()

            running_loss = loss_final.item()
            _, predicted = torch.max(probs_pass2, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        scheduler.step()

        train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct_predictions / total_samples

        val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)

        print(f'Epoch {epoch}/{epochs}')
        print(f'\tCurrent LR: {current_lr:.6f}')
        print(f'\tTrain Loss: {train_loss:.4f}')
        print(f'\tTrain Accuracy: {train_accuracy:.4f}')
        print(f'\tVal Loss: {val_loss:.4f}')
        print(f'\tVal Accuracy: {val_accuracy:.4f}')

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'lr': current_lr,
        })

        save_checkpoint(checkpoint_file, epoch, model, optimizer, scheduler, history)

    print('Training finished!')
