from torchvision.datasets import ImageFolder

from train import train_loop
from histo_net import HistoNet
import torchvision.transforms as T
import torch

import kagglehub
import torch.nn as nn

from pathlib import Path
import os

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset


class ImageFilelistDataset(Dataset):
    """Custom dataset that takes a list of image paths and labels."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE, EPOCHS_S1, EPOCHS_S2, EPOCHS_DBN_PRE, EPOCHS_DBN_FINE = 64, 50, 75, 20, 100

    # Download latest version
    path = kagglehub.dataset_download("ice778/comprehensive-lung-cancer-imaging-dataset-clid")

    HISTO_DATA_PATH = Path(path) / 'Histopathological Images'
    CT_SCAN_DATA_PATH = Path(path) / 'CT Scan'

    path = kagglehub.dataset_download("thedevastator/cancer-patients-and-air-pollution-a-new-link")

    CLINICAL_DATA_PATH = Path(path) / 'cancer patient data sets.csv'

    image_size = 256

    train_transform = T.Compose([
        T.RandomResizedCrop(image_size),

        # Aggressive geometric and spatial transformations
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(degrees=180, translate=(0.25, 0.25), scale=(0.7, 1.3), shear=(-30, 30, -30, 30)),
        T.RandomPerspective(distortion_scale=0.6, p=0.7),

        T.RandomApply([
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
        ], p=0.5),

        # Convert to Tensor and Normalize
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

        # Strong regularization via acclusion
        T.RandomErasing(p=0.75, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0, inplace=False),
    ])

    val_transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if os.path.exists(HISTO_DATA_PATH):
        full_dataset_info = ImageFolder(root=HISTO_DATA_PATH)
        paths, labels = zip(*full_dataset_info.samples)

        # Create stratified splits
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_ds = ImageFilelistDataset(train_paths, train_labels, transform=train_transform)
        val_ds = ImageFilelistDataset(val_paths, val_labels, transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

        model = HistoNet(num_classes=len(full_dataset_info.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        trained_model = train_loop(model, train_loader, val_loader, criterion, EPOCHS_S1, 'HistoNet', device)
    else:
        print(f"WARNING: Directory not found: {HISTO_DATA_PATH}. Skipping Stage 1.")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()