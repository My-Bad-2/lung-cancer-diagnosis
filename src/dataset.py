from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

import pandas as pd
import os
import numpy as np
import json


def _default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


class LungHistDataset(Dataset):
    def __init__(self, csv_file_path, image_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file_path)
        self.image_dir = image_dir

        self.class_map = {
            'normal': 0,
            'adenocarcinoma_well': 1,
            'adenocarcinoma_moderately': 2,
            'adenocarcinoma_poorly': 3,
            'squamous_cell_carcinoma_well': 4,
            'squamous_cell_carcinoma_moderately': 5,
            'squamous_cell_carcinoma_poorly': 6,
        }

        self.image_paths = []
        self.labels = []

        for index, row in self.data_frame.iterrows():
            superclass = row['superclass']
            subclass = row['subclass'] if pd.notna(row['subclass']) else ''

            key = f'{superclass.lower()}_{subclass.lower()}'.strip('_')
            key = key.replace(' ', '_')
            if 'normal' in key:
                key = 'normal'

            if key in self.class_map:
                label = self.class_map[key]

                img_file = f'{superclass}_{subclass}_{row['resolution']}_{row['image_id']}.jpg'
                img_file = img_file.replace('None', '').replace(' ', '_').strip('_')
                img_file = img_file.replace('_.jpg', '.jpg').replace('__', '_')

                full_path = Path(os.path.join(self.image_dir, img_file))

                if os.path.exists(full_path):
                    self.image_paths.append(full_path)
                    self.labels.append(label)

        self.transform = transform if transform is not None else _default_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
