import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import GroupKFold
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Log1pTransform:
    def __call__(self, img):
        return torch.log1p(img)

def idx_to_class(idx, num_classes=3):
    return ["separated", "stable", "separationstarted"][idx]

class SolubilityDataset(Dataset):
    def __init__(self, root, image_paths, targets, transform=None):
        self.root = root
        self.image_paths = [os.path.join(root, image_path) for image_path in image_paths]

        self.class_to_idx = {"separated": 0, "stable": 1, "separationstarted": 2}
        self.idx_to_class = ["separated", "stable", "separationstarted"]
        # Normalize targets: strip spaces and lowercase
        self.targets = [self.class_to_idx[target.strip().lower()] for target in targets if target.strip().lower() in self.class_to_idx]
        self.image_paths = [self.image_paths[i] for i, target in enumerate(targets) if target.strip().lower() in self.class_to_idx]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        class_label = self.targets[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(class_label), image_path



def get_transforms(config):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.CenterCrop(config['center_crop']),
        transforms.RandomAffine(degrees=config['degrees'], translate=config['translate'], scale=(config['scale_lower'], config['scale_upper'])),
        transforms.Resize(config['resize']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.CenterCrop(config['center_crop']),
        transforms.Resize(config['resize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, test_transform


def prep_data(train_df, val_df, test_df, config):
    train_transform, test_transform = get_transforms(config)

    train_dataset = SolubilityDataset(config['data_dir'], train_df['file'].tolist(), train_df['class'].tolist(), transform=train_transform)
    val_dataset = SolubilityDataset(config['data_dir'], val_df['file'].tolist(), val_df['class'].tolist(), transform=test_transform)
    test_dataset = SolubilityDataset(config['data_dir'], test_df['file'].tolist(), test_df['class'].tolist(), transform=test_transform)

    # Weighted sampler to address class imbalance
    class_counts = np.bincount(train_dataset.targets, minlength=3)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)

    return train_loader, val_loader, test_loader



def cross_val_solubility(config):
    df = pd.read_csv(os.path.join(config['data_dir'], "annotations.csv"))
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna()

    df['class'] = df['class'].astype(str).str.strip().str.lower()
    # Filter only the 3 classes we care about
    df = df[df['class'].isin(["separated", "stable", "separationstarted"])]

    groups = df["groupid"]  

    test_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43) 
    val_sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43) 

    for trainval_idx, test_idx in test_sgkf.split(df, y=df["class"], groups=groups):
        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]

        trainval_groups = trainval_df["groupid"]

        for train_idx, val_idx in val_sgkf.split(trainval_df, y=trainval_df["class"], groups=trainval_groups):
            train_df = trainval_df.iloc[train_idx]
            val_df = trainval_df.iloc[val_idx]
            break

        yield prep_data(train_df, val_df, test_df, config)




