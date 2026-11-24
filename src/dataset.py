"""
PyTorch Dataset and DataLoader module for Cat and Dog Classification.
Uses stratified splits defined in results/dataset_split.json.
"""

import json
from pathlib import Path
from PIL import Image
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms():
    """
    Returns standard transforms:
    - Train: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalization
    - Validation/Test: Resize(256), CenterCrop(224), Normalization
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


class CatDogDataset(Dataset):
    """
    Dataset loader referencing split paths and labels.
    """
    def __init__(self, split_name: str, split_json_path: str = "results/dataset_split.json", transform: Optional[Callable] = None):
        self.split_name = split_name
        self.transform = transform

        with open(split_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for path_str, meta in data["splits"].items():
            if meta["split"] == split_name:
                self.samples.append({
                    "path": path_str,
                    "label": meta["label"],
                    "category": meta["category"]
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        label = sample["label"]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32), sample["path"]


def get_dataloaders(batch_size: int = 32, num_workers: int = 2):
    train_tf, eval_tf = get_transforms()

    train_ds = CatDogDataset(split_name="train_internal", transform=train_tf)
    val_ds = CatDogDataset(split_name="val_internal", transform=eval_tf)
    test_ds = CatDogDataset(split_name="holdout_test", transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader
