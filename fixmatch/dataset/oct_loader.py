import os
import logging
import torch
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .randaugment import RandAugmentMC
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json

logger = logging.getLogger(__name__)


def get_mean_std(dataset, name, num_workers=4):
    if os.path.exists(f"mean_std_{name}.json"):
        with open(f"mean_std_{name}.json", "r") as f:
            return json.load(f)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in tqdm(loader):
        for i in range(3):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))

    mean = mean.tolist()
    std = std.tolist()
    with open(f"mean_std_{name}.json", "w") as f:
        json.dump([mean, std], f)
    return mean, std


def get_oct_loaders(root, batch_size=64, num_workers=4, pin_memory=True, drop_last=True, mu=7, image_dim=(256, 256)):
    root = Path(root)
    no_transform = transforms.Compose([transforms.Resize(image_dim), transforms.ToTensor()])
    # labeled_oct_mean, labeled_oct_std = get_mean_std(ImageFolder(root / "labeled", transform=no_transform), "labeled")
    # unlabeled_oct_mean, unlabeled_oct_std = get_mean_std(ImageFolder(root / "unlabeled", transform=no_transform), "unlabeled")
    labeled_oct_mean = unlabeled_oct_mean = [0.485, 0.456, 0.406]
    labeled_oct_std = unlabeled_oct_std = [0.229, 0.224, 0.225]
    
    transform_labeled = transforms.Compose(
        [
            transforms.Resize(image_dim),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=labeled_oct_mean, std=labeled_oct_std),
        ]
    )
    transform_val = transforms.Compose([transforms.Resize(image_dim),transforms.ToTensor(), transforms.Normalize(mean=labeled_oct_mean, std=labeled_oct_std)])
    transform_unlabeled = TransformFixMatch(mean=unlabeled_oct_mean, std=unlabeled_oct_std, image_dim=image_dim)

    labeled_len = len(ImageFolder(root / "labeled"))
    labeled_indices = list(range(labeled_len))
    np.random.shuffle(labeled_indices)
    train_labeled_indices = labeled_indices[: math.ceil(0.8 * labeled_len)]
    val_labeled_indices = labeled_indices[math.ceil(0.8 * labeled_len) : math.ceil(0.9 * labeled_len)]
    test_labeled_indices = labeled_indices[math.ceil(0.9 * labeled_len) :]

    train_labeled_dataset = torch.utils.data.Subset(ImageFolder(root / "labeled", transform=transform_labeled), train_labeled_indices)
    val_labeled_dataset = torch.utils.data.Subset(ImageFolder(root / "labeled", transform=transform_val), val_labeled_indices)
    test_labeled_dataset = torch.utils.data.Subset(ImageFolder(root / "labeled", transform=transform_val), test_labeled_indices)
    train_unlabeled_dataset = ImageFolder(root / "unlabeled", transform=transform_unlabeled)

    train_labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_labeled_loader = DataLoader(
        val_labeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_labeled_loader = DataLoader(
        test_labeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    train_unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=batch_size * mu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return train_labeled_loader, val_labeled_loader, train_unlabeled_loader, test_labeled_loader


class TransformFixMatch(object):
    def __init__(self, mean, std, image_dim):
        self.weak = transforms.Compose(
            [transforms.Resize(image_dim), transforms.RandomHorizontalFlip(), transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect")]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
