"""DataLoader helper for triplet training.

This is a small utility that:
  - loads an ImageFolder dataset
  - wraps it into a TripletDataset (anchor/positive/negative sampling)
  - returns a PyTorch DataLoader

Use this if you want to inspect triplets being produced during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.triplet_dataset import TripletDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create and sanity-check a Triplet DataLoader.")
    p.add_argument("--data_dir", type=str, default="data/merged_dataset", help="ImageFolder root directory.")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def make_triplet_loader(data_dir: str, batch_size: int) -> DataLoader:
    base = ImageFolder(data_dir)
    image_paths = [p for (p, _) in base.samples]
    labels = [y for (_, y) in base.samples]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = TripletDataset(image_paths, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    args = parse_args()
    loader = make_triplet_loader(args.data_dir, args.batch_size)

    # Quick sanity-check: one batch
    anchor, positive, negative, label = next(iter(loader))
    print("Batch shapes:")
    print("  anchor  :", tuple(anchor.shape))
    print("  positive:", tuple(positive.shape))
    print("  negative:", tuple(negative.shape))
    print("  label   :", tuple(label.shape))


if __name__ == "__main__":
    main()
