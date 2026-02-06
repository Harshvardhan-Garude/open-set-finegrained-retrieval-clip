"""CNN baseline training: EfficientNet-B0 + metric learning.

This script trains an EfficientNet-B0 embedding model on an ImageFolder dataset
using:
  - Triplet loss (pytorch-metric-learning)
  - A miner operating on coarse labels (coarse-to-fine idea)
  - Class-balanced sampling via WeightedRandomSampler
  - Simple early stopping on validation loss
  - Optional t-SNE snapshot plots every N epochs

Expected dataset layout:
    <data_dir>/<fine_class_name>/*.jpg

Coarse label is inferred from the folder name by splitting on '_' and taking the first token.
Example: "truck_red1" -> coarse="truck"
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner

from models.eff_model import EffB0Embedding


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FineGrainedDataset(Dataset):
    """Wraps ImageFolder but returns (image, fine_label, coarse_label)."""

    def __init__(self, root: str, transform=None) -> None:
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.samples = self.dataset.samples
        self.transform = transform

        # fine label = ImageFolder class idx
        self.fine_labels = [label for _, label in self.samples]

        # coarse label derived from class name (folder)
        coarse_names: List[str] = []
        for _, label in self.samples:
            fine_name = self.dataset.classes[label]
            coarse_names.append(fine_name.split("_")[0])

        uniq = sorted(set(coarse_names))
        self.coarse_label_map = {name: idx for idx, name in enumerate(uniq)}
        self.coarse_labels = [self.coarse_label_map[c] for c in coarse_names]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        return img, self.fine_labels[idx], self.coarse_labels[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 embedding model with metric learning.")
    p.add_argument("--data_dir", type=str, default="data/merged_dataset")
    p.add_argument("--out_dir", type=str, default="outputs/cnn_baseline")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tsne_every", type=int, default=4, help="Save a t-SNE plot every N epochs (0 disables).")
    return p.parse_args()


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
    xy = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(xy[:, 0], xy[:, 1], c=labels, s=6, cmap="tab20")
    plt.title(title)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[model_training_improved] device={device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = FineGrainedDataset(args.data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Class-balanced sampler over *fine* labels (on the train subset)
    train_targets = [full_dataset.fine_labels[i] for i in train_dataset.indices]
    class_counts = Counter(train_targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = EffB0Embedding().to(device)

    # Triplet loss + miner (using coarse labels to mine informative triplets)
    loss_func = TripletMarginLoss(margin=0.2)
    miner = TripletMarginMiner(margin=0.3, type_of_triplets="hard")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience_ctr = 0
    best_path = out_dir / "efficientnet_best.pth"

    def validate() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for imgs, fine_y, coarse_y in val_loader:
                imgs = imgs.to(device)
                fine_y = fine_y.to(device)
                coarse_y = coarse_y.to(device)

                z = model(imgs)
                triplets = miner(z, coarse_y)
                if triplets[0].nelement() == 0:
                    continue
                loss = loss_func(z, fine_y, triplets)
                total += float(loss.item())
                n += 1
        return total / max(n, 1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0

        for imgs, fine_y, coarse_y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(device)
            fine_y = fine_y.to(device)
            coarse_y = coarse_y.to(device)

            z = model(imgs)
            triplets = miner(z, coarse_y)
            if triplets[0].nelement() == 0:
                continue

            loss = loss_func(z, fine_y, triplets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            steps += 1

        train_loss = total / max(steps, 1)
        val_loss = validate()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}")

        # Save t-SNE snapshots (validation embeddings)
        if args.tsne_every and epoch % args.tsne_every == 0:
            model.eval()
            all_z, all_y = [], []
            with torch.no_grad():
                for imgs, fine_y, _ in val_loader:
                    z = model(imgs.to(device)).cpu().numpy()
                    all_z.append(z)
                    all_y.append(fine_y.numpy())
            Z = np.concatenate(all_z, axis=0)
            Y = np.concatenate(all_y, axis=0)
            plot_tsne(Z, Y, out_dir / f"tsne_epoch_{epoch}.png", title=f"t-SNE embeddings (epoch {epoch})")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), best_path)
            print(f"[model_training_improved] saved best -> {best_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[model_training_improved] early stopping at epoch {epoch}")
                break

    # Loss curves
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(out_dir / "loss_plot.png", dpi=200)
    plt.close()

    print("[model_training_improved] done.")


if __name__ == "__main__":
    main()
