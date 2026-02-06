"""Extract embeddings from a trained CNN embedding model.

Typical use:
  1) Train a baseline (e.g., models/model_training_improved.py) which saves a .pth
  2) Run this script to embed the entire dataset and save numpy arrays
  3) Run evaluation/evaluate_embeddings.py for plots + Recall@K

Outputs (under --out_dir):
  - embeddings.npy
  - labels.npy
  - paths.npy
  - class_mapping.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from models.eff_model import EffB0Embedding


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CNN embeddings for retrieval evaluation.")
    p.add_argument("--data_dir", type=str, default="data/merged_dataset")
    p.add_argument("--weights", type=str, default=None, help="Path to .pth weights (state_dict). If omitted, uses ImageNet init.")
    p.add_argument("--out_dir", type=str, default="outputs/cnn_baseline")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[extract_embeddings] device={device}")

    model = EffB0Embedding(pretrained=True).to(device).eval()
    if args.weights:
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[extract_embeddings] loaded weights: {args.weights}")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds = ImageFolder(args.data_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    pd.Series(idx_to_class).to_csv(out_dir / "class_mapping.csv")

    embeds_list = []
    labels_list = []
    paths_list = []

    with torch.no_grad():
        for batch_idx, (imgs, y) in enumerate(tqdm(loader, desc="Embedding images")):
            imgs = imgs.to(device)
            z = model(imgs).cpu().numpy().astype(np.float32)
            embeds_list.append(z)
            labels_list.extend(y.numpy().tolist())

            start = batch_idx * args.batch_size
            end = start + y.shape[0]
            paths_list.extend([p for (p, _) in ds.samples[start:end]])

    np.save(out_dir / "embeddings.npy", np.vstack(embeds_list))
    np.save(out_dir / "labels.npy", np.asarray(labels_list, dtype=np.int64))
    np.save(out_dir / "paths.npy", np.asarray(paths_list, dtype=object))
    print(f"[extract_embeddings] saved -> {out_dir}")


if __name__ == "__main__":
    main()
