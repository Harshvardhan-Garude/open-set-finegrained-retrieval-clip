"""Extract CLIP image embeddings for a folder-structured dataset.

Expected dataset layout (torchvision ImageFolder):
    <data_dir>/
      class_a/xxx.jpg
      class_b/yyy.jpg
      ...

Outputs (saved under --out_dir):
  - clip_embeddings.npy  (N, D) float32
  - clip_labels.npy      (N,) int64
  - clip_paths.npy       (N,) str (absolute or relative paths as stored by ImageFolder)
  - class_mapping.csv    mapping from numeric label -> class name

Notes
-----
- This script uses OpenCLIP and runs CLIP in eval mode.
- By default we use the OpenAI-pretrained ViT-B-32 weights.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import open_clip


def get_device() -> str:
    # Prefer CUDA > Apple MPS > CPU
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CLIP image embeddings from an ImageFolder dataset.")
    p.add_argument("--data_dir", type=str, default="data/merged_dataset", help="Path to ImageFolder root.")
    p.add_argument("--out_dir", type=str, default="outputs/clip", help="Directory to save .npy outputs.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--model", type=str, default="ViT-B-32", help="CLIP model name.")
    p.add_argument("--pretrained", type=str, default="openai", help="OpenCLIP pretrained tag.")
    p.add_argument("--save_every", type=int, default=200, help="Save partial outputs every N batches (0 disables).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[clip_extract_embeddings] device={device}")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    dataset = ImageFolder(str(data_dir), transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Save class mapping (label -> class name)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    pd.Series(idx_to_class).to_csv(out_dir / "class_mapping.csv")
    print(f"[clip_extract_embeddings] saved class mapping -> {out_dir / 'class_mapping.csv'}")

    all_embeds: list[np.ndarray] = []
    all_labels: list[int] = []
    all_paths: list[str] = []

    partial_prefix = out_dir / "partial_clip"

    try:
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Extracting CLIP embeddings")):
            images = images.to(device)

            with torch.no_grad():
                embeds = model.encode_image(images)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                embeds_np = embeds.detach().cpu().numpy().astype(np.float32)

            all_embeds.append(embeds_np)
            all_labels.extend(labels.cpu().numpy().tolist())

            # dataset.samples is a list[(path, class_idx)] aligned with dataset ordering
            start = batch_idx * args.batch_size
            end = start + labels.shape[0]
            all_paths.extend([p for (p, _) in dataset.samples[start:end]])

            if args.save_every and (batch_idx + 1) % args.save_every == 0:
                print(f"[clip_extract_embeddings] saving partial outputs at batch {batch_idx + 1}")
                np.save(f"{partial_prefix}_embeddings.npy", np.vstack(all_embeds))
                np.save(f"{partial_prefix}_labels.npy", np.asarray(all_labels, dtype=np.int64))
                np.save(f"{partial_prefix}_paths.npy", np.asarray(all_paths, dtype=object))

    except Exception as e:
        print(f"[clip_extract_embeddings] ERROR: {e}")
        print("[clip_extract_embeddings] saving partial progress before exiting...")
        np.save(f"{partial_prefix}_embeddings.npy", np.vstack(all_embeds))
        np.save(f"{partial_prefix}_labels.npy", np.asarray(all_labels, dtype=np.int64))
        np.save(f"{partial_prefix}_paths.npy", np.asarray(all_paths, dtype=object))
        raise

    np.save(out_dir / "clip_embeddings.npy", np.vstack(all_embeds))
    np.save(out_dir / "clip_labels.npy", np.asarray(all_labels, dtype=np.int64))
    np.save(out_dir / "clip_paths.npy", np.asarray(all_paths, dtype=object))
    print(f"[clip_extract_embeddings] done -> {out_dir}")


if __name__ == "__main__":
    main()
