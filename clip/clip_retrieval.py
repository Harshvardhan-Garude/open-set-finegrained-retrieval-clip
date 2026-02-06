"""Image-to-image retrieval using precomputed CLIP embeddings.

Workflow:
  1) Run clip/clip_extract_embeddings.py to produce embeddings + paths.
  2) Use this script to query a single image and visualize top-K matches.

Output:
  - query_topk.png saved under --out_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import open_clip


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP image-to-image retrieval.")
    p.add_argument("--query_image", type=str, required=True, help="Path to a query image.")
    p.add_argument("--emb_dir", type=str, default="outputs/clip", help="Directory containing clip_embeddings.npy, clip_paths.npy.")
    p.add_argument("--out_dir", type=str, default="outputs/clip", help="Directory to save visualizations.")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--model", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="openai")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[clip_retrieval] device={device}")

    # Load CLIP model + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()

    # Load precomputed database embeddings
    db_embeds = np.load(emb_dir / "clip_embeddings.npy")
    db_paths = np.load(emb_dir / "clip_paths.npy", allow_pickle=True)

    db_embeds_t = torch.tensor(db_embeds, device=device, dtype=torch.float32)

    # Encode query image
    q_img = preprocess(Image.open(args.query_image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(q_img)
        q = q / q.norm(dim=-1, keepdim=True)

    # Cosine similarity with normalized embeddings == dot product
    sims = torch.matmul(db_embeds_t, q.squeeze(0))
    topk = torch.topk(sims, k=min(args.top_k, db_embeds_t.shape[0]))
    topk_idx = topk.indices.detach().cpu().numpy().tolist()

    # Plot top-k images
    cols = 5
    rows = int(np.ceil(len(topk_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        ax.axis("off")
        if ax_i >= len(topk_idx):
            continue
        idx = topk_idx[ax_i]
        img_path = str(db_paths[idx])
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_title(Path(img_path).name, fontsize=9)

    fig.suptitle(f"Top-{len(topk_idx)} CLIP matches", fontsize=14)
    out_path = out_dir / "query_topk.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[clip_retrieval] saved -> {out_path}")


if __name__ == "__main__":
    main()
