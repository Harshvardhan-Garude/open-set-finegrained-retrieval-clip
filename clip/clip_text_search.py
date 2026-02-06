"""Text-to-image retrieval using CLIP text embeddings.

Requires precomputed image embeddings from:
  clip/clip_extract_embeddings.py

Example:
  python clip/clip_text_search.py --prompt "red pickup truck" --top_k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
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
    p = argparse.ArgumentParser(description="CLIP text-to-image retrieval.")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt, e.g. 'green sports bike'.")
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
    print(f"[clip_text_search] device={device}")

    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    db_embeds = np.load(emb_dir / "clip_embeddings.npy")
    db_paths = np.load(emb_dir / "clip_paths.npy", allow_pickle=True)
    db_embeds_t = torch.tensor(db_embeds, device=device, dtype=torch.float32)

    tokens = tokenizer([args.prompt]).to(device)
    with torch.no_grad():
        q = model.encode_text(tokens)
        q = q / q.norm(dim=-1, keepdim=True)

    sims = torch.matmul(db_embeds_t, q.squeeze(0))
    topk = torch.topk(sims, k=min(args.top_k, db_embeds_t.shape[0]))
    topk_idx = topk.indices.detach().cpu().numpy().tolist()

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

    fig.suptitle(f'Top-{len(topk_idx)} matches for: "{args.prompt}"', fontsize=14)
    out_path = out_dir / "text_query_topk.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[clip_text_search] saved -> {out_path}")


if __name__ == "__main__":
    main()
