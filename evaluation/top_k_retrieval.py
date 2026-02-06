"""Top-K retrieval visualization for a query image.

Works with embeddings saved by evaluation/extract_embeddings.py.
By default uses cosine distance on the precomputed embeddings.

If you set --encode_query_with_model, it will encode the query image using the
same embedding model weights before retrieval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from torchvision import transforms

from models.eff_model import EffB0Embedding


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize top-K retrieval results.")
    p.add_argument("--query_image", type=str, required=True)
    p.add_argument("--embeddings", type=str, default="outputs/cnn_baseline/embeddings.npy")
    p.add_argument("--paths", type=str, default="outputs/cnn_baseline/paths.npy")
    p.add_argument("--out_path", type=str, default="outputs/cnn_baseline/topk.png")
    p.add_argument("--top_k", type=int, default=10)

    p.add_argument("--encode_query_with_model", action="store_true", help="Encode query using model instead of using precomputed embedding of same file.")
    p.add_argument("--weights", type=str, default=None, help="Optional model weights to encode query.")
    return p.parse_args()


def encode_query(query_image: str, device: str, weights: str | None) -> np.ndarray:
    model = EffB0Embedding(pretrained=True).to(device).eval()
    if weights:
        sd = torch.load(weights, map_location=device)
        model.load_state_dict(sd, strict=False)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img = Image.open(query_image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model(x).cpu().numpy().astype(np.float32)
    return z


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.load(args.embeddings).astype(np.float32)
    paths = np.load(args.paths, allow_pickle=True)

    device = get_device()

    if args.encode_query_with_model:
        q = encode_query(args.query_image, device, args.weights)
    else:
        # If user doesn't encode with model, we still need a query vector; encode it.
        q = encode_query(args.query_image, device, args.weights)

    d = cosine_distances(q, X)[0]
    idxs = np.argsort(d)[: args.top_k]

    cols = min(5, args.top_k)
    rows = int(np.ceil(args.top_k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= len(idxs):
            continue
        img_path = str(paths[idxs[i]])
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_title(f"Rank {i+1}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[top_k_retrieval] saved -> {out_path}")


if __name__ == "__main__":
    main()
