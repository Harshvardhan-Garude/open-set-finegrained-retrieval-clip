"""End-to-end CLIP retrieval + evaluation dashboard.

What it does:
  1) Optional: visualize top-K retrieval for a query image (if --query_image provided)
  2) Compute Recall@{1,5,10} on the embedding set (self-retrieval excluding self match)
  3) Produce PCA(50) + t-SNE(2) visualization of embeddings

Inputs:
  - clip_embeddings.npy, clip_paths.npy under --emb_dir
  - labels inferred from folder name in paths (ImageFolder class)

Outputs (under --out_dir):
  - query_topk.png               (if query image provided)
  - clip_recall_scores.csv
  - clip_recall_plot.png
  - clip_tsne_plot.png

Notes:
  - For large datasets, t-SNE can be slow. Use --tsne_max_points to subsample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import open_clip


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP retrieval + evaluation utilities.")
    p.add_argument("--emb_dir", type=str, default="outputs/clip", help="Directory containing clip_embeddings.npy, clip_paths.npy.")
    p.add_argument("--out_dir", type=str, default="outputs/clip", help="Directory to save plots/metrics.")
    p.add_argument("--query_image", type=str, default=None, help="Optional query image path for qualitative top-K.")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--model", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="openai")
    p.add_argument("--tsne_max_points", type=int, default=5000, help="Subsample to this many points for t-SNE (0 disables).")
    return p.parse_args()


def infer_label_from_path(p: str) -> str:
    # Assumes ImageFolder structure: .../<class_name>/<filename>
    parts = Path(p).parts
    return parts[-2] if len(parts) >= 2 else "unknown"


def compute_recall_at_k(emb: np.ndarray, labels: np.ndarray, k_list=(1, 5, 10)) -> dict[str, float]:
    # Cosine distance = 1 - cosine similarity.
    d = pairwise_distances(emb, emb, metric="cosine")
    np.fill_diagonal(d, np.inf)  # exclude self
    nn = np.argsort(d, axis=1)

    recalls: dict[str, float] = {}
    for k in k_list:
        correct = 0
        for i in range(len(labels)):
            nbrs = nn[i, :k]
            if labels[i] in labels[nbrs]:
                correct += 1
        recalls[f"Recall@{k}"] = correct / len(labels)
    return recalls


def save_query_topk(
    model,
    preprocess,
    device: str,
    query_image: str,
    db_embeds_t: torch.Tensor,
    db_paths: np.ndarray,
    out_path: Path,
    top_k: int,
) -> None:
    q_img = preprocess(Image.open(query_image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        q = model.encode_image(q_img)
        q = q / q.norm(dim=-1, keepdim=True)

    sims = torch.matmul(db_embeds_t, q.squeeze(0))
    topk = torch.topk(sims, k=min(top_k, db_embeds_t.shape[0]))
    idxs = topk.indices.detach().cpu().numpy().tolist()

    cols = 5
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_i, ax in enumerate(axes):
        ax.axis("off")
        if ax_i >= len(idxs):
            continue
        idx = idxs[ax_i]
        img_path = str(db_paths[idx])
        ax.imshow(Image.open(img_path).convert("RGB"))
        ax.set_title(Path(img_path).name, fontsize=9)

    fig.suptitle(f"Top-{len(idxs)} results for query: {Path(query_image).name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[clip_retrieval_and_evaluation] device={device}")

    db_embeds = np.load(emb_dir / "clip_embeddings.npy").astype(np.float32)
    db_paths = np.load(emb_dir / "clip_paths.npy", allow_pickle=True)

    # Labels inferred from path
    str_labels = np.array([infer_label_from_path(str(p)) for p in db_paths], dtype=object)
    uniq = sorted(set(str_labels.tolist()))
    label_to_idx = {lbl: i for i, lbl in enumerate(uniq)}
    y = np.array([label_to_idx[lbl] for lbl in str_labels], dtype=np.int64)

    # Qualitative retrieval (optional)
    if args.query_image:
        model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
        model = model.to(device).eval()
        db_embeds_t = torch.tensor(db_embeds, device=device, dtype=torch.float32)

        out_path = out_dir / "query_topk.png"
        save_query_topk(model, preprocess, device, args.query_image, db_embeds_t, db_paths, out_path, args.top_k)
        print(f"[clip_retrieval_and_evaluation] saved -> {out_path}")

    # Recall@K
    recalls = compute_recall_at_k(db_embeds, y, k_list=(1, 5, 10))
    df = pd.DataFrame([recalls])
    df.to_csv(out_dir / "clip_recall_scores.csv", index=False)
    print(f"[clip_retrieval_and_evaluation] saved -> {out_dir / 'clip_recall_scores.csv'}")

    # Plot recall@K
    ks = list(recalls.keys())
    vals = [recalls[k] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.bar(ks, vals)
    plt.ylim(0, 1)
    plt.title("CLIP Recall@K")
    plt.ylabel("Recall")
    plt.savefig(out_dir / "clip_recall_plot.png", dpi=200)
    plt.close()
    print(f"[clip_retrieval_and_evaluation] saved -> {out_dir / 'clip_recall_plot.png'}")

    # PCA + t-SNE
    X = np.nan_to_num(db_embeds, nan=0.0, posinf=0.0, neginf=0.0)
    if args.tsne_max_points and X.shape[0] > args.tsne_max_points:
        # deterministic subsample for reproducibility
        idx = np.linspace(0, X.shape[0] - 1, args.tsne_max_points).astype(int)
        X_vis = X[idx]
        y_vis = y[idx]
    else:
        X_vis = X
        y_vis = y

    print("[clip_retrieval_and_evaluation] running PCA(50) + t-SNE(2)...")
    pca = PCA(n_components=min(50, X_vis.shape[1])).fit_transform(X_vis)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    xy = tsne.fit_transform(pca)

    plt.figure(figsize=(10, 8))
    plt.scatter(xy[:, 0], xy[:, 1], c=y_vis, s=5, cmap="tab20")
    plt.title("t-SNE of CLIP Embeddings")
    plt.savefig(out_dir / "clip_tsne_plot.png", dpi=200)
    plt.close()
    print(f"[clip_retrieval_and_evaluation] saved -> {out_dir / 'clip_tsne_plot.png'}")

    print("[clip_retrieval_and_evaluation] done.")


if __name__ == "__main__":
    main()
