"""Evaluate embeddings (PCA/t-SNE + Recall@K).

Reads embeddings produced by:
  - evaluation/extract_embeddings.py  (CNN baseline)
  - clip/clip_extract_embeddings.py  (CLIP baseline)

Provide --embeddings, --labels, and optionally --paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval embeddings.")
    p.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .npy (N, D).")
    p.add_argument("--labels", type=str, required=True, help="Path to labels .npy (N,).")
    p.add_argument("--paths", type=str, default=None, help="Optional paths .npy for labeling plots.")
    p.add_argument("--out_dir", type=str, default="outputs/eval")
    p.add_argument("--tsne_max_points", type=int, default=5000)
    return p.parse_args()


def compute_recall_at_k(emb: np.ndarray, labels: np.ndarray, k_list=(1, 5, 10)) -> dict[str, float]:
    d = pairwise_distances(emb, emb, metric="cosine")
    np.fill_diagonal(d, np.inf)
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.embeddings).astype(np.float32)
    y = np.load(args.labels).astype(np.int64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Recall@K
    recalls = compute_recall_at_k(X, y)
    pd.DataFrame([recalls]).to_csv(out_dir / "recall_scores.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(list(recalls.keys()), list(recalls.values()))
    plt.ylim(0, 1)
    plt.title("Recall@K")
    plt.ylabel("Recall")
    plt.savefig(out_dir / "recall_plot.png", dpi=200)
    plt.close()

    # Subsample for t-SNE if needed
    if args.tsne_max_points and X.shape[0] > args.tsne_max_points:
        idx = np.linspace(0, X.shape[0] - 1, args.tsne_max_points).astype(int)
        X_vis, y_vis = X[idx], y[idx]
    else:
        X_vis, y_vis = X, y

    # PCA(2)
    pca2 = PCA(n_components=2).fit_transform(X_vis)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca2[:, 0], pca2[:, 1], c=y_vis, s=5, cmap="tab20")
    plt.title("PCA (2D) of embeddings")
    plt.savefig(out_dir / "pca_plot.png", dpi=200)
    plt.close()

    # PCA(50) + t-SNE(2)
    pca50 = PCA(n_components=min(50, X_vis.shape[1])).fit_transform(X_vis)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    xy = tsne.fit_transform(pca50)
    plt.figure(figsize=(10, 8))
    plt.scatter(xy[:, 0], xy[:, 1], c=y_vis, s=5, cmap="tab20")
    plt.title("t-SNE of embeddings")
    plt.savefig(out_dir / "tsne_plot.png", dpi=200)
    plt.close()

    print(f"[evaluate_embeddings] outputs saved -> {out_dir}")


if __name__ == "__main__":
    main()
