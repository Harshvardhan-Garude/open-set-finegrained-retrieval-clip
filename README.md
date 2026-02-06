# Open-Set Fine-Grained Image Retrieval using CLIP

This repository contains the code for our course project on **open-set fine-grained image retrieval**.
The goal is to retrieve visually similar images even when the query belongs to a **previously unseen class**.

The core idea: **frozen CLIP embeddings (ViT-B/32)** provide strong transfer for open-set retrieval, and enable
both **image-to-image** and **text-to-image** search in a shared embedding space.

## Repository structure

- `clip/` — CLIP embedding extraction + retrieval + evaluation
- `models/` — CNN baseline (EfficientNet-B0 embedding model) + training utilities
- `evaluation/` — generic embedding evaluation (Recall@K, PCA/t-SNE, top-K visualization)
- `data/` — dataset merge utilities and triplet dataloader helper
- `report/` — final report PDF

## Quickstart (CLIP)

1) **Extract embeddings**
```bash
python clip/clip_extract_embeddings.py --data_dir data/merged_dataset --out_dir outputs/clip
```

2) **Image → Image retrieval**
```bash
python clip/clip_retrieval.py --query_image path/to/query.jpg --emb_dir outputs/clip --out_dir outputs/clip
```

3) **Text → Image retrieval**
```bash
python clip/clip_text_search.py --prompt "red pickup truck" --emb_dir outputs/clip --out_dir outputs/clip
```

4) **Evaluation dashboard (Recall@K + t-SNE)**
```bash
python clip/clip_retrieval_and_evaluation.py --emb_dir outputs/clip --out_dir outputs/clip --query_image path/to/query.jpg
```

## Notes

- Datasets are not included in this repository.
- Outputs (plots, embeddings, checkpoints) are excluded via `.gitignore`.
- For details on datasets, splits, and results, see `report/final_report.pdf`.
