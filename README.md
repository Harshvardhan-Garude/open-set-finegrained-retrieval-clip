# Open-Set Fine-Grained Image Retrieval using CLIP

This project explores **open-set fine-grained image retrieval**, where the system must retrieve visually similar images even when the query belongs to a **previously unseen class**.

The project report compares **pretrained vision-language models (CLIP)** with supervised CNN baselines. Its reported results are summarized below; reproducing them requires the original dataset splits and experiment inputs.

The report is co-authored by **Justin Wang, Sudip Aryal, and Harshvardhan Garude**.

## Run image and text search

From the repository root, create a virtual environment with `python -m venv .venv`.
Activate it with `source .venv/bin/activate` on Linux or macOS, or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Install the dependencies
with `python -m pip install -r requirements.txt`.

Point the embedding script at a local image directory with one subfolder per
class. The script writes embeddings and image paths to `outputs/clip/` by default.

```bash
python clip/clip_extract_embeddings.py --data_dir path/to/image_dataset
python clip/clip_text_search.py --prompt "red pickup with roof rack"
python clip/clip_retrieval.py --query_image path/to/query.jpg
```

Keep the indexed images at their original paths so the search scripts can display
them. The quickstart uses frozen OpenCLIP embeddings and exact PyTorch dot-product
search. It does not load a learned projection head or build a FAISS index; those
belong to the reported experimental design. The first model run downloads the
OpenAI-pretrained weights unless they are already cached. The [project report](report/final_report.pdf) describes the dataset splits
and evaluation used for the results below.

---

## 🔍 Problem Setting
Fine-grained retrieval systems (e.g., vehicle make/model, trim, color) traditionally assume a **closed set** of classes during training.

In real-world deployments:
- New categories appear continuously
- Fine labels are incomplete or noisy
- Systems must generalize beyond training taxonomies

This project formulates retrieval as an **open-set problem** and evaluates whether vision–language pretraining can solve it effectively.

---

## 🚀 Key Contributions
- Built a **hierarchical retrieval pipeline** that prioritizes coarse object similarity before fine-grained attributes
- Replaced supervised CNN backbones with **frozen CLIP ViT-B/32 embeddings**
- Enabled **zero-shot text-to-image retrieval** (e.g., “red pickup with roof rack”)
- Achieved **98.1% Recall@10**, outperforming fine-tuned CNN baselines by a large margin
- Reduced training compute by **>90%** compared to CNN-based approaches

---

## 🧠 Method Overview
### Baselines
- ResNet-50 and EfficientNet-B0
- Trained with **ArcFace loss + hierarchical triplet loss**
- Class-balanced hierarchical sampling

### CLIP-Based Retrieval
- Frozen CLIP ViT-B/32 image encoder
- Lightweight projection head (MLP)
- Shared embedding space for image and text queries
- FAISS-based ANN indexing for fast retrieval

---

## 📊 Results
| Model | Recall@1 | Recall@5 | Recall@10 |
|------|---------|----------|-----------|
| EfficientNet-B0 | 0.196 | 0.586 | 0.768 |
| CLIP ViT-B/32 (frozen) | **0.882** | **0.967** | **0.981** |

CLIP achieves **4.5× higher Recall@1** while training **10× faster**.

---

## 🖼 Qualitative Analysis
- t-SNE visualizations show tighter clusters and better separation for unseen classes
- CNN baselines struggle with overlapping fine-grained clusters
- CLIP generalizes well across domains (vehicles, animals, everyday objects)

---

## 🧪 Datasets
- Stanford Cars
- Vehicle-10
- OpenImages (vehicle subset)

Datasets are merged into a unified **coarse → fine hierarchy** with explicit unseen-class splits for evaluation.

---

## 🛠 Tech Stack
- PyTorch
- OpenCLIP
- FAISS
- NumPy, Matplotlib
- scikit-learn

---

## 📌 Notes
This repository focuses on **research clarity and system design** rather than exhaustive hyperparameter tuning.  
All experiments were run on **single-GPU / consumer hardware**, emphasizing practicality.

---

## 🔮 Future Work
- Scaling FAISS indexing to millions of images
- CLIP distillation for edge deployment
- Web-based interactive search demo
- Domain-specific CLIP adaptation
