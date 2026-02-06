# Open-Set Fine-Grained Image Retrieval using CLIP

This project explores **open-set fine-grained image retrieval**, where the system must retrieve visually similar images even when the query belongs to a **previously unseen class**.

We demonstrate that **pretrained vision–language models (CLIP)** dramatically outperform supervised CNN baselines for this task, while requiring far less training compute.

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

## 📄 Report
The full technical report with ablations, visualizations, and failure analysis is available here:

📎 `Group 1 ML project final report.pdf`

---

## 🔮 Future Work
- Scaling FAISS indexing to millions of images
- CLIP distillation for edge deployment
- Web-based interactive search demo
- Domain-specific CLIP adaptation
