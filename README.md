# 🖼️ AI Image Quality Utilities

> 🇫🇷 Une version française de ce document est disponible dans [README_FR.md](README_FR.md).

This project brings together various tools to facilitate the evaluation of the aesthetic quality of images, organized around six complementary approaches:

- 📦 **Off-the-shelf tools**: direct use of well-known pre-trained models (NIMA, MUSIQ via PyIQA, CLIP via `aesthetic-predictor`, Qwen2.5-VL via Ollama) with no additional training required.
- 🎓 **Personalized learning**: fine-tuning (ViT, Random Forest) custom predictors tailored to one's own aesthetic preferences, trained from manually annotated images.
- 🏷️ **Direct manual annotation**: Gradio interface to rate each image individually (0 → 5), used to build the datasets required for personalized training.
- 🗳️ **Annotation via pairwise voting**: comparative voting on pairs of images (Condorcet, Elo) to reduce subjective bias and produce more consistent rankings than absolute scoring.
- 🔗 **A posteriori clustering**: automatic grouping of images either by visual similarity (pHash, duplicate detection) or by semantic proximity (ResNet50 + DBSCAN).
- 🗂️ **Library management**: CLI tools to physically sort and reorganize images into structured subfolders, based on prediction results or a CSV mapping file.

---

## 📂 Project Structure

The project is organized as self-contained, documented functional modules with clearly separated responsibilities:

```
ai-image-utils/
│
├── README.md                     # General installation and usage guide (this file)
│
├── pyproject.toml                # ⚙️ Global configuration and dependencies managed by UV
│
├── aesthetic_evaluator/          # 🔍 1. Quality Estimation & Analysis
│   ├── vit_aesthetic_evaluator.py      # CLI for tiered quality scoring (ViT)
│   ├── clip_aesthetic_evaluator.py     # Batch evaluation of large image sets (CLIP)
│   ├── musiq_aesthetic_evaluator.py    # Multi-scale evaluation (variable resolution images)
│   ├── nima_aesthetic_evaluator.py     # Aesthetic scoring and consensus (MobileNet)
│   └── vlm_aesthetic_evaluator.py      # Detailed textual critique via vision-language model (VLM)
│
├── training/                     # 🎓 2. Model Training & Evaluation
│   ├── train_quality_vit.py            # Supervised fine-tuning of Google Vision Transformer (ViT)
│   ├── train_quality_rf.py             # OpenCV descriptors + Random Forest regressor
│   ├── metrics_evaluator.py            # Confusion heatmap (Seaborn) & reports (Scikit-Learn)
│   └── example_vit_training.py         # Integration guide and full training example
│
├── gui/                          # 🖥️ 3. Dashboards & Manual Annotation
│   ├── quality_annotator.py            # Fast Gradio manual labelling (scores 0 to 5)
│   ├── ab_vote.py                      # Unified A/B comparative voting (Elo & Condorcet)
│   └── results_dashboard.py            # Unified Gradio dashboard (standard mode + HTML thumbnails)
│
├── utils/                        # 🛠️ 4. File Sorting & Organization
│   ├── reorganize_images_csv.py        # CSV-driven file organization (move/copy with unique renaming)
│   ├── sort_images_by_quality.py       # Copy images into quality-labelled subfolders
│   └── CopyTopNFiles.ps1               # PowerShell script to copy the top N files from a folder
│
├── clustering/                   # 🔗 5. Grouping Algorithms & Duplicate Detection
│   ├── cluster_by_duplicates.py        # Duplicate detection (fast CPU-based pHash)
│   └── cluster_by_semantics.py         # Semantic clustering (ResNet50 + DBSCAN)
│
└── tests/                        # 🧪 6. Integration Tests & Validation
    └── test_aesthetic_predictor.py     # Smoke test to validate the aesthetic predictor setup
```

> [!NOTE]
> Each subdirectory contains its own `README.md` with detailed descriptions and command-line usage for every script.

---

## ⚙️ Prerequisites & Dependencies (Managed by UV)

This project uses **`uv`**, the next-generation ultra-fast Python package manager. All dependencies are centralized at the root in [pyproject.toml](pyproject.toml).

### 🛠️ Quick setup:
1. Install `uv` if not already available:
   ```bash
   pip install uv  # or via the official installer
   ```
2. From the project root, synchronize the environment:
   ```bash
   uv sync
   ```
   This automatically creates a `.venv` folder and installs all required dependencies (TensorFlow, PyTorch, PyIQA, Transformers, Gradio, OpenCV, etc.).

---

## 🚀 Quick Start

All scripts can be run directly through the virtual environment using `uv run python`.

### 1. Manually annotate your image dataset
Launch the Gradio annotation interface to rate your images:
```bash
uv run python gui/quality_annotator.py "path/to/images"
```

### 2. Train a ViT classification model
Organize your images into quality subfolders (`quality_0` to `quality_5`) and launch the fine-tuning pipeline:
```bash
uv run python training/train_quality_vit.py
```

### 3. Analyze an image or folder (multi-model)
Several complementary models are available under `aesthetic_evaluator/`:

*   **Vision Transformer (ViT)** (CPU or GPU CUDA auto-detection):
    ```bash
    uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "path/to/image.jpg" -v
    uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "path/to/image.jpg" --cpu -v  # Force CPU mode
    ```
*   **NIMA (Neural Image Assessment)** (TensorFlow/MobileNet):
    ```bash
    uv run python aesthetic_evaluator/nima_aesthetic_evaluator.py "path/to/image_or_folder"
    ```
*   **MUSIQ metric (Multi-scale Image Quality)** (PyTorch/PyIQA):
    ```bash
    uv run python aesthetic_evaluator/musiq_aesthetic_evaluator.py "path/to/image.jpg"
    ```
*   **VLM multi-criteria analysis** (via local Ollama):
    ```bash
    uv run python aesthetic_evaluator/vlm_aesthetic_evaluator.py -i "path/to/image.jpg"
    ```

### 4. Run a pairwise voting session (A/B)
To finely rank image variants by comparing them two at a time via an interactive Gradio interface:
```bash
# Elo mode (default)
uv run python gui/ab_vote.py --method elo

# Condorcet mode
uv run python gui/ab_vote.py --method condorcet
```

---

## 📄 License
This project is distributed under the terms of the license found in the `LICENSE` file at the project root.
