# Project Context

## Purpose
AI Image Quality Utilities (`ai-image-utils`) is a Python toolkit for evaluating, annotating,
training, and managing the aesthetic quality of image libraries. It combines off-the-shelf
pre-trained models, personalized fine-tuning, pairwise comparative voting, automatic clustering,
and file organization utilities — all accessible via CLI scripts or local Gradio web interfaces.

## Tech Stack

### Machine Learning / Deep Learning
- **PyTorch** >= 2.2 — inference and training backbone for ViT, CLIP, MUSIQ
- **TensorFlow / Keras** >= 2.20 — NIMA model (MobileNet backbone)
- **Hugging Face Transformers** >= 4.40 — ViT model loading and fine-tuning (`Trainer`)
- **Hugging Face Datasets** >= 2.19 — dataset preparation for ViT training
- **PyIQA** >= 0.1.8 — multi-metric image quality assessment (MUSIQ, and others)
- **aesthetic-predictor** >= 0.1.2 — CLIP-based aesthetic scoring (Schuhmann)
- **scikit-learn** >= 1.4 — Random Forest regressor, DBSCAN clustering

### Computer Vision
- **OpenCV** >= 4.9 — low-level image descriptors (Laplacian, Sobel, entropy)
- **imagehash** >= 4.3 — perceptual hashing (pHash) for duplicate detection
- **torchvision** >= 0.17 — ResNet50 feature extraction
- **Pillow** >= 12.1 — universal image I/O

### Vision-Language Model (VLM)
- **LangChain-Ollama** — local inference with `qwen2.5vl:7b` via Ollama

### GUI
- **Gradio** >= 4.31 — local interactive web interfaces (annotation, voting, dashboard)

### Data & Visualization
- **Pandas** >= 2.2 — CSV result handling
- **Matplotlib / Seaborn** >= 3.8 / 0.13 — confusion matrix heatmaps
- **NumPy** >= 2.4

### CLI & Logging
- **Click** >= 8.1 — CLI definition for VLM evaluator
- **Loguru** >= 0.7 — structured logging

### Package Management
- **uv** — fast Python package manager; all deps declared in `pyproject.toml`

## Project Conventions

### Code Style
- Python >= 3.12 with type hints throughout
- Docstrings on all public functions and classes
- ASCII-safe console output (no Unicode in print statements)
- Thread-safe file writes using `threading.Lock` where needed

### Architecture Patterns
- **Script-based modular architecture** — no web framework, no REST API
- Each module directory is self-contained with its own CLI entry point and `README.md`
- Entry point convention: `if __name__ == "__main__": main()`
- Model artifacts stored under `models/` (not committed to git)
- Data (images, CSVs) stored under `data/inputs/` and `data/outputs/`

### Testing Strategy
- Smoke tests only (`tests/test_aesthetic_predictor.py`)
- Manual validation via Gradio dashboards

### Git Workflow
- No CI/CD configuration detected
- Large binary files (images, model weights, CSVs, JSON) excluded via `.gitignore`

## Important Constraints
- **TensorFlow and PyTorch must not be initialized in the same process** to avoid VRAM conflicts
  — NIMA (TF) and ViT/CLIP/MUSIQ (PyTorch) are strictly separated into distinct scripts
- **VLM requires Ollama running locally** with the `qwen2.5vl:7b` model pulled
- **NIMA requires `models/nima_weights/mobilenet_weights.h5`** — not included in the repo
- **ViT inference uses model weights from `models/aesthetic-classifier/`** — not included in the repo
- GPU CUDA is auto-detected; all PyTorch scripts fall back gracefully to CPU
