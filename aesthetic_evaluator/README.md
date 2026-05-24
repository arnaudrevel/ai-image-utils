# 🔍 Aesthetic Estimation Module (`aesthetic_evaluator/`)

> 🇫🇷 Une version française de ce document est disponible dans [README_FR.md](README_FR.md).

This module gathers all tools for the automatic analysis and estimation of the aesthetic or technical quality of images. All scripts follow a uniform naming convention based on their **underlying technology**: **`<technology>_aesthetic_evaluator.py`**.

---

## 📁 Tool Descriptions

### 1. `vit_aesthetic_evaluator.py` (Vision Transformer — Main CLI)
* **Description**: Highly optimized command-line client for evaluating the aesthetic quality of a single image or an entire folder using a **ViT (Vision Transformer)** model from Hugging Face. Classifies images into one of 6 quality tiers.
* **Features**:
  * Automatic CUDA GPU detection with half-precision mode (FP16 / Inference Mode).
  * `--cpu` option to force CPU execution.
  * Full export of scores and per-class probabilities to CSV or JSON.
* **Usage**:
  ```bash
  uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "path/to/image_or_folder" -o "results.csv" -v
  uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "path/to/image_or_folder" --cpu -o "results.json"
  ```

### 2. `clip_aesthetic_evaluator.py` (CLIP — Batch Inference)
* **Description**: Scans a directory for images and estimates their raw aesthetic score (continuous value from 1 to 10) using the **CLIP**-based aesthetic predictor by Christoph Schuhmann. Processes images asynchronously in batches to manage RAM usage.
* **Usage**:
  ```bash
  uv run python aesthetic_evaluator/clip_aesthetic_evaluator.py --root-dir "path/to/images" --output "scores.html"
  ```

### 3. `musiq_aesthetic_evaluator.py` (MUSIQ — Variable Resolution)
* **Description**: Uses the **MUSIQ** multi-scale image quality model via the PyIQA library on PyTorch. Particularly robust for evaluating images at variable resolutions while preserving multi-scale details. Supports GPU CUDA acceleration.
* **Usage**:
  ```bash
  uv run python aesthetic_evaluator/musiq_aesthetic_evaluator.py "path/to/image.jpg"
  ```

### 4. `nima_aesthetic_evaluator.py` (NIMA — Aesthetic Consensus)
* **Description**: Implements the **NIMA** (Neural Image Assessment) model with a MobileNet backbone under TensorFlow/Keras. Predicts the probability distribution of image ratings across 10 classes, computes the mean aesthetic score and consensus standard deviation, and produces a summary table. Supports batch and recursive evaluation.
* **Usage**:
  ```bash
  uv run python aesthetic_evaluator/nima_aesthetic_evaluator.py "path/to/image_or_folder"
  ```

### 5. `vlm_aesthetic_evaluator.py` (VLM — Multi-criteria Critique)
* **Description**: Leverages a local vision-language model (**VLM** `qwen2.5vl:7b` or similar via Ollama) to generate a rich, structured, and argued textual critique across 5 aesthetic dimensions (Composition, Colors/Lighting, Textures, Emotion, Technique) with a final aesthetic score out of 10.
* **Usage**:
  ```bash
  uv run python aesthetic_evaluator/vlm_aesthetic_evaluator.py -i "image.jpg"
  ```
