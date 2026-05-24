# 🧪 Integration Tests & Validation Module (`tests/`)

> 🇫🇷 Une version française de ce document est disponible dans [README_FR.md](README_FR.md).

This module contains quick smoke tests and basic integration scripts used to validate the functional behavior of your evaluation software dependencies.

---

## 📁 Script Descriptions

### 1. `test_aesthetic_predictor.py` (Integration Smoke Test)
* **Description**: Minimal script designed to ensure that the aesthetic prediction library (`aesthetic_predictor`) is correctly installed on your machine and interfaces properly with Pillow to load a JPEG test image.
* **How it works**:
  * Loads a demonstration image from `data/inputs/labeled_tiers/quality_5/`.
  * Calls the `predict_aesthetic` function to evaluate the image.
  * Prints the raw aesthetic score obtained.
* **Usage**:
  ```bash
  uv run python tests/test_aesthetic_predictor.py
  ```
