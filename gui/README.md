# 🖼️ Gradio GUI Module (`gui/`)

> 🇫🇷 Une version française de ce document est disponible dans [README_FR.md](README_FR.md).

This module gathers the local interactive web interfaces built with the Gradio library. They allow for ergonomic manual image annotation, pairwise comparative ranking, or visual and statistical exploration of model predictions.

---

## 📁 Interface Descriptions

### 1. `quality_annotator.py` (Absolute Manual Annotator)
* **Description**: Simplified web interface allowing the user to review a batch of images, assign each a quality rating from 0 (Very Poor) to 5 (Excellent), and save annotations incrementally to a CSV file.
* **Features**:
  * Thread-safe concurrent writes (using a threading lock).
  * Reactive navigation (Previous, Next/Skip buttons).
  * Automatic resume: skips images already annotated in a previous session.
* **Usage**:
  ```bash
  uv run python gui/quality_annotator.py "path/to/images" --recursive
  ```

### 2. `ab_vote.py` (Unified A/B Comparative Voting)
* **Description**: Subjective evaluation interface based on pairwise comparisons. Users vote interactively in random duels between two images, enabling a highly accurate global aesthetic ranking by reducing subjective bias.
* **Available methods**:
  * **Elo system** (`--method elo`): Scores evolve non-linearly (factor $K=32$) based on strength gap and win probability. At the end of the session, raw scores are linearly normalized from $0.0$ to $5.0$.
  * **Condorcet voting** (`--method condorcet`): Scores evolve linearly in fixed steps ($+0.1$/$-0.1$), bounded between $0$ and $5$. Images reaching the minimum ($0$) or maximum ($5$) score are permanently ranked and excluded from future draws to focus attention on undecided cases.
* **Features**:
  * Automatic loading of input images from `data/inputs/pairwise_voting/` by default.
  * Automatic session save and resume via `annotation_state.json`.
  * Export of final results and history to CSV at `data/outputs/predictions/pairwise_rankings.csv`.
* **Usage**:
  ```bash
  # Default Elo mode
  uv run python gui/ab_vote.py --method elo

  # Condorcet mode
  uv run python gui/ab_vote.py --method condorcet
  ```

### 3. `results_dashboard.py` (Unified Dashboard)
* **Description**: Visualization dashboard for loading a CSV results file and interactively exploring aesthetic scores. Offers two display modes switchable via an in-interface toggle:
  * **Standard mode** (default): displays all CSV columns with readable formatting (percentages, success/failure emojis).
  * **Thumbnail mode**: reduces display to essential columns (`image_path`, `quality_label`) and injects image previews directly into the table via HTML `<img>` tags.
* **Features**:
  * Switch between modes without reloading the CSV file.
  * Click a row to display the image in full resolution in the viewer.
  * Statistical summary (total, successes, failures, average score).
* **Usage**:
  ```bash
  uv run python gui/results_dashboard.py
  ```
