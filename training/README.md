# 🎓 Model Training & Evaluation Module (`training/`)

> 🇫🇷 Une version française de ce document est disponible dans [README_FR.md](README_FR.md).

This module hosts the tools required to train (fine-tune) pre-trained image quality models, whether modern deep learning architectures (Vision Transformers) or classical computer vision approaches (regression on OpenCV descriptors). It also provides functions for computing and visualizing performance metrics.

---

## 📁 Script Descriptions

### 1. `train_quality_vit.py` (ViT Fine-tuning)
* **Description**: Fine-tunes a Vision Transformer (ViT by Google) on your own dataset. Handles reading from sorted folders (`quality_0` to `quality_5`), preprocessing image tensors to $224 \times 224$ pixels, and uses the Hugging Face `Trainer` package to optimize the learning process.
* **How it works**:
  * Indexes image folders.
  * Prepares the dataset and splits it into $70\%$ training and $30\%$ validation.
  * Trains a 6-class classification model.

### 2. `train_quality_rf.py` (OpenCV Descriptors + Random Forest)
* **Description**: Classical computer vision pipeline. Extracts mathematical descriptors from pixel intensities (Laplacian variance for sharpness, standard deviation for contrast, mean for exposure, visual entropy, and Sobel magnitude) and trains a Random Forest regressor with Scikit-Learn.
* **How it works**:
  * Computes OpenCV image features.
  * Splits the dataset (80/20) and trains the `RandomForestRegressor`.
  * Saves the model as a Pickle file (`quality_model.pkl`).

### 3. `metrics_evaluator.py` (Performance Calculator)
* **Description**: Diagnostic evaluation script. Generates a detailed classification report (precision, recall, F1-score per rating) and dynamically displays a confusion matrix heatmap using Seaborn and Matplotlib to measure model deviations.

### 4. `example_vit_training.py` (Integration Example Script)
* **Description**: Practical and self-contained implementation guide. Shows how to structure your code to load the ViT classifier, prepare data from disk, train the model, and predict the quality of test images. *(Relative imports have been corrected to make it runnable from the first launch)*.
