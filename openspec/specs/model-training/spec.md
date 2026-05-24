# model-training Specification

## Purpose
Fine-tune pre-trained image quality models on manually annotated datasets to produce
personalized aesthetic predictors aligned with the user's own preferences. Supports both
modern deep learning (ViT) and classical computer vision (Random Forest) approaches.

---

## Requirements

### Requirement: ViT Fine-tuning Pipeline
The system SHALL fine-tune a pre-trained Vision Transformer (ViT) model from Hugging Face
on a locally organized dataset of images sorted into quality subfolders (`quality_0` to `quality_5`).

#### Scenario: Successful ViT training
- **GIVEN** images are organized under `quality_0/` through `quality_5/` subfolders
- **WHEN** `train_quality_vit.py` is executed
- **THEN** the dataset is split 70/30, the model is fine-tuned for 6-class classification,
  and weights are saved to `models/aesthetic-classifier/`

#### Scenario: Empty or missing quality folder
- **GIVEN** one or more `quality_N/` subfolders are empty or missing
- **WHEN** the training script is launched
- **THEN** the script reports the missing class and exits with a clear error

---

### Requirement: Random Forest Regressor Training
The system SHALL train a Random Forest regressor on OpenCV-extracted image descriptors
(Laplacian sharpness, contrast std-dev, mean brightness, entropy, Sobel magnitude)
to predict a continuous aesthetic quality score.

#### Scenario: Successful Random Forest training
- **GIVEN** a labeled image dataset accessible from disk
- **WHEN** `train_quality_rf.py` is executed
- **THEN** descriptors are extracted, the dataset is split 80/20, the regressor is trained,
  and the model is saved as `quality_model.pkl`

#### Scenario: Model persistence
- **GIVEN** training completes successfully
- **WHEN** the pickle file is loaded in a new process
- **THEN** the model produces consistent predictions without retraining

---

### Requirement: Performance Metrics Evaluation
The system SHALL compute and display a confusion matrix heatmap and a full classification report
(precision, recall, F1-score per class) for a trained model evaluated on a test set.

#### Scenario: Metrics visualization
- **GIVEN** a trained model and a labeled test set
- **WHEN** `metrics_evaluator.py` is executed
- **THEN** a Seaborn heatmap is displayed and the Scikit-Learn classification report is printed

---

### Requirement: Integration Example
The system SHALL provide a self-contained example script demonstrating how to load, train,
and run inference with the ViT classifier using the project's module structure.

#### Scenario: Example runs end-to-end
- **GIVEN** image data is present in the expected directory structure
- **WHEN** `example_vit_training.py` is executed directly
- **THEN** the full pipeline (load → train → predict) completes without import errors
