# aesthetic-evaluation Specification

## Purpose
Estimate the aesthetic quality of images using a selection of pre-trained or fine-tuned models,
each based on a distinct technology. The evaluator produces quality scores or critiques that
can be exported for downstream use (sorting, training, dashboards).

---

## Requirements

### Requirement: ViT-based Quality Classification
The system SHALL evaluate the aesthetic quality of one image or an entire folder using a
fine-tuned Vision Transformer (ViT) model, classifying each image into one of 6 discrete
quality tiers (0 = Very Poor → 5 = Excellent).

#### Scenario: Single image evaluation with GPU
- **GIVEN** a valid image path and a CUDA-capable GPU is available
- **WHEN** `vit_aesthetic_evaluator.py` is invoked
- **THEN** the model loads in FP16 mode on the GPU and returns the predicted quality tier,
  confidence score, and per-class probability distribution

#### Scenario: Folder evaluation with CSV export
- **GIVEN** a folder containing images and an output CSV path
- **WHEN** `vit_aesthetic_evaluator.py` is invoked with `--recursive -o results.csv`
- **THEN** all images are evaluated and results are written to the CSV with columns:
  `image_path`, `predicted_quality`, `quality_label`, `confidence`, `prob_quality_0..5`

#### Scenario: CPU fallback
- **GIVEN** no CUDA GPU is available or `--cpu` flag is set
- **WHEN** the model is loaded
- **THEN** inference runs on CPU in FP32 mode without error

#### Scenario: Missing model weights
- **GIVEN** `models/aesthetic-classifier/` does not exist
- **WHEN** the script is launched
- **THEN** the script exits with a clear error message indicating the missing model path

---

### Requirement: CLIP-based Batch Aesthetic Scoring
The system SHALL evaluate large batches of PNG images using the CLIP-based `aesthetic-predictor`
model, processing them in configurable batch sizes to manage RAM usage.

#### Scenario: Successful batch scoring
- **GIVEN** a directory containing PNG images
- **WHEN** `clip_aesthetic_evaluator.py` is invoked with `--root-dir`
- **THEN** images are processed in batches of 100, per-batch statistics are printed,
  and global statistics (min, max, mean, std) are displayed at the end

#### Scenario: Corrupted image in batch
- **GIVEN** a batch containing one unreadable image file
- **WHEN** the batch is processed
- **THEN** the error is logged, the image is skipped, and processing continues

---

### Requirement: MUSIQ Multi-scale Quality Scoring
The system SHALL evaluate the aesthetic quality of a single image using the MUSIQ model
via PyIQA, with automatic GPU/CPU selection.

#### Scenario: Successful MUSIQ scoring
- **GIVEN** a valid image path
- **WHEN** `musiq_aesthetic_evaluator.py` is invoked
- **THEN** a single continuous quality score is printed with 4 decimal precision

#### Scenario: Image not found
- **GIVEN** the provided image path does not exist on disk
- **WHEN** the script is launched
- **THEN** a clear error message is displayed and the script exits gracefully

---

### Requirement: NIMA Aesthetic Consensus Scoring
The system SHALL evaluate the aesthetic quality of one image or a folder using the NIMA model
(MobileNet backbone, TensorFlow), predicting a probability distribution over 10 rating levels
and computing the mean score and standard deviation.

#### Scenario: Single image NIMA evaluation
- **GIVEN** a valid image path and `models/nima_weights/mobilenet_weights.h5` exists
- **WHEN** `nima_aesthetic_evaluator.py` is invoked
- **THEN** the mean aesthetic score (1.0–10.0) and standard deviation are printed

#### Scenario: Missing NIMA weights
- **GIVEN** `models/nima_weights/mobilenet_weights.h5` does not exist
- **WHEN** the script is launched
- **THEN** a clear error message indicates the expected file path

---

### Requirement: VLM Multi-criteria Textual Critique
The system SHALL generate a detailed, structured aesthetic critique of an image using a local
vision-language model (Qwen2.5-VL 7B via Ollama), covering 5 aesthetic dimensions with a
final score out of 10.

#### Scenario: Successful VLM critique
- **GIVEN** a valid image path and Ollama is running locally with `qwen2.5vl:7b` pulled
- **WHEN** `vlm_aesthetic_evaluator.py -i image.jpg` is invoked
- **THEN** a structured critique is printed covering Composition, Colors/Lighting,
  Textures, Emotion, and Technique, with a global score /10

#### Scenario: Ollama unavailable
- **GIVEN** Ollama is not running or the model is not pulled
- **WHEN** the script is launched
- **THEN** a descriptive connection error is raised and logged via Loguru
