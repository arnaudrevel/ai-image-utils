# manual-annotation Specification

## Purpose
Enable a user to manually label images with absolute aesthetic quality scores (0 to 5)
through an ergonomic local Gradio web interface, producing a structured CSV dataset
suitable for model training.

---

## Requirements

### Requirement: Image Discovery and Loading
The system SHALL scan a given directory for supported image files (jpg, jpeg, png, bmp, tiff, webp)
and present them one by one for annotation.

#### Scenario: Recursive scan
- **GIVEN** a directory containing images in nested subdirectories and `--recursive` is set
- **WHEN** `quality_annotator.py` is launched
- **THEN** all images across all subdirectories are discovered and queued for annotation

#### Scenario: Empty directory
- **GIVEN** the target directory contains no supported image files
- **WHEN** the script is launched
- **THEN** a warning is displayed and the Gradio server does not start

---

### Requirement: Session Resume
The system SHALL automatically skip images that were already annotated in a previous session,
allowing incremental annotation across multiple sessions.

#### Scenario: Partial session resumed
- **GIVEN** `annotations.csv` exists with N previously annotated images
- **WHEN** the annotator is relaunched on the same directory
- **THEN** already-annotated images are skipped and the interface starts on the first unannotated image

---

### Requirement: Thread-safe Annotation Persistence
The system SHALL write each annotation to a CSV file immediately and atomically upon validation,
using a threading lock to prevent data corruption from concurrent Gradio requests.

#### Scenario: Annotation saved on validate
- **GIVEN** an image is displayed and the user selects a quality rating
- **WHEN** the "Valider" button is clicked
- **THEN** a new CSV row is appended with timestamp, image path, label (int), and class name
  without corrupting existing data

---

### Requirement: Navigation Controls
The system SHALL provide Previous, Skip, and Validate buttons to allow flexible navigation
through the image queue.

#### Scenario: Skip without annotating
- **GIVEN** an image is displayed
- **WHEN** the "Ignorer" button is clicked
- **THEN** the image is skipped without writing to CSV and the next unannotated image is shown

#### Scenario: Go back to previous image
- **GIVEN** the user has moved past an image
- **WHEN** the "Retour" button is clicked
- **THEN** the previous image in the queue is displayed (annotation is not re-written)
