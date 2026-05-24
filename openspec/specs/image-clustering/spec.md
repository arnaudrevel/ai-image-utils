# image-clustering Specification

## Purpose
Automatically group images from a source folder into clusters based on either visual similarity
(perceptual hashing for duplicate/near-duplicate detection) or semantic proximity
(deep feature extraction + density-based clustering).

---

## Requirements

### Requirement: Duplicate Detection via pHash
The system SHALL detect duplicate or near-duplicate images using perceptual hashing (pHash)
and group them into named cluster folders in a destination directory.

#### Scenario: Duplicates found and grouped
- **GIVEN** a source folder containing images with some near-identical copies
- **WHEN** `cluster_by_duplicates.py` is invoked with a Hamming distance threshold
- **THEN** groups of similar images are copied into `groupe_001/`, `groupe_002/`, etc.
  in the destination folder

#### Scenario: No duplicates found
- **GIVEN** all images are visually distinct (Hamming distances exceed the threshold)
- **WHEN** `cluster_by_duplicates.py` is invoked
- **THEN** no output folders are created and a summary is printed indicating zero groups found

#### Scenario: Corrupted or unreadable image
- **GIVEN** a source folder contains a corrupted image file
- **WHEN** pHash computation is attempted on it
- **THEN** the image is skipped with an error log and processing continues for the remaining images

---

### Requirement: Semantic Clustering via ResNet50 + DBSCAN
The system SHALL extract deep semantic feature vectors from images using ResNet50 and group
them using DBSCAN clustering with cosine similarity, placing noise (unique/isolated) images
in a separate folder.

#### Scenario: Semantic clusters formed
- **GIVEN** a source folder of images sharing some common visual themes or subjects
- **WHEN** `cluster_by_semantics.py` is invoked with `--eps` and `--min_samples`
- **THEN** images are grouped into `cluster_0/`, `cluster_1/`, etc. based on semantic proximity

#### Scenario: Noise images isolated
- **GIVEN** some images are semantically unique and do not belong to any cluster
- **WHEN** DBSCAN assigns them label -1
- **THEN** those images are placed in a `noise/` folder in the destination directory

#### Scenario: GPU acceleration
- **GIVEN** a CUDA-capable GPU is available
- **WHEN** `cluster_by_semantics.py` is launched
- **THEN** ResNet50 feature extraction runs on GPU, reducing processing time for large batches
