# library-management Specification

## Purpose
Physically sort and reorganize image files on disk into structured subfolders, driven either
by aesthetic quality prediction results (CSV with predicted scores) or by a generic CSV mapping
file (source path → destination folder name).

---

## Requirements

### Requirement: Quality-based Image Sorting
The system SHALL read a prediction CSV file and copy each image into a quality-labelled subfolder
(`0_VeryPoor` to `5_Excellent`) under a specified output directory.

#### Scenario: Images sorted by predicted quality
- **GIVEN** a CSV file with `image_path` and `predicted_quality` columns
- **WHEN** `sort_images_by_quality.py` is invoked with the CSV and an output directory
- **THEN** each image is copied (via `shutil.copy2`) into the corresponding quality subfolder,
  preserving original file timestamps and metadata

#### Scenario: Missing source image
- **GIVEN** a CSV row references an image path that no longer exists on disk
- **WHEN** the sort script processes that row
- **THEN** a warning is printed for that file and processing continues for the remaining images

#### Scenario: Output directory auto-creation
- **GIVEN** the specified output directory does not exist
- **WHEN** the script starts
- **THEN** the directory and all six quality subfolders are created automatically

---

### Requirement: CSV-driven Generic File Organization
The system SHALL move or copy image files from arbitrary source paths to named destination
subfolders as specified by a mapping CSV with `path` and `dir` columns.

#### Scenario: Images moved according to mapping
- **GIVEN** a CSV mapping with valid `path` and `dir` columns
- **WHEN** `reorganize_images_csv.py` is invoked
- **THEN** each image is moved to `<target_directory>/<dir>/` as specified in the CSV

#### Scenario: Copy mode instead of move
- **GIVEN** the `--copy` flag is set
- **WHEN** the script processes files
- **THEN** images are copied (preserving metadata) instead of moved, leaving originals intact

#### Scenario: Filename conflict resolution
- **GIVEN** a file with the same name already exists in the destination subfolder
- **WHEN** the script attempts to place the image there
- **THEN** the incoming file is automatically renamed with an incremental suffix (e.g., `photo_1.jpg`)
  without overwriting existing data

---

### Requirement: Top-N File Extraction (PowerShell)
The system SHALL provide a PowerShell utility to copy the top N files (sorted by name) from a
source folder to a destination folder, with optional overwrite support.

#### Scenario: Top N files copied
- **GIVEN** a source directory with more than N files
- **WHEN** `CopyTopNFiles.ps1` is run with `-NumberOfFiles N`
- **THEN** exactly N files are copied to the destination in sorted order
