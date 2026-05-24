# 🛠️ File Sorting & Organization Utilities (`utils/`)

> 🇫🇷 Une version française de ce document est disponible dans [README_FR.md](README_FR.md).

This module contains utility scripts for managing, sorting, and physically organizing image files on disk, based on annotation or prediction files in CSV format.

---

## 📁 Script Descriptions

### 1. `reorganize_images_csv.py` (Generic CSV-driven Organization)
* **Description**: Moves (or copies) images scattered across the disk into specific subfolders based on the `path` (source path) and `dir` (destination folder name) columns of a mapping CSV file.
* **Features**:
  * **Conflict resolution**: If a file with the same name already exists in the target directory, the script automatically appends an incremental counter (e.g., `photo_1.jpg`) to avoid overwriting data.
  * **Copy option**: The `--copy` flag dynamically patches the move function with a copy (`shutil.copy2`) that preserves file metadata.
* **Usage**:
  ```bash
  python utils/reorganize_images_csv.py "path/to/mapping.csv" "path/to/target_folder" --copy
  ```

### 2. `sort_images_by_quality.py` (Sort by Quality Score)
* **Description**: Specialized script that reads a CSV file containing quality predictions (with `image_path` and `predicted_quality` columns) and automatically places physical images into 6 distinct structured folders, from `0_VeryPoor` to `5_Excellent`.
* **Features**:
  * Automatic creation of target quality directories.
  * Uses `shutil.copy2` to preserve original file timestamps and metadata.
* **Usage**:
  ```bash
  python utils/sort_images_by_quality.py "path/to/predictions.csv" "path/to/output_directory"
  ```

### 3. `CopyTopNFiles.ps1` (Selective Image Copy — PowerShell)
* **Description**: PowerShell utility script for quickly extracting and copying the top $N$ files (sorted by name or relevance) from a source folder to a destination folder, with automatic directory creation and optional overwrite handling.
* **Usage**:
  ```powershell
  .\utils\CopyTopNFiles.ps1 -SourcePath "path/to/source" -DestinationPath "path/to/destination" -NumberOfFiles 10 -Overwrite
  ```
