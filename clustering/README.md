# 🔗 Grouping Algorithms & Duplicate Detection Module (`clustering/`)

> 🇫🇷 Une version française de ce document est disponible dans [README_FR.md](README_FR.md).

This module is dedicated to automatic visual clustering and the detection of duplicates or semantic variants within image libraries. It offers two distinct approaches suited to different use cases:

---

## 📁 Tool Descriptions

### 1. `cluster_by_duplicates.py` (Duplicate & Near-Duplicate Detection)
* **Description**: Detects exact or near-exact copies using the **pHash** (Perceptual Hash) fingerprint based on the Discrete Cosine Transform (DCT). Images with a Hamming distance (number of differing bits) below the chosen threshold are grouped together and placed in `groupe_00X/` folders.
* **Features**:
  * **Ultra-lightweight**: No heavy neural network involved — runs in milliseconds on CPU only.
  * **Use case**: Ideal for cleaning a dataset by finding strict duplicates, resized images, JPG-recompressed variants, or images with minor color filters applied.
* **Usage**:
  ```bash
  uv run python clustering/cluster_by_duplicates.py "path/to/source_folder" "path/to/destination_folder" --threshold 5
  ```

### 2. `cluster_by_semantics.py` (Deep Semantic Clustering)
* **Description**: An unsupervised transfer learning approach. The script extracts 2048-dimensional deep feature vectors using the **ResNet50** vision network (pre-trained on ImageNet), then applies Scikit-Learn's density-based **DBSCAN** algorithm with cosine similarity to group images.
* **Features**:
  * **Intelligent & Semantic**: Groups images sharing the same concepts, spatial compositions, or subjects — even when pixels and resolution differ entirely.
  * **Dynamic cluster count**: DBSCAN automatically determines the optimal number of clusters.
  * **Noise isolation**: Images deemed unique or isolated are not forced into a group and are placed separately in a `noise/` folder.
* **Usage**:
  ```bash
  uv run python clustering/cluster_by_semantics.py "path/to/source" "path/to/destination" --eps 0.5 --min_samples 2
  ```
