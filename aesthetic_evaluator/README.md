# 🔍 Module d'Estimation Esthétique (`aesthetic_evaluator/`)

Ce module regroupe l'ensemble des outils d'analyse automatique et d'estimation de la qualité esthétique ou technique d'images. Tous les scripts adoptent un style de nommage uniforme basé sur leur **technologie sous-jacente** sous la forme : **`<technologie>_aesthetic_evaluator.py`**.

---

## 📁 Description des Outils

### 1. `vit_aesthetic_evaluator.py` (Vision Transformer - CLI Principal)
* **Description** : Client en ligne de commande hautement optimisé pour évaluer la qualité esthétique d'une image ou d'un dossier complet à l'aide d'un modèle **ViT (Vision Transformer)** de Hugging Face. Il classe les images dans l'un des 6 paliers de qualité en français (Très mauvaise à Excellente).
* **Fonctionnalités** :
  * Détection automatique de GPU CUDA et mode demi-précision (FP16/Inference Mode).
  * Option `--cpu` pour forcer l'exécution sur processeur.
  * Exportation complète des scores et des probabilités par classes aux formats CSV ou JSON.
* **Lancement** :
  ```bash
  uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "chemin/vers/image_ou_dossier" -o "resultats.csv" -v
  uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "chemin/vers/image_ou_dossier" --cpu -o "resultats.json"
  ```

### 2. `clip_aesthetic_evaluator.py` (CLIP - Inférence par Lots)
* **Description** : Parcourt un répertoire à la recherche d'images et estime leur score esthétique brut (note continue de 1 à 10) à l'aide du modèle **CLIP** d'évaluation esthétique de Christoph Schuhmann. Il effectue un traitement asynchrone par lots (batches) pour réguler l'usage de la mémoire RAM.
* **Lancement** :
  ```bash
  uv run python aesthetic_evaluator/clip_aesthetic_evaluator.py --root-dir "chemin/vers/images" --output "scores.html"
  ```

### 3. `musiq_aesthetic_evaluator.py` (MUSIQ - Résolution Variable)
* **Description** : Utilise le modèle d'évaluation de qualité multi-échelle **MUSIQ** via la bibliothèque PyIQA sous PyTorch. Cette approche est particulièrement robuste pour évaluer des images à résolutions variables tout en préservant leurs détails multi-échelles. Supporte l'accélération GPU CUDA.
* **Lancement** :
  ```bash
  uv run python aesthetic_evaluator/musiq_aesthetic_evaluator.py "chemin/vers/image.jpg"
  ```

### 4. `nima_aesthetic_evaluator.py` (NIMA - Consensus Esthétique)
* **Description** : Implémente le modèle **NIMA** (Neural Image Assessment) avec une architecture MobileNet sous TensorFlow/Keras. Il prédit la distribution de probabilité des notes d'une image sur 10, calcule le score esthétique moyen et l'écart-type de consensus, puis génère un tableau de synthèse. Supporte l'évaluation par lot et récursive.
* **Lancement** :
  ```bash
  uv run python aesthetic_evaluator/nima_aesthetic_evaluator.py "chemin/vers/image_ou_dossier"
  ```

### 5. `vlm_aesthetic_evaluator.py` (VLM - Critique Multicritère)
* **Description** : Exploite un grand modèle de vision local (**VLM** `qwen2.5vl:7b` ou similaire via Ollama) pour générer une critique textuelle riche, structurée et argumentée selon 5 axes esthétiques (Composition, Couleurs/Lumières, Textures, Émotion, Technique) avec une note esthétique finale sur 10.
* **Lancement** :
  ```bash
  uv run python aesthetic_evaluator/vlm_aesthetic_evaluator.py -i "image.jpg"
  ```
