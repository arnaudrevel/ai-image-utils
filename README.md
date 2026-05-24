# 🖼️ AI Image Quality Utilities (Outils d'Évaluation de Qualité Esthétique)

Bienvenue dans la suite d'outils professionnels pour l'évaluation, l'annotation, l'entraînement et la gestion qualitative de banques d'images. Ce projet propose un ensemble de scripts allant du fine-tuning de Vision Transformers (ViT) à l'analyse esthétique assistée par grands modèles de vision locaux (VLM), en passant par des interfaces de duels A/B (Elo/Condorcet) et du clustering de doublons.

---

## 📂 Architecture Générale du Projet

Le projet est entièrement structuré sous forme de modules fonctionnels et documentés pour séparer proprement les rôles :

```
c:\Users\revel\OneDrive\Desktop\Python\Qualité image/
│
├── README.md                     # Guide général d'installation et d'utilisation (Ce fichier)
│
├── pyproject.toml                # ⚙️ Configuration globale et dépendances gérées par UV
│
├── aesthetic_evaluator/          # 🔍 1. Estimation & Analyse de Qualité
│   ├── vit_aesthetic_evaluator.py      # CLI d'estimation de qualité par paliers (ViT)
│   ├── clip_aesthetic_evaluator.py     # Évaluation de masse par lots d'images (CLIP)
│   ├── musiq_aesthetic_evaluator.py    # Évaluation multi-échelle (images à résolutions variables)
│   ├── nima_aesthetic_evaluator.py     # Notation esthétique et consensus (MobileNet)
│   └── vlm_aesthetic_evaluator.py      # Critique textuelle détaillée via modèle de vision (VLM)
│
├── training/                     # 🎓 2. Entraînement & Évaluation de Modèles
│   ├── train_quality_vit.py            # Fine-tuning supervisé de Google Vision Transformer (ViT)
│   ├── train_quality_rf.py             # Descripteurs OpenCV + Régresseur Random Forest
│   ├── metrics_evaluator.py            # Heatmap confusion (Seaborn) & rapports (Scikit-Learn)
│   └── example_vit_training.py         # Guide d'intégration et exemple d'entraînement complet
│
├── gui/                          # 🖥️ 3. Tableaux de Bord & Annotations Manuelles
│   ├── gui_quality_annotator.py        # Labellisation Gradio manuelle rapide (scores 0 à 5)
│   ├── gui_ab_vote.py                  # [NOUVEAU] Vote comparatif A/B unifié (Elo & Condorcet)
│   ├── gui_results_dashboard_basic.py  # Dashboard Gradio standard de visualisation de rapports
│   └── gui_results_dashboard.py        # Dashboard Gradio évolué avec miniatures HTML dynamiques
│
├── utils/                        # 🛠️ 4. Tri & Rangement de Fichiers
│   ├── reorganize_images_csv.py        # Rangement via fichier CSV (Déplacement/Copie avec renommage unique)
│   ├── sort_images_by_quality.py       # Copie ordonnée des images dans les sous-dossiers de note
│   └── CopyTopNFiles.ps1               # Script PowerShell de copie des N premiers fichiers d'un dossier
│
├── clustering/                   # 🔗 5. Algorithmes de Regroupement et Doublons
│   ├── cluster_by_duplicates.py        # [NOUVEAU NOM] Détection de doublons (pHash rapide sur CPU)
│   └── cluster_by_semantics.py         # [NOUVEAU NOM] Regroupement sémantique (ResNet50 + DBSCAN)
│
└── tests/                        # 🧪 7. Tests d'Intégration et Validation
    └── test_aesthetic_predictor.py     # Smoke test pour valider le chargement du prédicteur
```

> [!NOTE]
> Chacun de ces sous-dossiers contient son propre fichier `README.md` détaillant spécifiquement le rôle et les lignes de commandes de chaque script.

---

## ⚙️ Prérequis et Dépendances (Gestion par UV)

Ce projet utilise **`uv`**, le gestionnaire de paquets Python de nouvelle génération ultra-rapide. Toutes les dépendances sont centralisées à la racine dans le fichier [pyproject.toml](file:///c:/Users/revel/OneDrive/Desktop/Python/Qualité%20image/pyproject.toml).

### 🛠️ Installation simplifiée :
1. Installez `uv` si ce n'est pas déjà fait :
   ```bash
   pip install uv  # ou via leur installateur officiel
   ```
2. À la racine du projet, lancez la synchronisation de l'environnement :
   ```bash
   uv sync
   ```
   Cette commande crée automatiquement un dossier `.venv` et y installe l'intégralité des dépendances nécessaires (TensorFlow, PyTorch, PyIQA, Transformers, Gradio, OpenCV, etc.).

---

## 🚀 Démarrage Rapide

Tous les scripts peuvent être exécutés directement via l'environnement virtuel avec `uv run python`.

### 1. Annoter manuellement votre base d'images
Lancez l'interface d'annotation Gradio pour attribuer des notes à vos images :
```bash
uv run python gui/gui_quality_annotator.py "chemin/vers/images"
```

### 2. Entraîner un modèle de classification ViT
Organisez vos images dans des sous-dossiers de qualité (`quality_0` à `quality_5`) sous `bestImages/` et lancez le pipeline de fine-tuning :
```bash
uv run python training/train_quality_vit.py
```

### 3. Analyser une image ou un dossier (Multi-Modèles)
Vous disposez de plusieurs modèles complémentaires sous le dossier `aesthetic_evaluator/` :

*   **Vision Transformer (ViT)** (CPU ou GPU CUDA Auto-détection) :
    ```bash
    uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "chemin/vers/image.jpg" -v
    uv run python aesthetic_evaluator/vit_aesthetic_evaluator.py "chemin/vers/image.jpg" --cpu -v  # Forcer le mode CPU
    ```
*   **Modèle NIMA (Neural Image Assessment)** (TensorFlow/MobileNet) :
    ```bash
    uv run python aesthetic_evaluator/nima_aesthetic_evaluator.py "chemin/vers/image_ou_dossier"
    ```
*   **Métrique MUSIQ (Multi-scale Image Quality)** (PyTorch/PyIQA) :
    ```bash
    uv run python aesthetic_evaluator/musiq_aesthetic_evaluator.py "chemin/vers/image.jpg"
    ```
*   **Analyse Multicritère VLM** (via Ollama local) :
    ```bash
    uv run python aesthetic_evaluator/vlm_aesthetic_evaluator.py -i "chemin/vers/image.jpg"
    ```

### 4. Lancer une session de vote par duels (A/B)
Pour classer finement des variantes d'images en les comparant deux à deux (duels) via une interface interactive Gradio :
```bash
# Mode Elo (par défaut)
uv run python gui/gui_ab_vote.py --method elo

# Mode Condorcet
uv run python gui/gui_ab_vote.py --method condorcet
```


---

## 📄 Licence
Ce projet est distribué sous les termes de la licence présente dans le fichier `LICENSE` à la racine.
