# 🎓 Module d'Entraînement et d'Évaluation de Modèles (`training/`)

> 🇬🇧 An English version of this document is available in [README.md](README.md).

Ce module héberge les outils nécessaires pour entraîner (fine-tuner) des modèles pré-entraînés d'évaluation de la qualité, que ce soit des architectures modernes de deep learning (Vision Transformers) ou des approches classiques de computer vision (régression sur descripteurs OpenCV). Il propose également des fonctions de calcul et de visualisation des métriques de performance.

---

## 📁 Description des Scripts

### 1. `train_quality_vit.py` (Fine-tuning ViT)
* **Description** : Permet de fine-tuner un modèle Vision Transformer (ViT de Google) sur votre propre base de données. Il gère la lecture des dossiers triés (de `quality_0` à `quality_5`), le prétraitement des tenseurs d'images en $224 \times 224$ pixels, et utilise le package Hugging Face `Trainer` pour optimiser l'apprentissage.
* **Fonctionnement** :
  * Indexe les dossiers d'images.
  * Prépare le dataset et le découpe en $70\%$ d'entraînement et $30\%$ de validation.
  * Lance l'apprentissage de classification sur 6 classes.

### 2. `train_quality_rf.py` (Régresseur OpenCV + Random Forest)
* **Description** : Pipeline classique de vision par ordinateur. Il extrait des descripteurs mathématiques sur l'intensité des pixels (variance du Laplacien pour la netteté, écart-type pour le contraste, moyenne pour l'exposition, entropie visuelle et magnitude Sobel) et entraîne un régresseur Forêt Aléatoire (Random Forest) avec Scikit-Learn.
* **Fonctionnement** :
  * Calcule les caractéristiques OpenCV des images.
  * Divise le dataset (80/20) et entraîne le RandomForestRegressor.
  * Sauvegarde le modèle sous forme de fichier Pickle (`quality_model.pkl`).

### 3. `metrics_evaluator.py` (Calculateur de Performances)
* **Description** : Script d'évaluation diagnostique. Il génère le rapport de classification détaillé (précision, rappel, F1-score par note) et affiche dynamiquement la heatmap de la matrice de confusion à l'aide de Seaborn et Matplotlib pour mesurer les déviations du modèle.

### 4. `example_vit_training.py` (Script d'Exemple d'Intégration)
* **Description** : Guide de mise en œuvre pratique et autonome. Il montre comment structurer votre code pour charger le classificateur ViT, préparer les données disques, entraîner le modèle et prédire la qualité d'images de test. *(Les imports relatifs ont été corrigés pour le rendre fonctionnel dès le premier lancement)*.
