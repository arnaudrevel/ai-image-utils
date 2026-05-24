# 🧪 Module de Tests d'Intégration et de Validation (`tests/`)

> 🇬🇧 An English version of this document is available in [README.md](README.md).

Ce module contient les scripts de test rapide et d'intégration de base (smoke tests) servant à valider le comportement fonctionnel de vos dépendances logicielles d'évaluation.

---

## 📁 Description des Scripts

### 1. `test_aesthetic_predictor.py` (Smoke Test d'Intégration)
* **Description** : Script minimal conçu pour s'assurer que la bibliothèque de prédiction esthétique (`aesthetic_predictor`) est correctement installée sur votre machine et s'interface parfaitement avec Pillow pour charger une image de test JPEG.
* **Fonctionnement** :
  * Charge l'image de démonstration depuis `data/inputs/labeled_tiers/quality_5/`.
  * Invoque la fonction `predict_aesthetic` pour évaluer l'image.
  * Affiche le score esthétique brut obtenu.
* **Lancement** :
  ```bash
  uv run python tests/test_aesthetic_predictor.py
  ```
