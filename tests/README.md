# 🧪 Module de Tests d'Intégration et de Validation (`tests/`)

Ce module (anciennement `testaesthetic-predictor/`) contient les scripts de test rapide et d'intégration de base (smoke tests) servant à valider le comportement fonctionnel de vos dépendances logicielles d'évaluation.

---

## 📁 Description des Scripts

### 1. `test_aesthetic_predictor.py` (Smoke Test d'Intégration)
* **Description** : Script minimal conçu pour s'assurer que la bibliothèque de prédiction esthétique (`aesthetic_predictor`) est correctement installée sur votre machine et s'interface parfaitement avec Pillow pour charger une image de test JPEG.
* **Fonctionnement** :
  * Charge l'image de démonstration `bestImages/quality_5/00000-1671415246.jpg`.
  * Invoque la fonction `predict_aesthetic` pour évaluer l'image.
  * Affiche le score esthétique brut obtenu.
* **Lancement** :
  ```bash
  uv run python tests/test_aesthetic_predictor.py
  ```

