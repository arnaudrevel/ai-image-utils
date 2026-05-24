# 🖼️ Module d'Interfaces Graphiques Gradio (`gui/`)

> 🇬🇧 An English version of this document is available in [README.md](README.md).

Ce module rassemble les interfaces graphiques web interactives locales créées avec la bibliothèque Gradio. Elles permettent d'annoter manuellement des images de manière ergonomique, d'établir des classements comparatifs par paires, ou d'explorer visuellement et statistiquement les prédictions générées par vos modèles.

---

## 📁 Description des Interfaces

### 1. `quality_annotator.py` (Annotateur Manuel Absolu)
* **Description** : Interface web simplifiée permettant à l'utilisateur de passer en revue un lot d'images, de leur attribuer une note de 0 (Très mauvaise) à 5 (Excellente), et d'enregistrer ces annotations au format CSV.
* **Caractéristiques** :
  * Écriture concurrente sécurisée (utilisation d'un verrou thread-safe).
  * Navigation réactive (boutons Précédent, Suivant/Ignorer).
  * Reprise automatique : évite de proposer à nouveau les images déjà annotées lors d'une session précédente.
* **Lancement** :
  ```bash
  uv run python gui/quality_annotator.py "chemin/vers/images" --recursive
  ```

### 2. `ab_vote.py` (Vote Comparatif A/B Unifié)
* **Description** : Interface d'évaluation subjective basée sur les comparaisons par paires (pairwise comparisons). Les utilisateurs votent interactivement dans des duels aléatoires entre deux images, ce qui permet d'établir un classement esthétique global extrêmement précis en réduisant les biais subjectifs.
* **Méthodes disponibles** :
  * **Système Elo** (`--method elo`) : Les scores évoluent de manière non linéaire (facteur $K=32$) selon l'écart de force et la probabilité mathématique de gain. En fin de session, les scores bruts sont normalisés linéairement de $0.0$ à $5.0$.
  * **Vote Condorcet** (`--method condorcet`) : Les scores évoluent de manière linéaire par pas constants ($+0.1$/$-0.1$), bornés de $0$ à $5$. Les images ayant atteint le score minimal ($0$) ou maximal ($5$) sont définitivement classées et exclues des prochains tirages pour focaliser l'attention sur les cas indécis.
* **Caractéristiques** :
  * Chargement automatique des images d'entrée depuis `data/inputs/pairwise_voting/` par défaut.
  * Sauvegarde automatique et reprise de session via `annotation_state.json`.
  * Export des résultats finals et historiques au format CSV dans `data/outputs/predictions/pairwise_rankings.csv`.
* **Lancement** :
  ```bash
  # Lancement par défaut en mode Elo
  uv run python gui/ab_vote.py --method elo

  # Lancement en mode Condorcet
  uv run python gui/ab_vote.py --method condorcet
  ```

### 3. `results_dashboard.py` (Dashboard Unifié)
* **Description** : Dashboard de visualisation permettant de charger un fichier CSV de résultats et d'explorer interactivement les scores esthétiques. Propose deux modes d'affichage basculables via un toggle intégré à l'interface :
  * **Mode Standard** (défaut) : affiche toutes les colonnes du CSV avec formatage lisible (pourcentages, emojis succès/échec).
  * **Mode Miniatures** : réduit l'affichage aux colonnes essentielles (`image_path`, `quality_label`) et injecte des aperçus d'images directement dans le tableau via des balises HTML `<img>`.
* **Caractéristiques** :
  * Bascule entre les deux modes sans rechargement du fichier CSV.
  * Clic de ligne pour afficher l'image en haute résolution dans la visionneuse.
  * Résumé statistique (total, succès, échecs, note moyenne).
* **Lancement** :
  ```bash
  uv run python gui/results_dashboard.py
  ```
