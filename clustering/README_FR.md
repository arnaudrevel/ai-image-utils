# 🔗 Module d'Algorithmes de Regroupement et Doublons (`clustering/`)

> 🇬🇧 An English version of this document is available in [README.md](README.md).

Ce module est dédié au regroupement visuel (clustering) automatique et à la détection de doublons ou de variantes sémantiques au sein de vos banques d'images. Il propose deux approches distinctes adaptées à vos besoins :

---

## 📁 Description des Outils

### 1. `cluster_by_duplicates.py` (Détection de Doublons et Variantes Physiques)
* **Description** : Détecte les copies conformes ou quasi-conformes à l'aide de l'empreinte numérique **pHash** (Perceptual Hash) basée sur la transformée en cosinus discrète (DCT). Les images ayant une distance de Hamming (nombre de bits différents) inférieure au seuil choisi sont regroupées et rangées dans des dossiers `groupe_00X/`.
* **Caractéristiques** :
  * **Ultra-léger** : Aucun réseau de neurones lourd, s'exécute en quelques millisecondes sur CPU uniquement.
  * **Cas d'usage** : Idéal pour nettoyer une base en trouvant les doublons stricts, les images redimensionnées, compressées en JPG, ou avec de légers filtres de couleur.
* **Lancement** :
  ```bash
  uv run python clustering/cluster_by_duplicates.py "chemin/dossier_source" "chemin/dossier_destination" --threshold 5
  ```

### 2. `cluster_by_semantics.py` (Regroupement Sémantique Profond)
* **Description** : Approche d'apprentissage par transfert (Transfer Learning) non supervisée. Le script extrait des vecteurs de caractéristiques profondes à 2048 dimensions à l'aide du réseau de neurones de vision **ResNet50** (pré-entraîné sur ImageNet). Il applique ensuite l'algorithme de densité de Scikit-Learn **DBSCAN** avec similarité cosinus pour regrouper les images.
* **Caractéristiques** :
  * **Intelligent & Sémantique** : Regroupe des images partageant les mêmes concepts, compositions spatiales, ou sujets, même si les pixels et la résolution diffèrent totalement.
  * **Nombre de groupes dynamique** : DBSCAN détermine lui-même le nombre optimal de clusters.
  * **Isolation du bruit** : Les images jugées uniques/isolées ne sont pas forcées dans un groupe et sont rangées à part dans un dossier `noise/` (bruit).
* **Lancement** :
  ```bash
  uv run python clustering/cluster_by_semantics.py "chemin/source" "chemin/destination" --eps 0.5 --min_samples 2
  ```
