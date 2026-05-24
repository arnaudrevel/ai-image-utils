# 🛠️ Module d'Utilitaires de Rangement (`utils/`)

Ce module contient les scripts utilitaires servant à gérer, trier et structurer physiquement vos fichiers d'images sur votre disque dur en s'appuyant sur des fichiers d'annotations ou de prédictions au format CSV.

---

## 📁 Description des Scripts

### 1. `reorganize_images_csv.py` (Rangement CSV Générique)
* **Description** : Permet de déplacer (ou de copier) des images éparpillées sur le disque vers des sous-dossiers spécifiques en fonction des colonnes `path` (chemin d'origine) et `dir` (nom du dossier de destination) d'un CSV de mapping.
* **Caractéristiques** :
  * **Résolution des conflits** : Si un fichier portant le même nom existe déjà dans le répertoire cible, le script lui attribue automatiquement un compteur incrémental unique (ex: `photo_1.jpg`) pour éviter d'écraser des données.
  * **Option Copie** : L'option `--copy` patche dynamiquement la fonction de déplacement par une copie (`shutil.copy2`) préservant les métadonnées.
* **Fonctionnement** :
  ```bash
  python utils/reorganize_images_csv.py "chemin/vers/mapping.csv" "chemin/vers/dossier_cible" --copy
  ```

### 2. `sort_images_by_quality.py` (Tri par Note de Qualité)
* **Description** : Script spécialisé qui lit un fichier CSV contenant des prédictions de qualité (avec colonnes `image_path` et `predicted_quality`) et classe automatiquement les images physiques dans 6 dossiers structurés distincts, de `0_VeryPoor` à `5_Excellent`.
* **Caractéristiques** :
  * Création automatique des répertoires de qualité cibles.
  * Utilisation de `shutil.copy2` afin de préserver l'horodatage et les métadonnées originales des fichiers d'images.
* **Fonctionnement** :
  ```bash
  python utils/sort_images_by_quality.py "chemin/vers/predictions.csv" "chemin/vers/repertoire_tri"
  ```

### 3. `CopyTopNFiles.ps1` (Copie Sélectionnée d'Images - PowerShell)
* **Description** : Script utilitaire PowerShell permettant d'extraire et de copier rapidement les $N$ premiers fichiers (triés par nom ou pertinence) d'un dossier source vers un dossier de destination, avec gestion automatique des créations de dossiers et écrasement éventuel facultatif.
* **Fonctionnement** :
  ```powershell
  .\utils\CopyTopNFiles.ps1 -SourcePath "chemin/vers/source" -DestinationPath "chemin/vers/destination" -NumberOfFiles 10 -Overwrite
  ```

