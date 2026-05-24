"""
Trieur physique d'images par niveau de qualité esthétique prédite.

Ce script automatise la répartition d'un lot d'images dans des sous-dossiers
physiques spécifiques nommés (de 0_VeryPoor à 5_Excellent) en se basant sur un
fichier de prédictions CSV (généré par les outils d'évaluation de qualité).
Il utilise shutil.copy2 pour cloner les images tout en préservant leurs métadonnées d'origine.
"""

import os
import pandas as pd
import shutil
import argparse


def organize_images_by_quality(csv_path: str, output_dir: str):
    """
    Lit le fichier CSV de prédiction de qualité et copie chaque image dans son sous-dossier de qualité dédié.

    Args:
        csv_path (str): Chemin d'accès vers le fichier CSV de résultats.
        output_dir (str): Dossier racine de destination pour les classes triées.
    """
    # 1. Lecture du fichier CSV avec gestion d'erreurs
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier CSV: {str(e)}")
        return

    # 2. Dictionnaire de mapping pour associer chaque classe de qualité (0-5) à un sous-dossier explicite
    quality_mapping = {
        0: "0_VeryPoor",
        1: "1_Poor",
        2: "2_Medium",
        3: "3_Good",
        4: "4_VeryGood",
        5: "5_Excellent"
    }

    # 3. Création automatique des répertoires de destination si absents
    for quality_folder in quality_mapping.values():
        os.makedirs(os.path.join(output_dir, quality_folder), exist_ok=True)

    # 4. Parcours ligne par ligne du DataFrame
    for index, row in df.iterrows():
        image_path = row['image_path']
        predicted_quality = row['predicted_quality']

        # Vérification d'existence de l'image source d'origine
        if not os.path.exists(image_path):
            print(f"⚠️ Image d'origine introuvable sur le disque: {image_path}")
            continue

        # Extraction du nom simple du fichier
        filename = os.path.basename(image_path)

        # Détermination du dossier cible basé sur la prédiction
        quality_folder = quality_mapping.get(predicted_quality, "Unknown_Quality")
        dest_dir = os.path.join(output_dir, quality_folder)
        dest_path = os.path.join(dest_dir, filename)

        # Copie physique de l'image
        try:
            # L'utilisation de copy2 permet de conserver la date de création et d'autres métadonnées précieuses de l'image
            shutil.copy2(image_path, dest_path)
            print(f"✓ Copié : {filename} -> {quality_folder}/")
        except Exception as e:
            print(f"❌ Erreur lors du transfert de l'image {filename}: {str(e)}")


def main():
    """Point d'entrée du script CLI."""
    parser = argparse.ArgumentParser(description="Trie les images physiques par dossiers de qualité à partir d'un CSV de résultats.")
    parser.add_argument('csv_file', help="Chemin vers le fichier CSV de prédictions")
    parser.add_argument('output_directory', help="Répertoire racine de destination pour les dossiers triés")

    args = parser.parse_args()

    # Validation de l'existence du fichier CSV
    if not os.path.exists(args.csv_file):
        print(f"❌ Erreur : Le fichier CSV '{args.csv_file}' n'existe pas.")
        return

    # Validation et création du dossier racine de sortie
    try:
        os.makedirs(args.output_directory, exist_ok=True)
    except Exception as e:
        print(f"❌ Erreur lors de la création du répertoire racine de destination: {str(e)}")
        return

    # Lancement du tri des images
    organize_images_by_quality(args.csv_file, args.output_directory)


if __name__ == "__main__":
    main()
