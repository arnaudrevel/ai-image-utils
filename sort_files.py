import os
import pandas as pd
import shutil
import argparse

def organize_images_by_quality(csv_path, output_dir):
    """
    Organise les images en fonction de leur qualité prédite.

    Args:
        csv_path (str): Chemin vers le fichier CSV contenant les données
        output_dir (str): Répertoire de sortie où seront créés les sous-dossiers de qualité
    """
    # Lire le fichier CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV: {str(e)}")
        return

    # Créer un dictionnaire pour mapper les labels de qualité aux dossiers
    quality_mapping = {
        0: "0_VeryPoor",
        1: "1_Poor",
        2: "2_Medium",
        3: "3_Good",
        4: "4_VeryGood",
        5: "5_Excellent"
    }

    # Créer les dossiers de destination s'ils n'existent pas
    for quality in quality_mapping.values():
        os.makedirs(os.path.join(output_dir, quality), exist_ok=True)

    # Parcourir chaque ligne du DataFrame
    for index, row in df.iterrows():
        image_path = row['image_path']
        predicted_quality = row['predicted_quality']

        # Vérifier si le fichier existe
        if not os.path.exists(image_path):
            print(f"Fichier introuvable: {image_path}")
            continue

        # Obtenir le nom du fichier
        filename = os.path.basename(image_path)

        # Déterminer le dossier de destination
        quality_folder = quality_mapping.get(predicted_quality, "Unknown_Quality")
        dest_dir = os.path.join(output_dir, quality_folder)
        dest_path = os.path.join(dest_dir, filename)

        # Déplacer le fichier
        try:
            shutil.copy2(image_path, dest_path)  # Utilisation de copy2 pour préserver les métadonnées
            print(f"Copié: {filename} vers {quality_folder}")
        except Exception as e:
            print(f"Erreur lors de la copie de {filename}: {str(e)}")

def main():
    # Configuration du parseur d'arguments
    parser = argparse.ArgumentParser(description='Organise les images par qualité à partir d\'un fichier CSV.')
    parser.add_argument('csv_file', help='Chemin vers le fichier CSV contenant les données')
    parser.add_argument('output_directory', help='Répertoire de sortie pour les images organisées')

    # Analyse des arguments
    args = parser.parse_args()

    # Vérification que le fichier CSV existe
    if not os.path.exists(args.csv_file):
        print(f"Erreur: Le fichier CSV '{args.csv_file}' n'existe pas.")
        return

    # Vérification que le répertoire de sortie est valide
    try:
        os.makedirs(args.output_directory, exist_ok=True)
    except Exception as e:
        print(f"Erreur lors de la création du répertoire de sortie: {str(e)}")
        return

    # Exécution de l'organisation des images
    organize_images_by_quality(args.csv_file, args.output_directory)

if __name__ == "__main__":
    main()
