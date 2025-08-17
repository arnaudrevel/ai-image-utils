import csv
import os
import shutil
import argparse
from pathlib import Path

def move_images_from_csv(csv_file, target_directory):
    """
    Déplace les images vers les bons répertoires selon le fichier CSV
    
    Args:
        csv_file (str): Chemin vers le fichier CSV
        target_directory (str): Répertoire de destination racine
    """
    
    # Vérifier que le fichier CSV existe
    if not os.path.exists(csv_file):
        print(f"Erreur : Le fichier CSV '{csv_file}' n'existe pas.")
        return
    
    # Créer le répertoire de destination s'il n'existe pas
    target_path = Path(target_directory)
    target_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    error_count = 0
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row_num, row in enumerate(csv_reader, start=2):  # Start=2 car ligne 1 = header
                try:
                    source_path = row['path'].strip()
                    directory_name = row['dir'].strip()
                    
                    # Créer le chemin de destination complet
                    dest_dir = target_path / directory_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Obtenir le nom du fichier depuis le chemin source
                    filename = Path(source_path).name
                    dest_file = dest_dir / filename
                    
                    # Vérifier que le fichier source existe
                    if not os.path.exists(source_path):
                        print(f"Attention ligne {row_num}: Fichier source introuvable : {source_path}")
                        error_count += 1
                        continue
                    
                    # Déplacer le fichier
                    if dest_file.exists():
                        print(f"Attention ligne {row_num}: Le fichier de destination existe déjà : {dest_file}")
                        # Optionnel: créer un nom unique
                        base_name = dest_file.stem
                        extension = dest_file.suffix
                        counter = 1
                        while dest_file.exists():
                            new_name = f"{base_name}_{counter}{extension}"
                            dest_file = dest_dir / new_name
                            counter += 1
                        print(f"Renommé en : {dest_file}")
                    
                    shutil.move(source_path, dest_file)
                    print(f"✓ Déplacé : {filename} -> {directory_name}/")
                    moved_count += 1
                    
                except KeyError as e:
                    print(f"Erreur ligne {row_num}: Colonne manquante : {e}")
                    error_count += 1
                except Exception as e:
                    print(f"Erreur ligne {row_num}: {e}")
                    error_count += 1
    
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
        return
    
    print(f"\n--- Résumé ---")
    print(f"Images déplacées avec succès : {moved_count}")
    print(f"Erreurs rencontrées : {error_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Range les images selon un fichier CSV dans des sous-répertoires"
    )
    parser.add_argument(
        "csv_file", 
        help="Chemin vers le fichier CSV contenant les informations des images"
    )
    parser.add_argument(
        "target_directory", 
        help="Répertoire de destination où ranger les images"
    )
    parser.add_argument(
        "--copy", 
        action="store_true", 
        help="Copier les fichiers au lieu de les déplacer"
    )
    
    args = parser.parse_args()
    
    # Si l'option --copy est utilisée, modifier la fonction pour copier
    if args.copy:
        global shutil
        original_move = shutil.move
        shutil.move = shutil.copy2
        print("Mode copie activé")
    
    move_images_from_csv(args.csv_file, args.target_directory)

if __name__ == "__main__":
    main()
