"""
Organisateur physique d'images structuré à partir d'un fichier de mapping CSV.

Ce script permet de ranger automatiquement des images éparpillées sur le disque dur
dans des sous-dossiers spécifiques en se basant sur un fichier CSV contenant deux
colonnes essentielles : le chemin d'origine de l'image et le nom du dossier cible.
Il gère de manière autonome la résolution des conflits de noms de fichiers par incrémentation
et propose un mode "copie" au lieu de "déplacement" physique.
"""

import csv
import os
import shutil
import argparse
from pathlib import Path


def move_images_from_csv(csv_file: str, target_directory: str):
    """
    Parcourt le fichier CSV et déplace (ou copie) chaque image vers son sous-répertoire cible.

    Args:
        csv_file (str): Chemin d'accès vers le fichier CSV de mapping.
        target_directory (str): Répertoire de destination racine sous lequel seront créés les dossiers cibles.
    """
    
    # 1. Validation de l'existence du fichier CSV
    if not os.path.exists(csv_file):
        print(f"❌ Erreur : Le fichier CSV '{csv_file}' n'existe pas.")
        return
    
    # 2. Création automatique du répertoire racine de destination
    target_path = Path(target_directory)
    target_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    error_count = 0
    
    try:
        # Ouverture sécurisée en lecture UTF-8
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            # Parcours ligne par ligne
            for row_num, row in enumerate(csv_reader, start=2):  # start=2 car la ligne 1 correspond à l'entête
                try:
                    # Extraction et nettoyage des données des colonnes 'path' et 'dir'
                    source_path = row['path'].strip()
                    directory_name = row['dir'].strip()
                    
                    # Construction du dossier de destination spécifique
                    dest_dir = target_path / directory_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Extraction du nom du fichier original
                    filename = Path(source_path).name
                    dest_file = dest_dir / filename
                    
                    # Vérification d'existence du fichier source d'origine
                    if not os.path.exists(source_path):
                        print(f"⚠️ Alerte ligne {row_num}: Fichier source introuvable : {source_path}")
                        error_count += 1
                        continue
                    
                    # 3. RÉSOLUTION DES CONFLITS DE NOMS DE FICHIERS
                    # Si un fichier portant le même nom existe déjà dans le répertoire cible,
                    # nous évitons de l'écraser. Nous rajoutons un compteur incrémental (ex: photo_1.jpg).
                    if dest_file.exists():
                        print(f"⚠️ Alerte ligne {row_num}: Le fichier de destination existe déjà : {dest_file}")
                        base_name = dest_file.stem
                        extension = dest_file.suffix
                        counter = 1
                        while dest_file.exists():
                            new_name = f"{base_name}_{counter}{extension}"
                            dest_file = dest_dir / new_name
                            counter += 1
                        print(f"-> Renommé en : {dest_file.name}")
                    
                    # 4. Action physique de déplacement ou copie (dépendant de la fonction shutil.move patchée)
                    shutil.move(source_path, dest_file)
                    print(f"✓ Organisé : {filename} -> {directory_name}/")
                    moved_count += 1
                    
                except KeyError as e:
                    print(f"❌ Erreur ligne {row_num}: Colonne requise manquante dans le CSV : {e}")
                    error_count += 1
                except Exception as e:
                    print(f"❌ Erreur ligne {row_num}: {e}")
                    error_count += 1
    
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier CSV : {e}")
        return
    
    # Affichage du rapport récapitulatif final
    print(f"\n=== RAPPORT D'ORGANISATION DES FICHIERS ===")
    print(f"Images traitées avec succès : {moved_count}")
    print(f"Erreurs ou fichiers omis    : {error_count}")


def main():
    """Point d'entrée du script pour la gestion des arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Range les images dans des répertoires physiques ordonnés selon les indications d'un fichier CSV."
    )
    parser.add_argument(
        "csv_file", 
        help="Chemin vers le fichier CSV contenant les colonnes 'path' et 'dir'"
    )
    parser.add_argument(
        "target_directory", 
        help="Répertoire racine de destination pour le rangement"
    )
    parser.add_argument(
        "--copy", 
        action="store_true", 
        help="Copier les images au lieu de les déplacer physiquement (conserve les originaux)"
    )
    
    args = parser.parse_args()
    
    # 5. ASTUCE DYNAMIQUE PYTHON : OVERRIDE DE SHUTIL
    # Si l'utilisateur demande une copie via l'option --copy, nous remplaçons dynamiquement
    # la référence de la fonction 'shutil.move' par la fonction 'shutil.copy2'.
    # Cela évite de dupliquer la logique conditionnelle au cœur de la boucle de traitement.
    if args.copy:
        global shutil
        shutil.move = shutil.copy2
        print("💡 Mode COPIE activé (les images d'origine ne seront pas supprimées)")
    else:
        print("💡 Mode DÉPLACEMENT activé")
    
    # Lancement du rangement
    move_images_from_csv(args.csv_file, args.target_directory)


if __name__ == "__main__":
    main()
