#!/usr/bin/env python3
"""
Script pour regrouper les images similaires en utilisant imagehash
"""

import os
import sys
import shutil
import argparse
from collections import defaultdict
from pathlib import Path
import imagehash
from PIL import Image

# Extensions d'images supportées
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def calculate_image_hash(image_path, hash_size=8):
    """
    Calcule le hash d'une image
    
    Args:
        image_path (str): Chemin vers l'image
        hash_size (int): Taille du hash (plus petit = moins précis mais plus rapide)
    
    Returns:
        imagehash object ou None si erreur
    """
    try:
        with Image.open(image_path) as img:
            # Utilise perceptual hash (pHash) qui est robuste aux redimensionnements
            return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"Erreur lors du traitement de {image_path}: {e}")
        return None

def find_similar_images(source_dir, similarity_threshold=5):
    """
    Trouve les groupes d'images similaires
    
    Args:
        source_dir (str): Répertoire source contenant les images
        similarity_threshold (int): Seuil de similarité (0 = identiques, plus grand = plus permissif)
    
    Returns:
        dict: Dictionnaire avec les groupes d'images similaires
    """
    print("Analyse des images en cours...")
    
    # Dictionnaire pour stocker les hashes et leurs fichiers correspondants
    image_hashes = {}
    
    # Parcourir tous les fichiers du répertoire source
    source_path = Path(source_dir)
    image_files = []
    
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(source_path.rglob(f"*{ext}"))
        image_files.extend(source_path.rglob(f"*{ext.upper()}"))
    
    print(f"Trouvé {len(image_files)} fichiers image(s)")
    
    # Calculer les hashes
    for i, image_file in enumerate(image_files):
        if i % 50 == 0:  # Afficher le progrès
            print(f"Traitement: {i+1}/{len(image_files)}")
            
        hash_value = calculate_image_hash(str(image_file))
        if hash_value is not None:
            image_hashes[str(image_file)] = hash_value
    
    print(f"Hashes calculés pour {len(image_hashes)} images")
    
    # Grouper les images similaires
    groups = []
    processed = set()
    
    for image_path, hash_value in image_hashes.items():
        if image_path in processed:
            continue
            
        # Créer un nouveau groupe avec cette image
        current_group = [image_path]
        processed.add(image_path)
        
        # Trouver toutes les images similaires à celle-ci
        for other_path, other_hash in image_hashes.items():
            if other_path != image_path and other_path not in processed:
                # Calculer la différence entre les hashes
                hash_diff = hash_value - other_hash
                if hash_diff <= similarity_threshold:
                    current_group.append(other_path)
                    processed.add(other_path)
        
        # Ajouter le groupe seulement s'il contient plus d'une image
        if len(current_group) > 1:
            groups.append(current_group)
    
    return groups

def copy_similar_groups(groups, destination_dir):
    """
    Copie les groupes d'images similaires vers le répertoire de destination
    
    Args:
        groups (list): Liste des groupes d'images similaires
        destination_dir (str): Répertoire de destination
    """
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTrouvé {len(groups)} groupe(s) d'images similaires")
    
    for i, group in enumerate(groups, 1):
        group_dir = dest_path / f"groupe_{i:03d}"
        group_dir.mkdir(exist_ok=True)
        
        print(f"\nGroupe {i} ({len(group)} images):")
        
        for j, image_path in enumerate(group):
            image_file = Path(image_path)
            # Créer un nom unique pour éviter les conflits
            new_name = f"{j:02d}_{image_file.name}"
            dest_file = group_dir / new_name
            
            try:
                shutil.copy2(image_path, dest_file)
                print(f"  Copié: {image_file.name} -> {dest_file}")
            except Exception as e:
                print(f"  Erreur lors de la copie de {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Regrouper les images similaires en utilisant imagehash"
    )
    parser.add_argument(
        "source", 
        help="Répertoire source contenant les images"
    )
    parser.add_argument(
        "destination", 
        help="Répertoire de destination pour les groupes d'images"
    )
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=5,
        help="Seuil de similarité (défaut: 5, plus petit = plus strict)"
    )
    parser.add_argument(
        "--hash-size", 
        type=int, 
        default=8,
        help="Taille du hash (défaut: 8, plus grand = plus précis mais plus lent)"
    )
    
    args = parser.parse_args()
    
    # Vérifier que le répertoire source existe
    if not os.path.exists(args.source):
        print(f"Erreur: Le répertoire source '{args.source}' n'existe pas")
        sys.exit(1)
    
    if not os.path.isdir(args.source):
        print(f"Erreur: '{args.source}' n'est pas un répertoire")
        sys.exit(1)
    
    print(f"Répertoire source: {args.source}")
    print(f"Répertoire destination: {args.destination}")
    print(f"Seuil de similarité: {args.threshold}")
    print(f"Taille du hash: {args.hash_size}")
    print("-" * 50)
    
    # Trouver les groupes d'images similaires
    similar_groups = find_similar_images(
        args.source, 
        similarity_threshold=args.threshold
    )
    
    if not similar_groups:
        print("Aucun groupe d'images similaires trouvé")
        return
    
    # Copier les groupes vers la destination
    copy_similar_groups(similar_groups, args.destination)
    
    print(f"\nTerminé! {len(similar_groups)} groupe(s) copiés vers {args.destination}")

if __name__ == "__main__":
    main()
