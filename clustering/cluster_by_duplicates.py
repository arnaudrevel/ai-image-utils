#!/usr/bin/env python3
"""
Regroupement (Clustering) et Déduplication d'Images par Hash Perceptuel (pHash).

Ce script permet de regrouper automatiquement des images très similaires ou en doublons.
Il calcule le pHash (perceptual hash) de chaque image (robuste au changement de taille, 
de ratio ou de légères retouches couleur) et calcule la distance de Hamming entre les hashes.
Les groupes d'images similaires sont copiés dans des sous-dossiers distincts.
"""

import os
import sys
import shutil
import argparse
from collections import defaultdict
from pathlib import Path
import imagehash
from PIL import Image

# Extensions d'images prises en charge par Pillow
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def calculate_image_hash(image_path: str, hash_size: int = 8):
    """
    Calcule l'empreinte de hash perceptuel (pHash) d'un fichier image.

    Contrairement aux hashs cryptographiques (ex: MD5, SHA-256) où le moindre changement
    de pixel altère totalement le résultat, le pHash génère un hash similaire pour deux
    images visuellement proches.

    Args:
        image_path (str): Chemin vers l'image.
        hash_size (int): Largeur/hauteur du hash (défaut: 8). Plus la valeur est élevée, 
                         plus l'empreinte est précise mais plus la comparaison est lente.

    Returns:
        imagehash.ImageHash: Objet hash perceptuel, ou None si une erreur de lecture survient.
    """
    try:
        with Image.open(image_path) as img:
            # Utilisation de phash (Perceptual Hash) qui s'appuie sur la transformée en cosinus discrète (DCT)
            return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {image_path}: {e}")
        return None


def find_similar_images(source_dir: str, similarity_threshold: int = 5) -> list:
    """
    Parcourt le dossier source et regroupe toutes les images visuellement similaires.

    La comparaison se fait en calculant la distance de Hamming (soustraction d'objets hash).
    Si la différence est inférieure ou égale au seuil défini, les images sont considérées comme similaires.

    Args:
        source_dir (str): Répertoire source contenant les images à analyser.
        similarity_threshold (int): Seuil de tolérance (0 = copie conforme absolue, 
                                     plus le nombre est élevé, plus le tri est permissif).

    Returns:
        list: Liste de listes contenant les chemins des images regroupées.
    """
    print("Analyse et calcul des hashs perceptuels des images...")
    
    image_hashes = {}
    source_path = Path(source_dir)
    image_files = []
    
    # Récupération de tous les chemins d'images valides
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(source_path.rglob(f"*{ext}"))
        image_files.extend(source_path.rglob(f"*{ext.upper()}"))
    
    print(f"Trouvé : {len(image_files)} images au total.")
    
    # 1. Calcul des hashs
    for i, image_file in enumerate(image_files):
        if i % 50 == 0:  # Suivi visuel du progrès
            print(f"Traitement : {i+1}/{len(image_files)} images")
            
        hash_value = calculate_image_hash(str(image_file))
        if hash_value is not None:
            image_hashes[str(image_file)] = hash_value
    
    print(f"✓ Empreintes pHash générées avec succès pour {len(image_hashes)} images.")
    
    # 2. Regroupement (Clustering) par distance de Hamming
    groups = []
    processed = set()  # Permet d'éviter de traiter plusieurs fois la même image
    
    for image_path, hash_value in image_hashes.items():
        if image_path in processed:
            continue
            
        # Création d'un nouveau groupe potentiel ayant cette image comme référence
        current_group = [image_path]
        processed.add(image_path)
        
        # Comparaison avec toutes les autres images non encore classées
        for other_path, other_hash in image_hashes.items():
            if other_path != image_path and other_path not in processed:
                # Soustraire deux objets imagehash donne leur distance de Hamming (nombre de bits différents)
                hash_diff = hash_value - other_hash
                if hash_diff <= similarity_threshold:
                    current_group.append(other_path)
                    processed.add(other_path)
        
        # Le groupe n'est conservé que s'il comporte au moins un doublon ou une image similaire
        if len(current_group) > 1:
            groups.append(current_group)
    
    return groups


def copy_similar_groups(groups: list, destination_dir: str):
    """
    Copie les groupes d'images similaires identifiés dans des sous-dossiers spécifiques.

    Args:
        groups (list): Liste des listes de chemins d'images similaires.
        destination_dir (str): Répertoire cible où ranger les groupes.
    """
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nIdentification de {len(groups)} groupe(s) d'images doublons ou similaires.")
    
    for i, group in enumerate(groups, 1):
        # Création d'un dossier dédié par groupe (ex: groupe_001)
        group_dir = dest_path / f"groupe_{i:03d}"
        group_dir.mkdir(exist_ok=True)
        
        print(f"\n📂 Groupe {i} ({len(group)} images similaires) :")
        
        for j, image_path in enumerate(group):
            image_file = Path(image_path)
            # Ajout d'un préfixe numérique j pour éviter les collisions de noms de fichiers identiques
            new_name = f"{j:02d}_{image_file.name}"
            dest_file = group_dir / new_name
            
            try:
                # copy2 copie l'image ainsi que ses métadonnées d'origine
                shutil.copy2(image_path, dest_file)
                print(f"  -> Copié: {image_file.name} vers {dest_file.parent.name}/")
            except Exception as e:
                print(f"  ❌ Erreur de copie pour {image_path}: {e}")


def main():
    """Point d'entrée CLI pour le regroupement par pHash."""
    parser = argparse.ArgumentParser(
        description="Regroupe les images visuellement similaires ou doublons à l'aide d'un Perceptual Hash (pHash)."
    )
    parser.add_argument(
        "source", 
        help="Répertoire d'origine contenant les images à analyser"
    )
    parser.add_argument(
        "destination", 
        help="Répertoire de sortie pour le rangement des groupes d'images"
    )
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=5,
        help="Seuil de distance de Hamming (défaut: 5). Plus bas = plus strict/identique, plus haut = plus permissif"
    )
    parser.add_argument(
        "--hash-size", 
        type=int, 
        default=8,
        help="Taille du pHash (défaut: 8). Une plus grande taille augmente la précision mais ralentit le calcul"
    )
    
    args = parser.parse_args()
    
    # Validations de sécurité
    if not os.path.exists(args.source):
        print(f"❌ Erreur: Le dossier source '{args.source}' n'existe pas.")
        sys.exit(1)
    
    if not os.path.isdir(args.source):
        print(f"❌ Erreur: '{args.source}' n'est pas un répertoire standard.")
        sys.exit(1)
    
    print(f"📂 Répertoire source       : {args.source}")
    print(f"📂 Répertoire destination  : {args.destination}")
    print(f"🎯 Seuil de similarité     : {args.threshold}")
    print(f"📊 Résolution du pHash     : {args.hash_size}")
    print("-" * 50)
    
    # Recherche et copie
    similar_groups = find_similar_images(args.source, similarity_threshold=args.threshold)
    
    if not similar_groups:
        print("✓ Aucune image similaire ou en doublon détectée.")
        return
    
    copy_similar_groups(similar_groups, args.destination)
    
    print(f"\n🎉 Opération terminée avec succès ! {len(similar_groups)} groupes d'images triés.")


if __name__ == "__main__":
    main()
