"""
Module pour évaluer le score esthétique d'un dossier d'images par lots (batch processing).

Ce script parcourt récursivement ou non un dossier à la recherche d'images PNG,
les regroupe par lots (batchs) de taille fixe pour optimiser la mémoire et le temps
de calcul, puis utilise le modèle de prédiction esthétique pour générer un score.
Des statistiques globales sont également affichées à la fin du traitement.
"""

from argparse import ArgumentParser
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import tqdm

from PIL import Image

# Importation du prédicteur esthétique externe
from aesthetic_predictor import predict_aesthetic


def main():
    """
    Fonction principale du script.
    Configure le parseur d'arguments, charge les images, effectue les prédictions
    par lots et affiche les statistiques de qualité esthétique.
    """
    # Détermination du dossier parent du script
    current_dir = Path(__file__).parent
    
    # Configuration du parseur d'arguments en ligne de commande
    parser = ArgumentParser(description="Évaluation de la qualité esthétique d'un dossier d'images par lots.")
    parser.add_argument(
        "--root-dir", 
        type=Path, 
        default=current_dir / "cc0",
        help="Chemin vers le répertoire contenant les images à analyser (par défaut: dossier './cc0')"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=current_dir / "scores.html",
        help="Chemin du fichier HTML de sortie pour les résultats (par défaut: './scores.html')"
    )
    args = parser.parse_args()
    root_dir: Path = args.root_dir

    # Recherche et tri de toutes les images PNG dans le dossier racine et ses sous-dossiers
    files = sorted(list(root_dir.glob("**/*.png")))
    batch_size = 100
    all_scores = []

    # Traiter les fichiers par blocs (batchs) de `batch_size` pour optimiser l'utilisation de la RAM
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        images = []
        b64_images = []

        desc = f"Batch {i // batch_size + 1} ({len(batch_files)} fichiers)"
        for file in tqdm.tqdm(batch_files, desc=desc):
            try:
                # Charger l'image avec Pillow et forcer en RGB pour plus de robustesse face aux PNG RGBA/niveaux de gris
                img = Image.open(file).convert("RGB")
                images.append(img)
                # Encodage en base64 pour d'éventuels besoins d'affichage HTML ultérieurs
                b64_images.append(b64encode(file.read_bytes()).decode("utf-8"))
            except Exception as e:
                print(f"\nErreur lors de la lecture du fichier {file}: {e}")

        # S'assurer que le batch courant contient des images valides
        if images:
            # Prédire le score esthétique pour le batch courant
            # predict_aesthetic retourne un tenseur qu'on convertit en tableau NumPy unidimensionnel (ravel)
            batch_scores = predict_aesthetic(images).numpy().ravel()
            all_scores.extend(batch_scores.tolist())
            # Afficher quelques statistiques descriptives pour ce batch
            print(f"{desc} -> min: {batch_scores.min():.4f}, mean: {batch_scores.mean():.4f}")

    # Agréger et afficher les statistiques globales pour l'ensemble des images traitées
    if all_scores:
        import numpy as _np

        arr = _np.array(all_scores)
        print("\n=== STATISTIQUES ESTHÉTIQUES GLOBALES ===")
        print(f"Total d'images analysées : {len(all_scores)}")
        print(f"Score minimal            : {arr.min():.4f}")
        print(f"Score maximal            : {arr.max():.4f}")
        print(f"Score moyen              : {arr.mean():.4f}")
        print(f"Écart-type               : {arr.std():.4f}")
    else:
        print("Aucun fichier valide trouvé dans le répertoire spécifié.")

if __name__ == "__main__":
    main()
