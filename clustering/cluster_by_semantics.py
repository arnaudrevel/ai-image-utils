#!/usr/bin/env python3
"""
Regroupement d'Images Similaires par Vecteurs de Caractéristiques ResNet50 et Algorithme DBSCAN.

Ce script implémente une méthode avancée d'apprentissage par transfert (Transfer Learning).
Il utilise un réseau de neurones convolutionnel ResNet50 pré-entraîné sur ImageNet pour extraire
des vecteurs d'empreintes visuelles de haute dimension (2048 dimensions), puis applique
l'algorithme de clustering DBSCAN de Scikit-Learn (basé sur la similarité cosinus) pour regrouper
automatiquement et intelligemment les images similaires sans avoir à spécifier le nombre de groupes à l'avance.
"""

import os
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from collections import defaultdict


def extract_features(image_path: str, model, transform, device) -> np.ndarray:
    """
    Extrait le vecteur de caractéristiques profondes d'une image à l'aide de ResNet50.

    Args:
        image_path (str): Chemin d'accès vers le fichier image.
        model: Le modèle ResNet50 modifié (sans sa couche linéaire finale).
        transform: Les transformations spatiales à appliquer à l'image (Resize, Normalize).
        device: Le périphérique matériel d'exécution (CPU ou GPU CUDA).

    Returns:
        np.ndarray: Vecteur de caractéristiques de 2048 dimensions, normalisé L2.
    """
    # Ouverture de l'image et forçage en RGB (3 canaux)
    image = Image.open(image_path).convert('RGB')
    
    # Application des transformations et ajout d'une dimension de batch (unsqueeze)
    image = transform(image).unsqueeze(0).to(device)

    # Inférence sans calcul de gradients
    with torch.no_grad():
        features = model(image)

    # Normalisation L2 du vecteur de caractéristiques.
    # Diviser le vecteur par sa norme L2 permet de ramener sa longueur à 1.
    # Ainsi, la distance euclidienne standard équivaut à la similarité cosinus (idéal pour DBSCAN).
    features = F.normalize(features, p=2, dim=1)
    
    # Conversion du tenseur PyTorch en tableau NumPy à plat (1D)
    return features.cpu().numpy().flatten()


def load_images_from_folder(folder: str) -> list:
    """
    Recherche toutes les images prises en charge dans un répertoire racine.

    Args:
        folder (str): Dossier cible.

    Returns:
        list: Chemins complets des images trouvées.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            images.append(os.path.join(folder, filename))
    return images


def cluster_images(features: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
    """
    Exécute l'algorithme DBSCAN pour regrouper les vecteurs d'images similaires.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) est parfait ici :
    il regroupe les points denses et isole les points éloignés (bruit), ce qui évite
    de forcer des images uniques à rejoindre un cluster inapproprié.

    Args:
        features (np.ndarray): Matrice de caractéristiques (N_images x 2048).
        eps (float): Distance maximale de similarité pour considérer deux images voisines (seuil cosinus).
        min_samples (int): Nombre minimal d'images nécessaires pour former un groupe de similarité.

    Returns:
        np.ndarray: Tableau de labels entiers associés à chaque image (le bruit est labélisé -1).
    """
    # Utilisation de la métrique 'cosine' pour évaluer l'angle entre les vecteurs de caractéristiques
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(features)
    return clustering.labels_


def move_images_to_clusters(image_paths: list, labels: np.ndarray, dest_folder: str, handle_noise: bool = True):
    """
    Répartit et copie les images physiques dans des répertoires distincts correspondant à leurs clusters.

    Args:
        image_paths (list): Liste des chemins d'origine des images.
        labels (np.ndarray): Labels de clusters calculés par DBSCAN.
        dest_folder (str): Dossier racine de destination.
        handle_noise (bool): Si True, range les images uniques/bruit (label -1) dans un dossier dédié 'noise'.
    """
    # Création d'un mapping temporaire : { label_de_cluster: [chemins_d_images] }
    label_to_images = defaultdict(list)
    for img_path, label in zip(image_paths, labels):
        label_to_images[label].append(img_path)

    # Parcours des groupes pour la copie physique
    for label, images in label_to_images.items():
        if label == -1 and handle_noise:
            cluster_folder = os.path.join(dest_folder, "noise")
        else:
            cluster_folder = os.path.join(dest_folder, f"cluster_{label}")
            
        os.makedirs(cluster_folder, exist_ok=True)

        for img_path in images:
            # Copie simple de chaque fichier vers son dossier cible
            shutil.copy(img_path, os.path.join(cluster_folder, os.path.basename(img_path)))

    # Statistiques d'affichage
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✓ Images regroupées avec succès sous '{dest_folder}' dans {n_clusters} clusters distincts.")
    if -1 in labels and handle_noise:
        print(f"💡 {len(label_to_images[-1])} images uniques ont été qualifiées de 'bruit' et isolées dans le dossier 'noise'.")


def main(source_folder: str, dest_folder: str, eps: float = 0.5, min_samples: int = 2):
    """
    Pipeline principal : chargement du modèle ResNet, extraction et clustering.
    """
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Le dossier source '{source_folder}' n'existe pas.")
        
    os.makedirs(dest_folder, exist_ok=True)

    # 1. Configuration matérielle (GPU CUDA si disponible, sinon CPU standard)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Matériel utilisé pour l'extraction de caractéristiques : {device}")

    # 2. Chargement du modèle de vision ResNet50 pré-entraîné
    # Nous extrayons la structure interne de ResNet50 en éliminant la toute dernière couche linéaire FC
    # (Fully Connected) qui servait à classifier les 1000 classes d'ImageNet.
    # Nous obtenons ainsi un extracteur de caractéristiques pures à 2048 dimensions.
    resnet = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # Retire la couche de classification linéaire finale
    model = model.to(device)
    model.eval()  # Désactive les couches de dropout et batchnorm pour l'évaluation

    # 3. Définition des transformations spatiales requises par ResNet
    transform = transforms.Compose([
        transforms.Resize(256),             # Redimensionne l'image pour que le plus petit côté fasse 256 pixels
        transforms.CenterCrop(224),         # Recadrage central carré de 224x224 pixels
        transforms.ToTensor(),              # Conversion en tenseur PyTorch normalisé [0.0, 1.0]
        # Normalisation standard d'ImageNet (moyenne et écart-type sur chaque canal R, G, B)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Chargement et validation de la liste d'images
    image_paths = load_images_from_folder(source_folder)
    if not image_paths:
        raise ValueError(f"Aucune image trouvée dans le répertoire source '{source_folder}'.")

    print(f"Indexation terminée. Début de l'extraction sur {len(image_paths)} images...")

    # 5. Extraction séquentielle des empreintes profondes
    features = []
    for img_path in tqdm(image_paths, desc="Extraction de features"):
        try:
            feature = extract_features(img_path, model, transform, device)
            features.append(feature)
        except Exception as e:
            print(f"\n⚠️ Fichier ignoré suite à une erreur d'analyse {img_path}: {e}")
            
    features = np.array(features)

    # 6. Exécution du clustering DBSCAN
    print("Application de l'algorithme DBSCAN (similarité cosinus)...")
    labels = cluster_images(features, eps=eps, min_samples=min_samples)

    # 7. Rangement physique des fichiers
    move_images_to_clusters(image_paths, labels, dest_folder, handle_noise=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regroupe dynamiquement des images similaires à l'aide de ResNet50 et de DBSCAN."
    )
    parser.add_argument("source_folder", help="Chemin du dossier source contenant les images à trier")
    parser.add_argument("dest_folder", help="Chemin du dossier de destination final pour les clusters")
    parser.add_argument(
        "--eps", 
        type=float, 
        default=0.5,
        help="Rayon maximal de similarité cosinus pour DBSCAN (défaut: 0.5, plus bas = plus strict)"
    )
    parser.add_argument(
        "--min_samples", 
        type=int, 
        default=2,
        help="Nombre minimal d'images nécessaires pour constituer un groupe (défaut: 2)"
    )

    args = parser.parse_args()

    main(
        source_folder=args.source_folder,
        dest_folder=args.dest_folder,
        eps=args.eps,
        min_samples=args.min_samples
    )
