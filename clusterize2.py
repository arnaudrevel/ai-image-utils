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

def extract_features(image_path, model, transform, device):
    """Extraire les features d'une image en utilisant un modèle pré-entraîné."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image)

    # Normaliser les features pour la similarité cosinus
    features = F.normalize(features, p=2, dim=1)
    return features.cpu().numpy().flatten()

def load_images_from_folder(folder):
    """Charger les chemins des images depuis un dossier."""
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            images.append(os.path.join(folder, filename))
    return images

def cluster_images(features, eps=0.5, min_samples=2):
    """Regrouper les images en utilisant DBSCAN (nombre de clusters automatique)."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(features)
    return clustering.labels_

def move_images_to_clusters(image_paths, labels, dest_folder, handle_noise=True):
    """Déplacer les images dans des dossiers de destination en fonction des labels.
    Si handle_noise=True, les images marquées comme bruit (label=-1) sont placées dans un dossier 'noise'."""
    # Créer un dictionnaire pour mapper les labels aux chemins
    label_to_images = defaultdict(list)
    for img_path, label in zip(image_paths, labels):
        label_to_images[label].append(img_path)

    # Créer les dossiers de destination et déplacer les images
    for label, images in label_to_images.items():
        if label == -1 and handle_noise:
            cluster_folder = os.path.join(dest_folder, "noise")
        else:
            cluster_folder = os.path.join(dest_folder, f"cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)

        for img_path in images:
            shutil.copy(img_path, os.path.join(cluster_folder, os.path.basename(img_path)))

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Images regroupées dans {n_clusters} clusters sous {dest_folder}")
    if -1 in labels and handle_noise:
        print(f"{len(label_to_images[-1])} images marquées comme bruit (dans 'noise').")

def main(source_folder, dest_folder, eps=0.5, min_samples=2):
    """Fonction principale pour extraire les features et regrouper les images."""
    # Vérifier les dossiers
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Le dossier source {source_folder} n'existe pas.")
    os.makedirs(dest_folder, exist_ok=True)

    # Charger le modèle pré-entraîné (ResNet50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Enlever la couche FC
    model = model.to(device).eval()

    # Transformations pour les images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Charger les images
    image_paths = load_images_from_folder(source_folder)
    if not image_paths:
        raise ValueError(f"Aucune image trouvée dans {source_folder}")

    print(f"Traitement de {len(image_paths)} images...")

    # Extraire les features
    features = []
    for img_path in tqdm(image_paths, desc="Extraction des features"):
        feature = extract_features(img_path, model, transform, device)
        features.append(feature)
    features = np.array(features)

    # Clustering avec DBSCAN (nombre de clusters automatique)
    print("Clustering en cours avec DBSCAN...")
    labels = cluster_images(features, eps=eps, min_samples=min_samples)

    # Déplacer les images
    move_images_to_clusters(image_paths, labels, dest_folder, handle_noise=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regrouper des images similaires dans des dossiers en utilisant DBSCAN.")
    parser.add_argument("source_folder", help="Chemin du dossier source contenant les images.")
    parser.add_argument("dest_folder", help="Chemin du dossier de destination pour les clusters.")
    parser.add_argument("--eps", type=float, default=0.5,
                        help="Paramètre eps pour DBSCAN (distance maximale pour la similarité, par défaut: 0.5).")
    parser.add_argument("--min_samples", type=int, default=2,
                        help="Paramètre min_samples pour DBSCAN (nombre minimal d'images pour former un cluster, par défaut: 2).")

    args = parser.parse_args()

    main(
        source_folder=args.source_folder,
        dest_folder=args.dest_folder,
        eps=args.eps,
        min_samples=args.min_samples
    )
