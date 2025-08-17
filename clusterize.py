import os
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm
from collections import defaultdict

def extract_features(image_path, model, transform, device):
    """Extraire les features d'une image usando un modèle pré-entraîné."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image)

    # Normaliser les features (pour la similarité cosinus)
    features = F.normalize(features, p=2, dim=1)
    return features.cpu().numpy().flatten()

def load_images_from_folder(folder):
    """Charger les chemins des images depuis un dossier."""
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            images.append(os.path.join(folder, filename))
    return images

def cluster_images(features, method='dbscan', eps=0.5, min_samples=2, n_clusters=5):
    """Regrouper les images en fonction de leurs features."""
    if method == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(features)
    elif method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    else:
        raise ValueError("Méthode de clustering non supportée. Utilisez 'dbscan' ou 'kmeans'.")

    return clustering.labels_

def move_images_to_clusters(image_paths, labels, dest_folder):
    """Déplacer les images dans des dossiers de destination en fonction des labels."""
    # Créer un dictionnaire pour mapper les labels aux chemins
    label_to_images = defaultdict(list)
    for img_path, label in zip(image_paths, labels):
        label_to_images[label].append(img_path)

    # Créer les dossiers de destination et déplacer les images
    for label, images in label_to_images.items():
        cluster_folder = os.path.join(dest_folder, f"cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)

        for img_path in images:
            shutil.copy(img_path, os.path.join(cluster_folder, os.path.basename(img_path)))

    print(f"Images regroupées dans {len(label_to_images)} clusters sous {dest_folder}")

def main(source_folder, dest_folder, method='dbscan', eps=0.5, min_samples=2, n_clusters=5):
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

    # Clustering
    print("Clustering en cours...")
    labels = cluster_images(features, method=method, eps=eps, min_samples=min_samples, n_clusters=n_clusters)

    # Déplacer les images
    move_images_to_clusters(image_paths, labels, dest_folder)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regrouper des images similaires dans des dossiers.")
    parser.add_argument("source_folder", help="Chemin du dossier source contenant les images.")
    parser.add_argument("dest_folder", help="Chemin du dossier de destination pour les clusters.")
    parser.add_argument("--method", choices=['dbscan', 'kmeans'], default='dbscan',
                        help="Méthode de clustering (par défaut: dbscan).")
    parser.add_argument("--eps", type=float, default=0.5,
                        help="Paramètre eps pour DBSCAN (par défaut: 0.5).")
    parser.add_argument("--min_samples", type=int, default=2,
                        help="Paramètre min_samples pour DBSCAN (par défaut: 2).")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Nombre de clusters pour K-Means (par défaut: 5).")

    args = parser.parse_args()

    main(
        source_folder=args.source_folder,
        dest_folder=args.dest_folder,
        method=args.method,
        eps=args.eps,
        min_samples=args.min_samples,
        n_clusters=args.n_clusters
    )
