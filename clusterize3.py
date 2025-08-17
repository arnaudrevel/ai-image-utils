import os
import shutil
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import imagehash  # ImageHash est un package séparé
from sklearn.cluster import DBSCAN
from collections import defaultdict

class ImageSimilarityGrouper:
    def __init__(self, source_dir, dest_dir, similarity_threshold=0.9):
        """
        Initialise le groupeur d'images similaires
        
        Args:
            source_dir: Répertoire source contenant les images
            dest_dir: Répertoire de destination pour les groupes
            similarity_threshold: Seuil de similarité (0-1, plus proche de 1 = plus similaire)
        """
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.similarity_threshold = similarity_threshold
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        
    def calculate_hash(self, image_path, hash_type='dhash'):
        """
        Calcule le hash d'une image
        
        Args:
            image_path: Chemin vers l'image
            hash_type: Type de hash ('dhash', 'phash', 'ahash', 'whash')
        """
        try:
            with Image.open(image_path) as img:
                # Convertir en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                if hash_type == 'dhash':
                    return imagehash.dhash(img)
                elif hash_type == 'phash':
                    return imagehash.phash(img)
                elif hash_type == 'ahash':
                    return imagehash.average_hash(img)
                elif hash_type == 'whash':
                    return imagehash.whash(img)
                else:
                    return imagehash.dhash(img)
        except Exception as e:
            print(f"Erreur lors du calcul du hash pour {image_path}: {e}")
            return None

    def get_image_files(self):
        """Récupère tous les fichiers d'images dans le répertoire source"""
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(self.source_dir.glob(f'**/*{ext}'))
            image_files.extend(self.source_dir.glob(f'**/*{ext.upper()}'))
        return image_files

    def calculate_similarity_matrix(self, image_files, hash_type='dhash'):
        """Calcule la matrice de similarité entre toutes les images"""
        print("Calcul des hashs des images...")
        
        # Calculer les hashs
        hashes = []
        valid_files = []
        
        for i, img_path in enumerate(image_files):
            if i % 50 == 0:
                print(f"Traitement: {i}/{len(image_files)}")
                
            img_hash = self.calculate_hash(img_path, hash_type)
            if img_hash is not None:
                hashes.append(img_hash)
                valid_files.append(img_path)
        
        print(f"Calcul de la matrice de similarité pour {len(valid_files)} images...")
        
        # Calculer la matrice de similarité
        n = len(hashes)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            if i % 100 == 0:
                print(f"Calcul similarité: {i}/{n}")
            for j in range(i, n):
                if i == j:
                    similarity = 1.0
                else:
                    # Calculer la similarité basée sur la distance de Hamming
                    hamming_distance = hashes[i] - hashes[j]
                    # Convertir en similarité (0-1, où 1 = identique)
                    max_distance = 64  # Pour la plupart des hashs (8x8 = 64 bits)
                    similarity = 1 - (hamming_distance / max_distance)
                
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix, valid_files

    def cluster_images(self, similarity_matrix, valid_files):
        """Groupe les images similaires en utilisant DBSCAN"""
        print("Regroupement des images similaires...")
        
        # Convertir la matrice de similarité en matrice de distance
        distance_matrix = 1 - similarity_matrix
        
        # Utiliser DBSCAN pour le clustering
        eps = 1 - self.similarity_threshold  # Convertir le seuil de similarité en distance
        clustering = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Organiser les images par cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(valid_files[i])
        
        return clusters

    def create_output_structure(self, clusters):
        """Crée la structure de répertoires et copie les images"""
        print("Création de la structure de sortie...")
        
        # Créer le répertoire de destination
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistiques
        total_clusters = len(clusters)
        single_image_clusters = sum(1 for cluster in clusters.values() if len(cluster) == 1)
        multi_image_clusters = total_clusters - single_image_clusters
        
        print(f"\nRésultats du regroupement:")
        print(f"- Nombre total de groupes: {total_clusters}")
        print(f"- Groupes avec plusieurs images similaires: {multi_image_clusters}")
        print(f"- Images uniques: {single_image_clusters}")
        
        # Créer les dossiers et copier les images
        for cluster_id, image_paths in clusters.items():
            if len(image_paths) == 1:
                # Image unique - copier dans le dossier "uniques"
                unique_dir = self.dest_dir / "images_uniques"
                unique_dir.mkdir(exist_ok=True)
                dest_path = unique_dir / image_paths[0].name
                
                # Éviter les doublons de noms
                counter = 1
                while dest_path.exists():
                    stem = image_paths[0].stem
                    suffix = image_paths[0].suffix
                    dest_path = unique_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                try:
                    shutil.copy2(image_paths[0], dest_path)
                except Exception as e:
                    print(f"Erreur lors de la copie de {image_paths[0]}: {e}")
            else:
                # Groupe d'images similaires
                cluster_dir = self.dest_dir / f"groupe_similaire_{cluster_id:03d}"
                cluster_dir.mkdir(exist_ok=True)
                
                print(f"\nGroupe {cluster_id}: {len(image_paths)} images similaires")
                for img_path in image_paths:
                    print(f"  - {img_path.name}")
                    dest_path = cluster_dir / img_path.name
                    
                    # Éviter les doublons de noms
                    counter = 1
                    while dest_path.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        dest_path = cluster_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    try:
                        shutil.copy2(img_path, dest_path)
                    except Exception as e:
                        print(f"Erreur lors de la copie de {img_path}: {e}")

    def process_images(self, hash_type='dhash'):
        """Traite toutes les images du répertoire source"""
        if not self.source_dir.exists():
            raise ValueError(f"Le répertoire source {self.source_dir} n'existe pas")
        
        # Obtenir tous les fichiers d'images
        image_files = self.get_image_files()
        if not image_files:
            print("Aucune image trouvée dans le répertoire source")
            return
        
        print(f"Trouvé {len(image_files)} images à traiter")
        
        # Calculer la matrice de similarité
        similarity_matrix, valid_files = self.calculate_similarity_matrix(image_files, hash_type)
        
        if len(valid_files) == 0:
            print("Aucune image valide à traiter")
            return
        
        # Regrouper les images
        clusters = self.cluster_images(similarity_matrix, valid_files)
        
        # Créer la structure de sortie
        self.create_output_structure(clusters)
        
        print(f"\nTraitement terminé! Les résultats sont dans: {self.dest_dir}")

def main():
    parser = argparse.ArgumentParser(description='Regrouper les images similaires')
    parser.add_argument('source', help='Répertoire source contenant les images')
    parser.add_argument('destination', help='Répertoire de destination pour les groupes')
    parser.add_argument('--threshold', '-t', type=float, default=0.9,
                      help='Seuil de similarité (0-1, défaut: 0.9)')
    parser.add_argument('--hash-type', '-ht', choices=['dhash', 'phash', 'ahash', 'whash'],
                      default='dhash', help='Type de hash à utiliser (défaut: dhash)')
    
    args = parser.parse_args()
    
    # Valider le seuil
    if not 0 <= args.threshold <= 1:
        print("Erreur: Le seuil de similarité doit être entre 0 et 1")
        return
    
    try:
        grouper = ImageSimilarityGrouper(args.source, args.destination, args.threshold)
        grouper.process_images(args.hash_type)
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()
