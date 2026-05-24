"""
Évaluation esthétique des images à l'aide du modèle NIMA (Neural Image Assessment).

Ce module implémente le modèle NIMA avec une architecture basée sur MobileNet.
Il permet d'évaluer la qualité esthétique d'une image individuelle ou de toutes
les images d'un dossier. Le modèle NIMA prédit une distribution de probabilités 
sur 10 notes (de 1 à 10), à partir de laquelle on calcule la note moyenne (score) 
et l'écart-type (qui représente le niveau de consensus ou d'incertitude).

Prérequis :
    - TensorFlow / Keras
    - Pillow (PIL)
    - NumPy
    - Un fichier de poids pré-entraîné nommé 'mobilenet_weights.h5' dans le même répertoire.

Usage :
    python analysis/nima_aesthetic_evaluator.py <chemin_image_ou_dossier>
"""

import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

def build_model():
    """
    Construit le modèle NIMA en utilisant MobileNet comme extracteur de caractéristiques (backbone).

    Le modèle original MobileNet est tronqué (sans sa couche de classification supérieure).
    On y ajoute une couche de Dropout pour éviter le surapprentissage, suivie d'une couche 
    Dense avec une activation softmax pour prédire une distribution de probabilité sur 
    10 classes (représentant les notes de 1 à 10).

    Retourne :
        keras.Model : Le modèle NIMA compilé/prêt à recevoir les poids.
    """
    # Chargement du modèle de base MobileNet sans les couches de classification finales (include_top=False)
    # avec un pooling moyen global pour obtenir un vecteur de caractéristiques à plat.
    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights=None)
    
    # Couche de régularisation Dropout (taux élevé de 0.75 comme recommandé dans le papier NIMA)
    x = Dropout(0.75)(base_model.output)
    
    # Couche finale de classification pour estimer la probabilité de chaque note de 1 à 10
    x = Dense(10, activation='softmax')(x)
    
    # Création du modèle final connectant l'entrée de MobileNet à notre sortie de classification
    model = Model(base_model.input, x)
    return model

def load_image(img_path):
    """
    Charge, convertit et pré-traite une image pour la rendre compatible avec MobileNet.

    Les étapes de traitement incluent :
        1. Lecture de l'image.
        2. Conversion en espace de couleurs RGB si nécessaire.
        3. Redimensionnement aux dimensions attendues par MobileNet (224x224 pixels).
        4. Conversion en tableau NumPy de type float32.
        5. Normalisation spécifique à MobileNet (preprocess_input).
        6. Ajout d'une dimension pour le batch (dimension 0).

    Paramètres :
        img_path (str) : Chemin d'accès vers le fichier image.

    Retourne :
        np.ndarray : L'image pré-traitée sous forme de tenseur 4D de forme (1, 224, 224, 3).
    """
    # Ouverture du fichier image
    img = Image.open(img_path)
    
    # Forcer la conversion en RGB (gestion des images en niveaux de gris ou RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionnement bilinéaire vers la taille d'entrée du réseau (224x224)
    img = img.resize((224, 224), resample=Image.BILINEAR)
    
    # Conversion en tableau NumPy de type float32
    img = np.array(img).astype('float32')
    
    # Pré-traitement spécifique requis par MobileNet (normalisation des pixels entre -1 et 1)
    img = preprocess_input(img)
    
    # Expansion des dimensions pour simuler un batch de taille 1
    return np.expand_dims(img, axis=0)

def calculate_mean_score(scores):
    """
    Calcule la note esthétique moyenne (l'espérance mathématique) à partir de la distribution.

    NIMA prédit une distribution de probabilité sur les notes de 1 à 10.
    La note moyenne est la somme des produits de chaque note par sa probabilité associée.

    Paramètres :
        scores (np.ndarray) : Tableau de probabilités de taille 10 (somme = 1.0).

    Retourne :
        float : Le score esthétique moyen calculé (compris entre 1.0 et 10.0).
    """
    # Création du vecteur des notes possibles : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    si = np.arange(1, 11)
    
    # Calcul de la moyenne pondérée (Espérance = somme(note_i * p_i))
    mean = np.sum(scores * si)
    return mean

def process_path(path, model):
    """
    Parcourt et évalue esthétiquement un fichier unique ou l'ensemble des images d'un dossier.

    Parcourt récursivement le chemin fourni pour identifier toutes les images valides,
    puis utilise le modèle NIMA pour obtenir leurs scores et écarts-types.

    Paramètres :
        path (str) : Chemin vers un fichier image ou un dossier contenant des images.
        model (keras.Model) : Le modèle NIMA chargé avec ses poids.

    Retourne :
        list[dict] : Une liste de dictionnaires contenant pour chaque image :
                     'name' (nom de fichier), 'path' (chemin complet),
                     'score' (note moyenne), 'std' (écart-type).
    """
    # Extensions d'images prises en charge
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []

    # Vérifier si le chemin est un fichier ou un dossier
    if os.path.isfile(path):
        if path.lower().endswith(valid_exts):
            image_paths.append(path)
    elif os.path.isdir(path):
        # Parcours récursif de l'arborescence du répertoire
        for root, _, files in os.walk(path):
            # Exclusion des dossiers virtuels ou cachés typiques (.venv, .git)
            if '.venv' in root or '.git' in root:
                continue
            for f in files:
                if f.lower().endswith(valid_exts):
                    image_paths.append(os.path.join(root, f))
    
    if not image_paths:
        print(f"Aucune image valide trouvée à l'emplacement : {path}")
        return []

    results = []
    print(f"{len(image_paths)} image(s) trouvée(s). Début de l'évaluation esthétique...")

    # Traitement séquentiel de chaque image
    for img_path in image_paths:
        try:
            # 1. Chargement et pré-traitement de l'image
            img_array = load_image(img_path)
            
            # 2. Prédiction de la distribution de probabilité
            prediction = model.predict(img_array, verbose=0)
            scores = prediction[0]
            
            # 3. Calcul du score moyen
            mean_score = calculate_mean_score(scores)
            
            # 4. Calcul de l'écart-type (variance = somme( (note_i - moyenne)^2 * probabilité_i ))
            std_dev = np.sqrt(np.sum(((np.arange(1, 11) - mean_score) ** 2) * scores))
            
            results.append({
                'name': os.path.basename(img_path),
                'path': img_path,
                'score': mean_score,
                'std': std_dev
            })
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {img_path}: {e}")

    return results

def main():
    """
    Point d'entrée principal de l'outil d'évaluation NIMA.

    Vérifie les arguments, charge le modèle et ses poids, exécute l'analyse
    puis affiche les résultats triés par score esthétique décroissant sous forme de tableau.
    """
    # Validation des arguments en ligne de commande
    if len(sys.argv) < 2:
        print("Usage: python analysis/nima_aesthetic_evaluator.py <chemin_image_ou_repertoire>")
        return

    input_path = sys.argv[1]
    
    # Recherche des poids dans le dossier centralisé models/nima_weights/ (robuste et portable)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    weights_path = os.path.join(project_root, 'models', 'nima_weights', 'mobilenet_weights.h5')
    
    # Vérification de l'existence du fichier de poids pré-entraîné
    if not os.path.exists(weights_path):
        print(f"Fichier de poids introuvable : {weights_path}")
        print(f"Veuillez placer le fichier 'mobilenet_weights.h5' dans : {os.path.dirname(weights_path)}")
        return

    print("Chargement du modèle NIMA...")
    model = build_model()
    model.load_weights(weights_path)

    # Lancement de l'évaluation sur le chemin spécifié
    results = process_path(input_path, model)
    
    if not results:
        return

    # Tri des résultats du meilleur score au moins bon (ordre décroissant)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Affichage du tableau de synthèse des scores
    print("\n" + "="*65)
    print(f"{'Nom de l''image':<30} | {'Score':<7} | {'Écart-type':<10} | {'Évaluation'}")
    print("-" * 65)
    
    for res in results:
        # Classification subjective simplifiée basée sur le score moyen
        assessment = "Élevée (High)" if res['score'] > 6.5 else "Moyenne (Average)" if res['score'] > 4.5 else "Faible (Low)"
        print(f"{res['name'][:30]:<30} | {res['score']:>7.2f} | {res['std']:>10.2f} | {assessment}")
    
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
