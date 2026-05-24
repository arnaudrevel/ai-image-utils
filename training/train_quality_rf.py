"""
Entraîneur de modèle de qualité d'images basé sur des caractéristiques OpenCV et un Random Forest.

Ce script implémente une approche classique de vision par ordinateur (Computer Vision).
Il extrait des descripteurs mathématiques bas-niveau représentatifs de la qualité
esthétique et technique d'une image (netteté via le Laplacien, contraste via l'écart-type,
luminosité, entropie de l'information et gradients de Sobel) et entraîne un régresseur
Forêt Aléatoire (Random Forest) de Scikit-Learn pour estimer le score de qualité.
"""

import os
import cv2
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


def extract_features(image_path: str) -> list:
    """
    Extrait les caractéristiques mathématiques d'une image pour l'évaluation de qualité.

    Les caractéristiques extraites sont :
    1. Netteté (Variance du Laplacien) : Évalue la présence de détails fins.
    2. Contraste (Écart-type des pixels) : Analyse la dynamique des tons sombres/clairs.
    3. Luminosité moyenne : Calcule l'exposition globale de l'image.
    4. Statistiques de l'histogramme : Moyenne et écart-type des fréquences de gris.
    5. Netteté directionnelle (Sobel) : Mesure la force des contours verticaux et horizontaux.
    6. Entropie : Mesure la quantité d'information et de complexité visuelle de l'image.

    Args:
        image_path (str): Chemin d'accès vers le fichier image.

    Returns:
        list: Liste de nombres flottants représentant les caractéristiques, ou None si échec.
    """
    # Lecture de l'image via OpenCV (BGR)
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Conversion de l'image en niveaux de gris (simplification des calculs d'intensité)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1. Netteté (variance du Laplacien)
    # L'opérateur Laplacien calcule les dérivées secondes spatiales de l'image.
    # Une image nette avec des contours marqués aura une variance très élevée.
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(laplacian_var)
    
    # 2. Contraste (écart-type des pixels)
    # Plus l'écart-type de la distribution des pixels est grand, plus l'image est contrastée.
    contrast = gray.std()
    features.append(contrast)
    
    # 3. Luminosité moyenne
    # La valeur moyenne des pixels donne une estimation directe de la sur/sous-exposition.
    brightness = gray.mean()
    features.append(brightness)
    
    # 4. Histogramme - Caractéristiques statistiques de la répartition des teintes de gris
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_mean = hist.mean()
    hist_std = hist.std()
    features.extend([hist_mean, hist_std])
    
    # 5. Gradient de Sobel pour la netteté et la force des contours
    # Calcule les gradients horizontaux (x) et verticaux (y).
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Calcul de la magnitude moyenne au carré du gradient global
    sobel_mean = (sobelx**2 + sobely**2).mean()
    features.append(sobel_mean)
    
    # 6. Entropie (mesure de la quantité d'information visuelle)
    # Une image uniforme a une entropie proche de 0, une image complexe et riche a une entropie élevée.
    hist_norm = hist / hist.sum()
    # Ajout d'une constante infinitésimale (1e-10) pour éviter l'erreur mathématique log2(0)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
    features.append(entropy)
    
    return features


def train_quality_model(dataset_path: str, model_output_path: str = "quality_model.pkl") -> RandomForestRegressor:
    """
    Orchestre l'extraction des caractéristiques et l'entraînement du Random Forest.

    Ce script suppose que la vérité terrain (le score esthétique) est codée dans le nom
    du fichier d'entraînement (ex: 'image3.jpg' correspond à un score cible de 2).

    Args:
        dataset_path (str): Dossier contenant les images d'entraînement.
        model_output_path (str): Fichier pickle cible pour enregistrer le modèle.

    Returns:
        RandomForestRegressor: Modèle entraîné.
    """
    print("Collecte des données d'entraînement et extraction des caractéristiques OpenCV...")
    
    X = []  # Vecteurs de caractéristiques
    y = []  # Labels de qualité (vérité terrain)
    
    # Parcourir le répertoire d'images
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(dataset_path, filename)
            
            # Extraction du niveau de qualité basé sur le chiffre présent dans le nom de fichier
            try:
                base_name = os.path.splitext(filename)[0]
                # Extrait tous les caractères numériques présents dans le nom
                image_num = int(''.join(filter(str.isdigit, base_name)))
                # Règle empirique de mapping : image1 -> qualité 0, image6 -> qualité 5
                quality = min(image_num - 1, 5)
                quality = max(quality, 0)
            except Exception:
                continue
            
            # Extraction des descripteurs
            features = extract_features(image_path)
            if features is not None:
                X.append(features)
                y.append(quality)
    
    if len(X) == 0:
        print("❌ Aucune image d'entraînement valide trouvée.")
        return None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Données collectées : {len(X)} images")
    print(f"Répartition par note : {np.bincount(y.astype(int))}")
    
    # Division 80% Entraînement / 20% Test pour l'évaluation de généralisation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du régresseur Forêt Aléatoire
    print("Entraînement du modèle RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Évaluation statistique sur le jeu de test
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Erreur quadratique moyenne (MSE) : {mse:.3f}")
    print(f"Score R² (Coefficient de dét.)  : {r2:.3f}")
    
    # Sérialisation du modèle entraîné via Pickle
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"💾 Modèle de régression sauvegardé dans: `{model_output_path}`")
    return model


def predict_image_quality(images_directory: str, model_path: str = "quality_model.pkl", output_csv: str = "quality_predictions.csv"):
    """
    Charge le modèle de régression RandomForest et estime la qualité des images d'un dossier.
    Génère un rapport de classement au format CSV.

    Args:
        images_directory (str): Dossier contenant les nouvelles images à évaluer.
        model_path (str): Fichier pickle du modèle pré-entraîné.
        output_csv (str): Nom du fichier CSV de sortie pour enregistrer les prédictions.
    """
    # 1. Chargement asynchrone du modèle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Modèle de régression chargé depuis : `{model_path}`")
    except FileNotFoundError:
        print(f"❌ Modèle introuvable à l'adresse : `{model_path}`")
        return
    
    results = []
    
    print("Inférence et prédiction de la qualité des images...")
    
    # Parcourir et traiter les images du dossier cible
    for filename in os.listdir(images_directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(images_directory, filename)
            
            features = extract_features(image_path)
            
            if features is not None:
                # Redimensionnement des caractéristiques pour correspondre à l'entrée attendue par scikit-learn (1, -1)
                features_array = np.array(features).reshape(1, -1)
                # Prédiction continue (flottante)
                predicted_quality = model.predict(features_array)[0]
                
                # Limitation rigoureuse de la note prédite entre 0.0 et 5.0 et arrondi à 2 décimales
                predicted_quality = max(0.0, min(5.0, round(predicted_quality, 2)))
                
                results.append({
                    'filename': filename,
                    'predicted_quality': predicted_quality,
                    'quality_category': f"Qualité {int(round(predicted_quality))}"
                })
                
                print(f"-> {filename}: Qualité estimée = {predicted_quality}/5")
    
    # 2. Sauvegarde sous format CSV ordonné du plus qualitatif au moins qualitatif
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('predicted_quality', ascending=False)
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Résultats triés sauvegardés dans: `{output_csv}`")
        print(f"Nombre total d'images analysées: {len(results)}")
    else:
        print("⚠️ Aucune image valide n'a pu être prédite.")


# Exemple d'utilisation du script
if __name__ == "__main__":
    # Chemin d'accès relatif vers le dossier d'images étiquetées sous data/inputs/labeled_tiers (robuste)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'data', 'inputs', 'labeled_tiers')
    print("=== ÉTAPE 1 : ENTRAÎNEMENT DU MODÈLE ===")
    if os.path.exists(dataset_path):
        train_quality_model(dataset_path)
    else:
        print(f"Dossier d'entraînement '{dataset_path}' introuvable. Étape 1 ignorée.")
        
    # Étape 2: Inférence et prédiction sur de nouvelles images
    test_images_path = "path/to/test/images"
    print("\n=== ÉTAPE 2 : PRÉDICTION SUR NOUVELLES IMAGES ===")
    if os.path.exists(test_images_path):
        predict_image_quality(test_images_path)
    else:
        print(f"Dossier de test '{test_images_path}' introuvable. Étape 2 ignorée.")
        
    print("\nProcessus terminé.")
