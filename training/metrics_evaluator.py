# metrics_evaluator.py
"""
Outil d'évaluation des performances pour les modèles de classification d'images.

Ce script fournit des utilitaires graphiques et textuels pour mesurer l'exactitude
d'un modèle par rapport à des vérités terrain (ground truth). Il trace notamment une
matrice de confusion avec Seaborn et affiche un rapport détaillé de classification
(précision, rappel, F1-score) via Scikit-Learn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def evaluate_model(classifier, test_images, test_labels) -> list:
    """
    Évalue un modèle de classification de qualité sur un ensemble d'images de test.
    Calcule et trace la matrice de confusion graphique et affiche le rapport textuel.

    Args:
        classifier: Instance du classificateur de qualité (doit implémenter predict_single_image).
        test_images (list): Liste des chemins d'accès vers les images de test.
        test_labels (list): Liste des classes de qualité réelles (vérité terrain) associées.

    Returns:
        list: Liste des prédictions générées par le modèle.
    """
    predictions = []
    
    # 1. Génération des prédictions pour chaque image de test
    for image_path in test_images:
        result = classifier.predict_single_image(image_path)
        # On extrait la classe prédite (de 0 à 5)
        predictions.append(result['predicted_quality'])
    
    # 2. Calcul mathématique de la matrice de confusion
    cm = confusion_matrix(test_labels, predictions)
    
    # 3. Tracé graphique interactif de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True,          # Affiche les valeurs numériques dans chaque cellule
        fmt='d',             # Format d'entier décimal
        cmap='Blues',        # Palette de couleurs bleues dégradées
        xticklabels=range(6),# Graduation des prédictions (0-5)
        yticklabels=range(6) # Graduation des vérités terrain (0-5)
    )
    plt.title('Matrice de confusion - Qualité Esthétique')
    plt.ylabel('Vraie qualité (Vérité terrain)')
    plt.xlabel('Qualité prédite par le modèle')
    plt.tight_layout()       # Ajustement automatique des marges
    plt.show()
    
    # 4. Génération du rapport analytique textuel détaillé
    # Génère précision, rappel (recall) et f1-score pour chaque classe
    report = classification_report(
        test_labels, 
        predictions, 
        target_names=[f'Qualité {i}' for i in range(6)]
    )
    print("\n=== RAPPORT D'ÉVALUATION DÉTAILLÉ ===")
    print(report)
    
    return predictions
