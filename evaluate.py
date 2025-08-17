# evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def evaluate_model(classifier, test_images, test_labels):
    """Évalue le modèle sur un jeu de test"""
    predictions = []
    
    for image_path in test_images:
        result = classifier.predict_single_image(image_path)
        predictions.append(result['predicted_quality'])
    
    # Matrice de confusion
    cm = confusion_matrix(test_labels, predictions)
    
    # Affichage
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(6), yticklabels=range(6))
    plt.title('Matrice de confusion - Qualité esthétique')
    plt.ylabel('Vraie qualité')
    plt.xlabel('Qualité prédite')
    plt.show()
    
    # Rapport détaillé
    report = classification_report(test_labels, predictions, 
                                 target_names=[f'Qualité {i}' for i in range(6)])
    print(report)
    
    return predictions
