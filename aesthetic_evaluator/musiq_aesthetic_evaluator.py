"""
Évaluation esthétique des images à l'aide de la métrique MUSIQ (PyIQA).

Ce script utilise la bibliothèque PyIQA (Python Image Quality Assessment) pour
évaluer la qualité esthétique d'une image en s'appuyant sur le modèle MUSIQ 
(Multi-scale Image Quality Assessor). Le modèle MUSIQ est particulièrement performant
pour évaluer des images à résolutions variables en préservant leurs détails multi-échelles.

Prérequis :
    - PyIQA (pip install pyiqa)
    - PyTorch (torch)

Usage :
    python analysis/musiq_aesthetic_evaluator.py [chemin_image]
"""

import sys
import os
import pyiqa
import torch

def main():
    # Détermination du fichier image à évaluer
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Chemin par défaut vers l'image de démonstration starry_night.jpg sous data/inputs/labeled_tiers/ (robuste)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        image_path = os.path.join(project_root, 'data', 'inputs', 'labeled_tiers', 'starry_night.jpg')

    # Configuration du périphérique d'exécution : GPU (CUDA) si disponible, sinon CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialisation du modèle d'évaluation de la qualité d'image (MUSIQ)
    print(f"Chargement de la métrique MUSIQ sur le périphérique : {device}...")
    try:
        metric = pyiqa.create_metric('musiq', device=device)
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle MUSIQ : {e}")
        print("Veuillez vérifier que pyiqa et torch sont installés correctement.")
        return

    # Vérification de l'existence de l'image de test
    if not os.path.exists(image_path):
        print(f"Image cible introuvable : {image_path}")
        print("Veuillez fournir un chemin d'image valide en argument.")
        return

    # Évaluation esthétique de l'image spécifiée
    print(f"Évaluation de l'image : {image_path} ...")
    try:
        score = metric(image_path)
        # score est un tenseur PyTorch, on extrait la valeur scalaire avec .item()
        val_score = score.item() if hasattr(score, 'item') else float(score)
        print(f"\n==========================================")
        print(f"[SUCCES] Evaluation MUSIQ reussie !")
        print(f"Chemin : {image_path}")
        print(f"Score esthetique MUSIQ : {val_score:.4f}")
        print(f"==========================================\n")
    except Exception as e:
        print(f"Erreur lors de l'évaluation de l'image : {e}")

if __name__ == "__main__":
    main()
