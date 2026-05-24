"""
Test d'intégration minimal pour le prédicteur esthétique externe.

Ce court script sert de test de fumée (smoke test) pour vérifier que la bibliothèque
externe 'aesthetic_predictor' est correctement installée, importable, et capable
de traiter et d'évaluer une image de démonstration au format standard JPEG.
"""

import os
from aesthetic_predictor import predict_aesthetic
from PIL import Image

def main():
    """
    Charge une image test réputée d'excellente qualité (quality_5)
    et affiche son score esthétique brut prédit par le modèle.
    """
    # Chemin d'accès relatif robuste vers l'image de test sous data/inputs/labeled_tiers/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_test_path = os.path.join(project_root, 'data', 'inputs', 'labeled_tiers', 'quality_5', 'Image-008.jpg')
    print(f"Lancement du test d'intégration sur : '{image_test_path}'...")
    
    try:
        # Chargement de l'image de test via Pillow
        img = Image.open(image_test_path)
        
        # Calcul de la note esthétique
        score = predict_aesthetic(img)
        print("[SUCCES] Test d'integration reussi !")
        print(f"Score esthetique brut predit : {score}")
    except FileNotFoundError:
        print(f"[ATTENTION] Le fichier image de test '{image_test_path}' est manquant.")
        print("Veuillez vous assurer que le dossier d'images de demonstration est present pour executer ce test.")
    except Exception as e:
        print(f"[ERREUR] Echec du test d'integration : {e}")

if __name__ == "__main__":
    main()
