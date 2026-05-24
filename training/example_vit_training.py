"""
Exemple d'utilisation complet du pipeline d'entraînement ViT pour la qualité esthétique.

Ce script montre la mise en œuvre pas-à-pas de la classe AestheticQualityClassifier :
1. Indexation et lecture des dossiers de qualité (quality_0 à quality_5).
2. Instanciation du classificateur basé sur ViT pré-entraîné.
3. Prétraitement et création du jeu d'évaluation train/validation (80/20).
4. Lancement de l'entraînement (fine-tuning) supervisé.
5. Inférence et prédictions sur de nouvelles images de test.

Note : Les imports manquants ont été rajoutés pour rendre ce script autonome et opérationnel.
"""

# Importation des composants clés du module d'entraînement ViT
from train_quality_vit import AestheticQualityClassifier, organize_data_from_folder


if __name__ == "__main__":
    # 1. Préparation et indexation des données d'entraînement
    # Adaptez ce chemin vers le dossier racine contenant vos sous-dossiers 'quality_0' à 'quality_5'
    base_folder = "path/to/your/organized/data"
    
    print(f"Indexation des sous-dossiers sous : '{base_folder}'...")
    # organize_data_from_folder renvoie la liste des chemins et les notes (labels) associés
    image_paths, labels = organize_data_from_folder(base_folder)
    
    # 2. Initialisation et chargement de la structure du modèle ViT
    # Nous utilisons par défaut le modèle 'vit-base-patch16-224' pré-entraîné par Google
    classifier = AestheticQualityClassifier("google/vit-base-patch16-224")
    classifier.load_model()
    
    # 3. Préparation et découpage du dataset
    # Convertit les images physiques en tenseurs utilisables par ViT
    dataset = classifier.preprocess_images(image_paths, labels)
    
    # Séparation robuste avec 20% des images réservées pour la validation/évaluation
    datasets = classifier.create_train_val_split(dataset, test_size=0.2)
    
    # 4. Inscription des paramètres et exécution de l'entraînement supervisé
    # Les poids ajustés seront sauvegardés dans le dossier "./aesthetic-quality-model"
    trainer = classifier.train(
        datasets['train'], 
        datasets['validation'],
        output_dir="./aesthetic-quality-model",
        num_epochs=15,    # Nombre d'époques d'apprentissage
        batch_size=16     # Taille des lots traités par étape de gradient
    )
    
    # 5. Inférence : Évaluation et test sur de nouvelles images inédites
    test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]
    
    print("\n=== PRÉVISIONS SUR IMAGES DE TEST ===")
    for img_path in test_images:
        try:
            # Réalisation d'une estimation esthétique
            result = classifier.predict_single_image(img_path)
            print(f"Image   : {img_path}")
            print(f"Qualité esthétique estimée : {result['predicted_quality']}/5")
            print(f"Confiance du modèle        : {result['confidence']:.1%}")
            print("---")
        except FileNotFoundError:
            print(f"Image {img_path} non trouvée sur le disque. Test d'inférence ignoré.")
        except Exception as e:
            print(f"Erreur d'inférence sur {img_path}: {e}")
