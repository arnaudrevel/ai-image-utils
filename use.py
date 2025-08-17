# Exemple d'utilisation complète
if __name__ == "__main__":
    # 1. Préparation des données
    base_folder = "path/to/your/organized/data"
    image_paths, labels = organize_data_from_folder(base_folder)
    
    # 2. Initialisation et entraînement
    classifier = AestheticQualityClassifier("google/vit-base-patch16-224")
    classifier.load_model()
    
    # 3. Préparation du dataset
    dataset = classifier.preprocess_images(image_paths, labels)
    datasets = classifier.create_train_val_split(dataset, test_size=0.2)
    
    # 4. Entraînement
    trainer = classifier.train(
        datasets['train'], 
        datasets['validation'],
        output_dir="./aesthetic-quality-model",
        num_epochs=15,
        batch_size=16
    )
    
    # 5. Test sur de nouvelles images
    test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]
    
    for img_path in test_images:
        result = classifier.predict_single_image(img_path)
        print(f"Image: {img_path}")
        print(f"Qualité esthétique: {result['predicted_quality']}/5")
        print(f"Confiance: {result['confidence']:.3f}")
        print("---")
