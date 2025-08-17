import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    TrainingArguments, 
    Trainer
)
from datasets import Dataset, DatasetDict
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import os
from typing import Dict, List, Tuple

class AestheticQualityClassifier:
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialise le classificateur de qualité esthétique
        
        Args:
            model_name: Nom du modèle pré-entraîné à utiliser
        """
        self.model_name = model_name
        self.num_labels = 6  # Classes 0 à 5
        self.image_processor = None
        self.model = None
        
    def load_model(self):
        """Charge le modèle et le processeur d'images"""
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        # Configuration du modèle pour 6 classes
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Optionnel : modifier la couche de classification finale
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, 
                self.num_labels
            )
    
    def preprocess_images(self, image_paths: List[str], labels: List[int]):
        """
        Prétraite les images pour l'entraînement
        
        Args:
            image_paths: Liste des chemins vers les images
            labels: Liste des labels de qualité (0-5)
        """
        def load_and_process_image(example):
            image = Image.open(example['image_path']).convert('RGB')
            inputs = self.image_processor(image, return_tensors="pt")
            example['pixel_values'] = inputs['pixel_values'].squeeze()
            return example
        
        # Créer le dataset
        dataset_dict = {
            'image_path': image_paths,
            'labels': labels
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(load_and_process_image, remove_columns=['image_path'])
        
        return dataset
    
    def create_train_val_split(self, dataset, test_size: float = 0.2):
        """Divise le dataset en train/validation"""
        dataset = dataset.train_test_split(test_size=test_size)
        return DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test']
        })
    
    def compute_metrics(self, eval_pred):
        """Calcule les métriques d'évaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
    
    def train(self, 
              train_dataset, 
              val_dataset,
              output_dir: str = "./aesthetic-classifier",
              num_epochs: int = 10,
              batch_size: int = 16,
              learning_rate: float = 2e-5):
        """
        Entraîne le modèle
        
        Args:
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation
            output_dir: Répertoire de sortie pour sauvegarder le modèle
            num_epochs: Nombre d'époques
            batch_size: Taille du batch
            learning_rate: Taux d'apprentissage
        """
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            learning_rate=learning_rate,
            save_total_limit=2,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Entraînement
        trainer.train()
        
        # Sauvegarde du modèle final
        trainer.save_model()
        self.image_processor.save_pretrained(output_dir)
        
        return trainer
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Prédit la qualité esthétique d'une seule image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Dictionnaire avec la prédiction et les probabilités
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
        
        return {
            'predicted_quality': predicted_class,
            'confidence': confidence,
            'all_probabilities': predictions.squeeze().tolist()
        }
    
    def load_trained_model(self, model_path: str):
        """Charge un modèle pré-entraîné"""
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)

# Fonction utilitaire pour organiser vos données
def organize_data_from_folder(base_folder: str) -> Tuple[List[str], List[int]]:
    """
    Organise les données depuis des dossiers nommés par qualité
    
    Structure attendue:
    base_folder/
    ├── quality_0/
    ├── quality_1/
    ├── quality_2/
    ├── quality_3/
    ├── quality_4/
    └── quality_5/
    """
    image_paths = []
    labels = []
    
    for quality in range(6):  # 0 à 5
        folder_path = os.path.join(base_folder, f"quality_{quality}")
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(quality)
    
    return image_paths, labels

# Exemple d'utilisation
def main():
    # Initialisation du classificateur
    classifier = AestheticQualityClassifier()
    classifier.load_model()
    
    # # Exemple de données (remplacez par vos vrais chemins et labels)
    # image_paths = [
    #     "path/to/image1.jpg",  # qualité 0 (mauvaise)
    #     "path/to/image2.jpg",  # qualité 1
    #     "path/to/image3.jpg",  # qualité 2
    #     "path/to/image4.jpg",  # qualité 3
    #     "path/to/image5.jpg",  # qualité 4
    #     "path/to/image6.jpg",  # qualité 5 (excellente)
    #     # Ajoutez plus d'images...
    # ]
    
    # labels = [0, 1, 2, 3, 4, 5]  # Labels correspondants

    pathtoimg="C://Users//revel//OneDrive//Desktop//Qualité image//bestImages"

    image_paths, labels = organize_data_from_folder(pathtoimg)
    
    # Préparation des données
    dataset = classifier.preprocess_images(image_paths, labels)
    datasets = classifier.create_train_val_split(dataset, test_size=0.2)
    
    # Entraînement
    trainer = classifier.train(
        datasets['train'], 
        datasets['validation'],
        num_epochs=10,
        batch_size=8,
        learning_rate=2e-5
    )
    
    # Test sur une nouvelle image
    # result = classifier.predict_single_image("path/to/test_image.jpg")
    # print(f"Qualité prédite: {result['predicted_quality']}/5")
    # print(f"Confiance: {result['confidence']:.3f}")



if __name__ == "__main__":
    main()
