"""
Entraîneur de modèles ViT pour la classification de la qualité esthétique d'images.

Ce module encapsule l'intégralité du pipeline d'apprentissage profond pour fine-tuner
des modèles de vision Transformers (ViT de Google ou équivalents) sur des tâches d'évaluation
d'images. Il gère la lecture des données réparties dans des sous-dossiers par niveau (0 à 5),
le prétraitement via Hugging Face Datasets, le calcul de métriques (Accuracy, MSE, RMSE)
et l'entraînement managé à l'aide de la classe Trainer.
"""

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
    """
    Classificateur et entraîneur basé sur l'architecture Vision Transformer (ViT).
    S'occupe de l'import des poids pré-entraînés, de la tokenisation des pixels
    et de l'entraînement supervisé pour classifier la qualité d'images en 6 classes.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Initialise l'entraîneur.

        Args:
            model_name (str): Nom du modèle sur le Hugging Face Hub ou chemin local du modèle.
        """
        self.model_name = model_name
        self.num_labels = 6  # Classes allant de 0 (Très mauvaise) à 5 (Excellente)
        self.image_processor = None
        self.model = None
        
    def load_model(self):
        """
        Charge en mémoire le processeur d'images de ViT (rescaling, normalisation) et
        le modèle de classification. Modifie dynamiquement la tête linéaire finale (classifier)
        pour correspondre aux 6 classes cibles.
        """
        # Le processeur gère le redimensionnement automatique en 224x224 et la normalisation d'ImageNet
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        # Chargement du modèle avec instruction d'ignorer la taille de la couche de classification finale d'origine
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Remplacement explicite de la couche linéaire finale de classification
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, 
                self.num_labels
            )
    
    def preprocess_images(self, image_paths: List[str], labels: List[int]) -> Dataset:
        """
        Prépare les images et génère un Dataset Hugging Face optimisé pour l'entraînement.

        Args:
            image_paths (List[str]): Chemins d'accès des fichiers images.
            labels (List[int]): Liste des notes (labels) de qualité (0 à 5).

        Returns:
            Dataset: Dataset PyTorch pré-mappé contenant les valeurs de pixels d'images.
        """
        def load_and_process_image(example):
            """Fonction interne exécutée par lot (map) pour charger et transformer les images en tenseurs."""
            try:
                image = Image.open(example['image_path']).convert('RGB')
                # Préparation spatiale de l'image (224x224 pixels standard pour ViT)
                inputs = self.image_processor(image, return_tensors="pt")
                # squeeze() élimine la dimension de batch (1) pour stocker les pixel_values individuelles
                example['pixel_values'] = inputs['pixel_values'].squeeze()
            except Exception as e:
                # Gestion de secours en cas d'image corrompue : on génère un tenseur de zéros
                print(f"Erreur de lecture sur l'image {example['image_path']}: {e}")
                example['pixel_values'] = torch.zeros((3, 224, 224))
            return example
        
        # Création d'un dictionnaire initial pour le Dataset
        dataset_dict = {
            'image_path': image_paths,
            'labels': labels
        }
        
        # Conversion en Dataset Hugging Face et cartographie (mapping)
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(load_and_process_image, remove_columns=['image_path'])
        
        return dataset
    
    def create_train_val_split(self, dataset: Dataset, test_size: float = 0.2) -> DatasetDict:
        """
        Divise de manière aléatoire le Dataset en jeux d'entraînement et de validation.

        Args:
            dataset (Dataset): Le jeu complet.
            test_size (float): Proportion d'images allouée à la validation (ex: 0.2 pour 20%).

        Returns:
            DatasetDict: Dictionnaire contenant les splits 'train' et 'validation'.
        """
        dataset = dataset.train_test_split(test_size=test_size)
        return DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test']
        })
    
    def compute_metrics(self, eval_pred) -> Dict:
        """
        Fonction callback de calcul de métriques qualitatives durant l'évaluation.

        Args:
            eval_pred: Tuple contenant les prédictions (logits) et les vrais labels.

        Returns:
            Dict: Dictionnaire des métriques (Accuracy, MSE pour l'erreur de distance, RMSE).
        """
        predictions, labels = eval_pred
        # argmax récupère l'indice ayant la probabilité la plus élevée
        predictions = np.argmax(predictions, axis=1)
        
        # Calcul des statistiques d'évaluation
        accuracy = accuracy_score(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'rmse': np.sqrt(mse)  # Racine de l'erreur quadratique moyenne pour exprimer la distance d'erreur sur l'échelle 0-5
        }
    
    def train(self, 
              train_dataset: Dataset, 
              val_dataset: Dataset,
              output_dir: str = "./aesthetic-classifier",
              num_epochs: int = 10,
              batch_size: int = 16,
              learning_rate: float = 2e-5) -> Trainer:
        """
        Configure les hyperparamètres et lance la boucle d'entraînement du modèle.

        Args:
            train_dataset: Dataset d'entraînement.
            val_dataset: Dataset de validation.
            output_dir (str): Dossier local où seront sauvegardés le modèle et ses checkpoints.
            num_epochs (int): Nombre d'époques d'apprentissage complet.
            batch_size (int): Taille de lot (batch size) pour l'optimisation par gradient.
            learning_rate (float): Vitesse d'apprentissage de l'optimiseur AdamW.

        Returns:
            Trainer: L'objet d'entraînement Hugging Face Trainer instancié.
        """
        
        # Configuration détaillée des paramètres d'entraînement (TrainingArguments)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,                  # Montée en température linéaire du taux d'apprentissage
            weight_decay=0.01,                 # Régularisation L2 pour éviter le surapprentissage
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="epoch",             # Évaluation effectuée à la fin de chaque époque
            save_strategy="epoch",             # Checkpoint enregistré à la fin de chaque époque
            load_best_model_at_end=True,       # Recharge le meilleur modèle trouvé lors de la validation à la fin
            metric_for_best_model="accuracy",  # Métrique de sélection du meilleur checkpoint
            greater_is_better=True,
            learning_rate=learning_rate,
            save_total_limit=2,                # Conserver au maximum les 2 meilleurs checkpoints
        )
        
        # Instanciation de l'orchestrateur Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Lancement effectif de l'entraînement asynchrone / synchrone
        print("🚀 Démarrage du fine-tuning du modèle Vision Transformer...")
        trainer.train()
        
        # Sauvegarde finale sur le disque dur
        trainer.save_model()
        self.image_processor.save_pretrained(output_dir)
        print(f"✓ Entraînement terminé. Modèle sauvegardé dans : `{output_dir}`")
        
        return trainer
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Réalise une estimation esthétique rapide sur une image.

        Args:
            image_path (str): Chemin de l'image.

        Returns:
            Dict: Résultats (classe prédite, score de confiance, probabilités détaillées).
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor(image, return_tensors="pt")
        
        # Transfert temporaire sur GPU si le modèle y est déjà situé
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
        
        return {
            'predicted_quality': predicted_class,
            'confidence': confidence,
            'all_probabilities': predictions.squeeze().cpu().tolist()
        }
    
    def load_trained_model(self, model_path: str):
        """
        Charge en mémoire les poids d'un classificateur précédemment entraîné localement.

        Args:
            model_path (str): Dossier contenant les fichiers config.json et pytorch_model.bin du modèle.
        """
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)


def organize_data_from_folder(base_folder: str) -> Tuple[List[str], List[int]]:
    """
    Parcourt une arborescence ordonnée de dossiers de qualité pour collecter
    les chemins et les classes correspondantes.

    Structure attendue sous base_folder :
    base_folder/
    ├── quality_0/ (Très mauvaises images)
    ├── quality_1/
    ├── quality_2/
    ├── quality_3/
    ├── quality_4/
    └── quality_5/ (Images d'excellence)

    Args:
        base_folder (str): Dossier racine de la base de données.

    Returns:
        Tuple[List[str], List[int]]: Chemins d'accès complets et liste des labels associés.
    """
    image_paths = []
    labels = []
    
    for quality in range(6):  # Classes 0 à 5
        folder_path = os.path.join(base_folder, f"quality_{quality}")
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                # Filtrage strict sur les formats d'images
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(quality)
        else:
            print(f"Avertissement : Sous-dossier absent : '{folder_path}'")
    
    return image_paths, labels


def main():
    """
    Point d'entrée de test d'entraînement.
    Parcourt un dossier d'images local et lance un fine-tuning de démonstration.
    """
    # Chemin d'accès relatif vers le dossier d'images étiquetées sous data/inputs/labeled_tiers
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    pathtoimg = os.path.join(project_root, 'data', 'inputs', 'labeled_tiers')

    if not os.path.exists(pathtoimg):
        print(f"❌ Dossier d'images absent à l'emplacement : '{pathtoimg}'. Veuillez adapter le chemin avant de lancer.")
        return

    # Initialisation du classificateur ViT
    classifier = AestheticQualityClassifier()
    classifier.load_model()
    
    # Lecture et indexation des dossiers de qualité 0 à 5
    image_paths, labels = organize_data_from_folder(pathtoimg)
    print(f"📁 Données d'entraînement indexées : {len(image_paths)} images trouvées.")
    
    if len(image_paths) == 0:
        print("❌ Aucune image à entraîner.")
        return

    # Préparation des tenseurs du Dataset
    dataset = classifier.preprocess_images(image_paths, labels)
    # Division 70% Entraînement / 30% Validation
    datasets = classifier.create_train_val_split(dataset, test_size=0.3)
    
    # Lancement du processus d'entraînement
    classifier.train(
        datasets['train'], 
        datasets['validation'],
        num_epochs=10,
        batch_size=8,
        learning_rate=2e-5
    )


if __name__ == "__main__":
    main()
