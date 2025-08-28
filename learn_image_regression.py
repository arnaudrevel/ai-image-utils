import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor, 
    AutoModel,
    TrainingArguments, 
    Trainer
)
from transformers.modeling_outputs import ModelOutput
from datasets import Dataset, DatasetDict
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from typing import Dict, List, Tuple, Optional
import gc

# Configuration GPU avec gestion d'erreurs
def setup_gpu():
    """Configure le GPU si disponible"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Nettoyer le cache au démarrage
        torch.cuda.empty_cache()
        
    else:
        device = torch.device("cpu")
        print("Aucun GPU détecté, utilisation du CPU")
    
    return device

class RegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class AestheticRegressionModel(nn.Module):
    def __init__(self, model_name: str, device):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        self.loss_fn = nn.MSELoss()
        self.device = device
        
    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        
        logits = self.regressor(pooled_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            # S'assurer que labels est sur le bon device
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            loss = self.loss_fn(logits, labels)
        
        return RegressionOutput(loss=loss, logits=logits)

class AestheticQualityRegressor:
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        self.model_name = model_name
        self.image_processor = None
        self.model = None
        self.device = setup_gpu()

    def load_model(self):
        """Charge le modèle et le processeur d'images"""
        print("Chargement du modèle...")
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AestheticRegressionModel(self.model_name, self.device)
        self.model = self.model.to(self.device)
        print(f"Modèle chargé sur: {next(self.model.parameters()).device}")

    def preprocess_images(self, image_paths: List[str], labels: List[float]):
        """Prétraite les images en évitant les problèmes de pickle"""
        print("Prétraitement des images...")
        
        processed_data = []
        
        for i, (image_path, label) in enumerate(zip(image_paths, labels)):
            try:
                image = Image.open(image_path).convert('RGB')
                inputs = self.image_processor(image, return_tensors="pt")
                
                processed_data.append({
                    'pixel_values': inputs['pixel_values'].squeeze(),
                    'labels': float(label)
                })
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Erreur avec l'image {image_path}: {e}")
                continue
        
        dataset = Dataset.from_list(processed_data)
        dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        
        return dataset

    def create_train_val_split(self, dataset, test_size: float = 0.2):
        """Divise le dataset en train/validation"""
        dataset = dataset.train_test_split(test_size=test_size, seed=42)
        return DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test']
        })

    def compute_metrics(self, eval_pred):
        """Calcule les métriques d'évaluation pour la régression"""
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }

    def train(self, train_dataset, val_dataset, num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
        """Entraîne le modèle avec gestion d'erreurs pickle"""
        
        # Arguments d'entraînement - IMPORTANT: dataloader_num_workers=0
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="mae",
            greater_is_better=False,
            
            # SOLUTION PRINCIPALE: Désactiver le multiprocessing
            dataloader_num_workers=0,  # Évite l'erreur pickle
            dataloader_pin_memory=False,
            
            # Optimisations GPU
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            
            # Nettoyage mémoire
            save_total_limit=2,
            gradient_checkpointing=True if torch.cuda.is_available() else False,
        )
        
        # Data collator simple
        def collate_fn(batch):
            pixel_values = torch.stack([item['pixel_values'] for item in batch])
            labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float)
            
            return {
                'pixel_values': pixel_values,
                'labels': labels
            }
        
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=collate_fn,
                compute_metrics=self.compute_metrics,
            )
            
            print("Début de l'entraînement...")
            trainer.train()
            
            return trainer
            
        except Exception as e:
            print(f"Erreur pendant l'entraînement: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

# Fonction utilitaire pour organiser les données
def organize_data_from_folder(base_folder: str) -> Tuple[List[str], List[float]]:
    """Organise les données depuis des dossiers nommés par qualité"""
    image_paths = []
    labels = []

    for quality in range(6):
        folder_path = os.path.join(base_folder, f"quality_{quality}")
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(float(quality))

    return image_paths, labels

def main():
    try:
        # Initialisation
        regressor = AestheticQualityRegressor()
        regressor.load_model()

        pathtoimg = "C://Users//revel//OneDrive//Desktop//Qualité image//bestImages - save"
        image_paths, labels = organize_data_from_folder(pathtoimg)

        print(f"Nombre d'images: {len(image_paths)}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Qualité {label}: {count} images")

        if len(image_paths) == 0:
            print("Aucune image trouvée !")
            return

        # Préparation des données
        dataset = regressor.preprocess_images(image_paths, labels)
        datasets = regressor.create_train_val_split(dataset, test_size=0.2)

        print(f"Dataset d'entraînement: {len(datasets['train'])} images")
        print(f"Dataset de validation: {len(datasets['validation'])} images")

        # Entraînement avec paramètres conservateurs
        trainer = regressor.train(
            datasets['train'], 
            datasets['validation'],
            num_epochs=3,
            batch_size=2,  # Batch size très petit pour éviter les problèmes mémoire
            learning_rate=2e-5
        )

        print("Entraînement terminé avec succès !")
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        
        # Nettoyage en cas d'erreur
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
