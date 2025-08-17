#!/usr/bin/env python3
"""
Client en ligne de commande pour l'évaluation de la qualité esthétique d'images
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import pandas as pd

class AestheticCLI:
    def __init__(self, model_path: str = "./aesthetic-classifier"):
        """
        Initialise le client CLI
        
        Args:
            model_path: Chemin vers le modèle entraîné
        """
        self.model_path = model_path
        self.model = None
        self.image_processor = None
        self.quality_labels = {
            0: "Très mauvaise",
            1: "Mauvaise", 
            2: "Médiocre",
            3: "Correcte",
            4: "Bonne",
            5: "Excellente"
        }
        
    def load_model(self):
        """Charge le modèle entraîné"""
        try:
            print(f"Chargement du modèle depuis {self.model_path}...")
            self.model = AutoModelForImageClassification.from_pretrained(self.model_path)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
            self.model.eval()  # Mode évaluation
            print("✓ Modèle chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            sys.exit(1)
    
    def is_valid_image(self, file_path: str) -> bool:
        """Vérifie si le fichier est une image valide"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in valid_extensions
    
    def predict_image_quality(self, image_path: str) -> Dict:
        """
        Prédit la qualité esthétique d'une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Dictionnaire avec les résultats de prédiction
        """
        try:
            # Chargement et traitement de l'image
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                inputs = self.image_processor(image, return_tensors="pt")
            
            # Prédiction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                all_probs = probabilities.squeeze().tolist()
            
            return {
                'image_path': image_path,
                'predicted_quality': predicted_class,
                'quality_label': self.quality_labels[predicted_class],
                'confidence': confidence,
                'all_probabilities': all_probs,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'predicted_quality': None,
                'quality_label': None,
                'confidence': None,
                'all_probabilities': None,
                'success': False,
                'error': str(e)
            }
    
    def get_image_files(self, path: str, recursive: bool = False) -> List[str]:
        """
        Récupère la liste des fichiers images dans un répertoire
        
        Args:
            path: Chemin vers le répertoire
            recursive: Recherche récursive dans les sous-dossiers
            
        Returns:
            Liste des chemins vers les images
        """
        image_files = []
        path_obj = Path(path)
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
            
        for file_path in path_obj.glob(pattern):
            if file_path.is_file() and self.is_valid_image(str(file_path)):
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def format_single_result(self, result: Dict, verbose: bool = False) -> str:
        """Formate le résultat pour une seule image"""
        if not result['success']:
            return f"❌ Erreur pour {result['image_path']}: {result['error']}"
        
        output = []
        output.append(f"📁 Image: {result['image_path']}")
        output.append(f"⭐ Qualité: {result['predicted_quality']}/5 ({result['quality_label']})")
        output.append(f"🎯 Confiance: {result['confidence']:.1%}")
        
        if verbose:
            output.append("📊 Distribution des probabilités:")
            for i, prob in enumerate(result['all_probabilities']):
                bar = "█" * int(prob * 20)
                output.append(f"   {i}/5: {prob:.1%} {bar}")
        
        return "\n".join(output)
    
    def format_summary(self, results: List[Dict]) -> str:
        """Crée un résumé des résultats"""
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if not successful_results:
            return "❌ Aucune image traitée avec succès"
        
        # Statistiques
        qualities = [r['predicted_quality'] for r in successful_results]
        avg_quality = sum(qualities) / len(qualities)
        quality_counts = {i: qualities.count(i) for i in range(6)}
        
        summary = []
        summary.append("=" * 50)
        summary.append("📈 RÉSUMÉ")
        summary.append("=" * 50)
        summary.append(f"✅ Images traitées: {len(successful_results)}")
        summary.append(f"❌ Erreurs: {len(failed_results)}")
        summary.append(f"📊 Qualité moyenne: {avg_quality:.2f}/5")
        summary.append("")
        summary.append("📋 Distribution des qualités:")
        
        for quality, count in quality_counts.items():
            if count > 0:
                percentage = (count / len(successful_results)) * 100
                bar = "█" * int(percentage / 5)
                summary.append(f"   {quality}/5 ({self.quality_labels[quality]}): {count} images ({percentage:.1f}%) {bar}")
        
        return "\n".join(summary)

def main():
    parser = argparse.ArgumentParser(
        description="Évalue la qualité esthétique d'images à l'aide d'un modèle entraîné",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s image.jpg                          # Analyser une seule image
  %(prog)s /path/to/images/                   # Analyser toutes les images d'un dossier
  %(prog)s /path/to/images/ -r               # Recherche récursive dans les sous-dossiers
  %(prog)s image.jpg -v                       # Mode verbose avec probabilités détaillées
  %(prog)s /path/to/images/ -o results.json  # Exporter les résultats en JSON
  %(prog)s /path/to/images/ -o results.csv   # Exporter les résultats en CSV
        """
    )
    
    parser.add_argument(
        "input",
        help="Chemin vers une image ou un répertoire d'images"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="./aesthetic-classifier",
        help="Chemin vers le modèle entraîné (défaut: ./aesthetic-classifier)"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recherche récursive dans les sous-dossiers"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode verbose avec probabilités détaillées"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Fichier de sortie pour sauvegarder les résultats (JSON ou CSV)"
    )
    
    parser.add_argument(
        "-s", "--summary",
        action="store_true",
        help="Afficher un résumé statistique (activé automatiquement pour les dossiers)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        help="Seuil de confiance minimum pour afficher les résultats"
    )
    
    args = parser.parse_args()
    
    # Vérification de l'existence du fichier/dossier
    if not os.path.exists(args.input):
        print(f"❌ Erreur: Le chemin '{args.input}' n'existe pas")
        sys.exit(1)
    
    # Initialisation du client
    cli = AestheticCLI(args.model)
    cli.load_model()
    
    # Détermination du type d'input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Une seule image
        if not cli.is_valid_image(args.input):
            print(f"❌ Erreur: '{args.input}' n'est pas une image valide")
            sys.exit(1)
            
        result = cli.predict_image_quality(args.input)
        
        # Filtrage par seuil de confiance
        if args.threshold and result['success'] and result['confidence'] < args.threshold:
            print(f"⚠️  Résultat filtré (confiance {result['confidence']:.1%} < {args.threshold:.1%})")
            return
        
        print(cli.format_single_result(result, args.verbose))
        results = [result]
        
    elif input_path.is_dir():
        # Répertoire d'images
        image_files = cli.get_image_files(args.input, args.recursive)
        
        if not image_files:
            print(f"❌ Aucune image trouvée dans '{args.input}'")
            sys.exit(1)
        
        print(f"🔍 Traitement de {len(image_files)} images...")
        print("=" * 50)
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\r⏳ Progression: {i}/{len(image_files)}", end="", flush=True)
            result = cli.predict_image_quality(image_file)
            results.append(result)
        
        print()  # Nouvelle ligne après la barre de progression
        
        # Filtrage par seuil de confiance
        if args.threshold:
            original_count = len(results)
            results = [r for r in results if not r['success'] or r['confidence'] >= args.threshold]
            filtered_count = original_count - len(results)
            if filtered_count > 0:
                print(f"⚠️  {filtered_count} résultats filtrés par seuil de confiance")
        
        # Affichage des résultats
        for result in results:
            if result['success']:
                print(cli.format_single_result(result, args.verbose))
                print("-" * 30)
        
        # Résumé automatique pour les dossiers
        if len(results) > 1 or args.summary:
            print(cli.format_summary(results))
    
    else:
        print(f"❌ Erreur: '{args.input}' n'est ni un fichier ni un répertoire")
        sys.exit(1)
    
    # Sauvegarde des résultats
    if args.output:
        save_results(results, args.output)

def save_results(results: List[Dict], output_file: str):
    """Sauvegarde les résultats dans un fichier"""
    try:
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.json':
            # Export JSON
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(results),
                'successful_predictions': len([r for r in results if r['success']]),
                'results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        elif output_path.suffix.lower() == '.csv':
            # Export CSV
            df_data = []
            for result in results:
                row = {
                    'image_path': result['image_path'],
                    'predicted_quality': result['predicted_quality'] if result['success'] else None,
                    'quality_label': result['quality_label'] if result['success'] else None,
                    'confidence': result['confidence'] if result['success'] else None,
                    'success': result['success'],
                    'error': result['error']
                }
                
                # Ajout des probabilités
                if result['success']:
                    for i, prob in enumerate(result['all_probabilities']):
                        row[f'prob_quality_{i}'] = prob
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_file, index=False, encoding='utf-8')
        
        else:
            raise ValueError("Format de fichier non supporté. Utilisez .json ou .csv")
        
        print(f"💾 Résultats sauvegardés dans: {output_file}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")

if __name__ == "__main__":
    main()
