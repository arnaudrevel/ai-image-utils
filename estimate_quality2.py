#!/usr/bin/env python3
"""
Client en ligne de commande pour l'évaluation de la qualité esthétique d'images
Optimisé pour l'utilisation GPU
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
        self.device = None
        self.quality_labels = {
            0: "Très mauvaise",
            1: "Mauvaise",
            2: "Médiocre",
            3: "Correcte",
            4: "Bonne",
            5: "Excellente"
        }

    def load_model(self):
        """Charge le modèle entraîné et le place sur le GPU si disponible"""
        try:
            print(f"Chargement du modèle depuis {self.model_path}...")

            # Vérification de la disponibilité du GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Utilisation de: {self.device}")
            if self.device.type == 'cuda':
                print(f"GPU détecté: {torch.cuda.get_device_name(0)}")

            # Chargement du modèle et du processeur d'images
            self.model = AutoModelForImageClassification.from_pretrained(self.model_path)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)

            # Déplacement du modèle vers le GPU et optimisations
            self.model = self.model.to(self.device)
            self.model.eval()  # Mode évaluation

            # Désactive le calcul des gradients pour l'inférence
            torch.set_grad_enabled(False)

            # Optionnel: utilise la demi-précision si le GPU le supporte
            if self.device.type == 'cuda':
                self.model = self.model.half()

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
        Prédit la qualité esthétique d'une image en utilisant le GPU si disponible

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

            # Déplacement des tenseurs vers le GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Prédiction avec optimisation pour l'inférence
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                all_probs = probabilities.squeeze().cpu().tolist()  # Conversion sur CPU pour la sérialisation

            # Libération de la mémoire GPU si nécessaire
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

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

    def format_single_result(self, result: Dict, verbose: bool = False) -> str:
        """Formate un résultat individuel pour l'affichage"""
        if not result['success']:
            return f"❌ {result['image_path']}: {result['error']}"

        output = [
            f"📷 Image: {result['image_path']}",
            f"🏷 Qualité prédite: {result['quality_label']} ({result['predicted_quality']}/5)",
            f"📊 Confiance: {result['confidence']:.1%}"
        ]

        if verbose:
            output.append("📈 Probabilités détaillées:")
            for i, prob in enumerate(result['all_probabilities']):
                label = self.quality_labels.get(i, f"Classe {i}")
                output.append(f"  - {label}: {prob:.1%}")

        return "\n".join(output)

    def format_summary(self, results: List[Dict]) -> str:
        """Génère un résumé des résultats"""
        successful = [r for r in results if r['success']]
        if not successful:
            return "Aucun résultat valide à résumer."

        qualities = [r['predicted_quality'] for r in successful if r['predicted_quality'] is not None]
        confidences = [r['confidence'] for r in successful if r['confidence'] is not None]

        avg_quality = sum(qualities) / len(qualities)
        avg_confidence = sum(confidences) / len(confidences)

        quality_dist = {k: 0 for k in self.quality_labels}
        for q in qualities:
            quality_dist[q] += 1

        summary = [
            "=== RÉSUMÉ DES RÉSULTATS ===",
            f"Images traitées: {len(results)}",
            f"Prédictions réussies: {len(successful)}",
            f"Qualité moyenne: {avg_quality:.2f}/5",
            f"Confiance moyenne: {avg_confidence:.1%}",
            "\nRépartition des qualités:",
        ]

        for label, count in quality_dist.items():
            summary.append(f"  - {self.quality_labels[label]}: {count}")

        return "\n".join(summary)

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

        if not path_obj.exists():
            return image_files

        if path_obj.is_file():
            return [str(path_obj)] if self.is_valid_image(str(path_obj)) else []

        pattern = "**/*" if recursive else "*"
        for file_path in path_obj.glob(pattern):
            if file_path.is_file() and self.is_valid_image(str(file_path)):
                image_files.append(str(file_path))

        return image_files

def main():
    """Point d'entrée principal du programme"""
    parser = argparse.ArgumentParser(description="Évalueur de qualité esthétique d'images")
    parser.add_argument("input", help="Chemin vers une image ou un répertoire d'images")
    parser.add_argument("-o", "--output", help="Fichier de sortie (JSON ou CSV)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Traiter les sous-dossiers récursivement")
    parser.add_argument("-v", "--verbose", action="store_true", help="Afficher les détails des prédictions")
    parser.add_argument("-t", "--threshold", type=float, help="Seuil de confiance minimum (0-1)")
    parser.add_argument("-s", "--summary", action="store_true", help="Afficher un résumé des résultats")

    args = parser.parse_args()
    input_path = Path(args.input)

    cli = AestheticCLI()
    cli.load_model()

    if not input_path.exists():
        print(f"❌ Erreur: '{args.input}' n'existe pas")
        sys.exit(1)

    if input_path.is_file():
        # Fichier unique
        if not cli.is_valid_image(str(input_path)):
            print(f"❌ '{args.input}' n'est pas une image valide")
            sys.exit(1)

        print(f"🔍 Traitement de l'image: {input_path}")
        result = cli.predict_image_quality(str(input_path))

        # Filtrage par seuil de confiance
        if args.threshold and result['success'] and result['confidence'] < args.threshold:
            print(f"⚠️  Résultat filtré (confiance {result['confidence']:.1%} < {args.threshold:.1%})")
            sys.exit(0)

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
