#!/usr/bin/env python3
"""
Client CLI pour l'évaluation de la qualité esthétique d'images (Optimisé CPU/GPU).

Ce script permet d'analyser une seule image ou un répertoire entier d'images
à l'aide d'un modèle de classification d'images pré-entraîné (ViT de Hugging Face).
Il détecte automatiquement la présence d'un GPU NVIDIA (via CUDA) pour exécuter l'inférence
en mode accéléré demi-précision (FP16), mais permet également de forcer l'exécution sur CPU.
Il propose un filtrage par confiance et un export des prédictions sous formats JSON et CSV.

Prérequis :
    - PyTorch
    - Transformers (Hugging Face)
    - Pillow
    - Pandas

Usage :
    python analysis/cli_predict_quality.py <chemin_image_ou_dossier> [options]
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
    """
    Classe unifiée gérant le chargement du modèle ViT, la prédiction des scores sur CPU ou GPU,
    et la mise en forme des rapports et résumés en ligne de commande de façon robuste.
    """

    def __init__(self, model_path: str = "models/aesthetic-classifier", force_cpu: bool = False):
        """
        Initialise l'évaluateur.

        Args:
            model_path (str): Chemin local vers les poids du modèle pré-entraîné.
            force_cpu (bool): Si True, force l'utilisation du processeur central (CPU).
        """
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.model = None
        self.image_processor = None
        self.device = None
        
        # Dictionnaire associant les indices de classe à leurs labels de qualité en français
        self.quality_labels = {
            0: "Tres mauvaise",
            1: "Mauvaise",
            2: "Mediocre",
            3: "Correcte",
            4: "Bonne",
            5: "Excellente"
        }

    def load_model(self):
        """
        Charge en mémoire le modèle de classification et le préprocesseur.
        Configure dynamiquement le périphérique (CPU ou GPU CUDA) et applique les
        optimisations adaptées (FP16 sur GPU, mode évaluation).
        """
        try:
            # Robustesse : Redirection automatique vers le sous-dossier checkpoint-XXX si nécessaire
            if os.path.exists(self.model_path) and not os.path.exists(os.path.join(self.model_path, "config.json")):
                checkpoints = [d for d in os.listdir(self.model_path) if d.startswith("checkpoint-")]
                if checkpoints:
                    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                    self.model_path = os.path.join(self.model_path, latest_checkpoint)
                    print(f"[INFO] Redirection automatique vers le checkpoint : {self.model_path}")

            print(f"[INFO] Chargement du modele depuis : {self.model_path} ...")

            # Détection du périphérique (GPU CUDA disponible vs CPU)
            if torch.cuda.is_available() and not self.force_cpu:
                self.device = torch.device("cuda")
                device_name = torch.cuda.get_device_name(0)
                print(f"[INFO] Pheripherique d'inference : GPU CUDA ({device_name})")
            else:
                self.device = torch.device("cpu")
                print("[INFO] Pheripherique d'inference : CPU")

            # Chargement des architectures Hugging Face
            self.model = AutoModelForImageClassification.from_pretrained(self.model_path)
            
            # Chargement du préprocesseur d'images avec repli robuste si preprocessor_config.json absent localement
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
            except Exception:
                print("[INFO] preprocessor_config.json absent localement. Repli sur 'google/vit-base-patch16-224'.")
                self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

            # Transfert du modèle sur le périphérique cible
            self.model = self.model.to(self.device)
            
            # Positionnement du modèle en mode évaluation (désactivation Dropout/BatchNorm)
            self.model.eval()

            # Optimisation majeure : Désactivation globale du calcul de gradients
            torch.set_grad_enabled(False)

            # Optimisation majeure sur GPU : Passage en demi-précision (FP16)
            if self.device.type == "cuda":
                self.model = self.model.half()

            print("[SUCCES] Modele charge avec succes.")
        except Exception as e:
            print(f"[ERREUR] Lors du chargement ou du transfert du modele: {e}")
            sys.exit(1)

    def is_valid_image(self, file_path: str) -> bool:
        """
        Détermine si un fichier est une image en vérifiant son extension.

        Args:
            file_path (str): Chemin d'accès du fichier.

        Returns:
            bool: True si le format est supporté.
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in valid_extensions

    def predict_image_quality(self, image_path: str) -> Dict:
        """
        Effectue une prédiction de qualité esthétique sur une seule image.
        Prend en charge le transfert automatique de tenseurs et l'inférence optimisée.

        Args:
            image_path (str): Chemin d'accès vers le fichier image.

        Returns:
            Dict: Dictionnaire contenant les métriques de classification (classe, confiance, probabilités).
        """
        try:
            # Traitement initial CPU (Lecture Pillow)
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                inputs = self.image_processor(image, return_tensors="pt")

            # Transfert des tenseurs d'entrée vers le GPU/CPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Si le modèle est sur GPU (FP16), on convertit également les inputs flottants en FP16
            if self.device.type == "cuda":
                inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

            # Utilisation du mode inférence natif de PyTorch (ultra-rapide)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                
                # Transfert obligatoire de la probabilité vers le CPU avant conversion en liste standard
                all_probs = probabilities.squeeze().cpu().tolist()

            # Libération immédiate du cache CUDA pour éviter les fragmentations de VRAM
            if self.device.type == "cuda":
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

    def get_image_files(self, path: str, recursive: bool = False) -> List[str]:
        """
        Recherche récursive ou directe de tous les fichiers images valides dans un répertoire.

        Args:
            path (str): Dossier à analyser.
            recursive (bool): Si True, parcourt tous les sous-répertoires.

        Returns:
            List[str]: Liste ordonnée de chemins d'images.
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

        return sorted(image_files)

    def format_single_result(self, result: Dict, verbose: bool = False) -> str:
        """
        Formate le résultat d'une image pour un affichage propre compatible console Windows (ASCII).

        Args:
            result (Dict): Résultat brut de prédiction.
            verbose (bool): Si True, affiche la distribution des probabilités.

        Returns:
            str: Chaîne de caractères formatée.
        """
        if not result['success']:
            return f"[ERREUR] Pour {result['image_path']}: {result['error']}"

        output = [
            f"Image              : {result['image_path']}",
            f"Qualite predite    : {result['predicted_quality']}/5 ({result['quality_label']})",
            f"Confiance          : {result['confidence']:.1%}"
        ]

        if verbose:
            output.append("Probabilites detaillees par classe :")
            for i, prob in enumerate(result['all_probabilities']):
                # Création d'une barre horizontale ASCII proportionnelle à la probabilité (largeur 20)
                bar = "=" * int(prob * 20)
                output.append(f"  - {i}/5 ({self.quality_labels[i]:13s}): {prob:6.1%} [{bar:<20}]")

        return "\n".join(output)

    def format_summary(self, results: List[Dict]) -> str:
        """
        Calcule et génère un résumé statistique des analyses effectuées sur un lot (ASCII).

        Args:
            results (List[Dict]): Liste des dictionnaires de prédiction individuels.

        Returns:
            str: Résumé formaté.
        """
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        if not successful_results:
            return "[ERREUR] Aucune image n'a pu etre traitee avec succes."

        # Calculs statistiques de base
        qualities = [r['predicted_quality'] for r in successful_results]
        confidences = [r['confidence'] for r in successful_results]
        
        avg_quality = sum(qualities) / len(qualities)
        avg_confidence = sum(confidences) / len(confidences)
        
        quality_counts = {i: qualities.count(i) for i in range(6)}

        summary = [
            "==================================================",
            "          RESUME STATISTIQUE DES PREDICTIONS      ",
            "==================================================",
            f"Images traitees avec succes : {len(successful_results)}",
            f"Erreurs de traitement       : {len(failed_results)}",
            f"Qualite esthetique moyenne  : {avg_quality:.2f}/5",
            f"Confiance moyenne           : {avg_confidence:.1%}",
            "",
            "Distribution par niveau de qualite :",
        ]

        for quality, count in quality_counts.items():
            percentage = (count / len(successful_results)) * 100
            bar = "=" * int(percentage / 5)
            summary.append(f"  - {quality}/5 ({self.quality_labels[quality]:13s}): {count:3d} images ({percentage:5.1f}%) [{bar:<20}]")

        return "\n".join(summary)


def save_results(results: List[Dict], output_file: str):
    """
    Sauvegarde la liste des résultats de prédiction dans un fichier externe JSON ou CSV.

    Args:
        results (List[Dict]): Données brutes de prédiction.
        output_file (str): Nom ou chemin du fichier cible.
    """
    try:
        output_path = Path(output_file)

        if output_path.suffix.lower() == '.json':
            # Export complet sous format JSON
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(results),
                'successful_predictions': len([r for r in results if r['success']]),
                'results': results
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        elif output_path.suffix.lower() == '.csv':
            # Extraction des données pour table CSV
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

                # Ajout des probabilités de chaque classe dans des colonnes dédiées
                if result['success']:
                    for i, prob in enumerate(result['all_probabilities']):
                        row[f'prob_quality_{i}'] = prob

                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(output_file, index=False, encoding='utf-8')

        else:
            raise ValueError("Format de fichier non supporte. Veuillez utiliser une extension '.json' ou '.csv'")

        print(f"[INFO] Resultats exportes avec succes dans : {output_file}")

    except Exception as e:
        print(f"[ERREUR] Lors de la sauvegarde des resultats: {e}")


def main():
    """
    Fonction principale du module CLI unifié.
    Définit les arguments de ligne de commande et orchestre le flux d'analyse d'images.
    """
    # Recherche du modèle par défaut de manière relative au script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_model_path = project_root / "models" / "aesthetic-classifier"

    parser = argparse.ArgumentParser(
        description="Evalue la qualite esthetique d'images a l'aide du modele ViT (Auto-detection CPU/GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemples d'utilisation:
  python analysis/cli_predict_quality.py image.jpg                        # Analyser une image (Auto CPU/GPU)
  python analysis/cli_predict_quality.py /dossier/                        # Analyser toutes les images d'un dossier
  python analysis/cli_predict_quality.py /dossier/ -r                     # Recherche recursive
  python analysis/cli_predict_quality.py image.jpg --cpu -v               # Forcer CPU et mode detaille
  python analysis/cli_predict_quality.py /dossier/ -o resultats.csv        # Exporter les resultats en CSV
        """
    )

    parser.add_argument(
        "input",
        help="Chemin vers le fichier d'une image ou le repertoire a etudier"
    )

    parser.add_argument(
        "-m", "--model",
        default=str(default_model_path),
        help="Chemin vers le modele d'images ViT (par defaut: models/aesthetic-classifier)"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Scanner recursivement les sous-dossiers"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode detaille (affiche la distribution des probabilites)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Enregistrer les resultats sous format .json ou .csv"
    )

    parser.add_argument(
        "-s", "--summary",
        action="store_true",
        help="Forcer l'affichage d'un resume general des statistiques"
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        help="Seuil de confiance minimum (entre 0 et 1) pour afficher les predictions"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forcer l'execution sur le processeur (CPU) meme si un GPU est disponible"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERREUR] Le chemin spécifie '{args.input}' n'existe pas.")
        sys.exit(1)

    # Initialisation de notre client CLI unifié
    cli = AestheticCLI(args.model, force_cpu=args.cpu)
    cli.load_model()

    input_path = Path(args.input)

    if input_path.is_file():
        # Traitement d'un fichier unique
        if not cli.is_valid_image(args.input):
            print(f"[ERREUR] Le fichier '{args.input}' n'est pas une image supportee.")
            sys.exit(1)

        result = cli.predict_image_quality(args.input)

        # Application du seuil de confiance
        if args.threshold and result['success'] and result['confidence'] < args.threshold:
            print(f"[ATTENTION] Resultat filtre (la confiance de {result['confidence']:.1%} est inferieure au seuil de {args.threshold:.1%})")
            return

        print(cli.format_single_result(result, args.verbose))
        results = [result]

    elif input_path.is_dir():
        # Traitement d'un lot d'images dans un répertoire
        image_files = cli.get_image_files(args.input, args.recursive)

        if not image_files:
            print(f"[ERREUR] Aucune image valide trouvee dans le repertoire '{args.input}'.")
            sys.exit(1)

        print(f"[INFO] Debut du traitement de {len(image_files)} images...")
        print("=" * 50)

        results = []
        for i, image_file in enumerate(image_files, 1):
            # Barre de progression console ASCII simple
            print(f"\r Analyse en cours : {i}/{len(image_files)} images", end="", flush=True)
            result = cli.predict_image_quality(image_file)
            results.append(result)

        print()  # saut de ligne après fin de progression

        # Application du filtre de confiance
        if args.threshold:
            original_count = len(results)
            results = [r for r in results if not r['success'] or r['confidence'] >= args.threshold]
            filtered_count = original_count - len(results)
            if filtered_count > 0:
                print(f"[ATTENTION] {filtered_count} predictions filtrees par seuil de confiance.")

        # Affichage individuel
        for result in results:
            if result['success']:
                print(cli.format_single_result(result, args.verbose))
                print("-" * 30)

        # Résumé général
        if len(results) > 1 or args.summary:
            print(cli.format_summary(results))

    else:
        print(f"[ERREUR] '{args.input}' n'est ni un fichier ni un repertoire standard.")
        sys.exit(1)

    # Sauvegarde si option spécifiée
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
