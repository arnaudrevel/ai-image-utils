import gradio as gr
import os
import random
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import numpy as np

class CondorcetImageAnnotator:
    def __init__(self, image_dir: str, results_file: str = "annotations.csv", state_file: str = "annotation_state.json"):
        self.image_dir = Path(image_dir)
        self.results_file = results_file
        self.state_file = state_file
        self.min_score = 0
        self.max_score = 5
        self.initial_score = 3
        
        # Extensions d'images supportées
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Charger les images et initialiser les scores
        self.images = self._load_images()
        self.scores = self._load_or_initialize_scores()
        self.comparisons = self._load_comparisons()
        
        # Variables pour l'interface
        self.current_pair = None
        
    def _load_images(self) -> List[str]:
        """Charge la liste des images du répertoire."""
        images = []
        for ext in self.image_extensions:
            images.extend(self.image_dir.glob(f"*{ext}"))
            images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return [str(img) for img in images]
    
    def _load_or_initialize_scores(self) -> Dict[str, float]:
        """Charge les scores depuis le fichier d'état ou initialise à 3."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                return state.get('scores', {img: self.initial_score for img in self.images})
        return {img: self.initial_score for img in self.images}
    
    def _load_comparisons(self) -> List[Dict]:
        """Charge l'historique des comparaisons."""
        if os.path.exists(self.results_file):
            try:
                df = pd.read_csv(self.results_file)
                return df.to_dict('records')
            except:
                return []
        return []
    
    def _save_state(self):
        """Sauvegarde l'état actuel."""
        state = {
            'scores': self.scores,
            'current_pair': self.current_pair
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _save_comparison(self, image1: str, image2: str, winner: str):
        """Sauvegarde une comparaison dans le fichier CSV."""
        comparison = {
            'timestamp': pd.Timestamp.now(),
            'image1': os.path.basename(image1),
            'image2': os.path.basename(image2),
            'winner': os.path.basename(winner),
            'score1_before': self.scores[image1],
            'score2_before': self.scores[image2]
        }
        
        # Mettre à jour les scores
        self._update_scores(image1, image2, winner)
        
        comparison['score1_after'] = self.scores[image1]
        comparison['score2_after'] = self.scores[image2]
        
        self.comparisons.append(comparison)
        
        # Sauvegarder dans le CSV
        df = pd.DataFrame(self.comparisons)
        df.to_csv(self.results_file, index=False)
        
        # Sauvegarder l'état
        self._save_state()
    
    def _update_scores(self, image1: str, image2: str, winner: str):
        """Met à jour les scores selon le système de Condorcet simplifié."""
        # Facteur d'ajustement (peut être ajusté selon les besoins)
        adjustment = 0.1
        
        if winner == image1:
            # image1 gagne
            new_score1 = min(self.max_score, self.scores[image1] + adjustment)
            new_score2 = max(self.min_score, self.scores[image2] - adjustment)
        else:
            # image2 gagne
            new_score1 = max(self.min_score, self.scores[image1] - adjustment)
            new_score2 = min(self.max_score, self.scores[image2] + adjustment)
        
        self.scores[image1] = new_score1
        self.scores[image2] = new_score2
    
    def _get_available_images(self) -> List[str]:
        """Retourne les images qui peuvent encore être comparées (score entre min et max)."""
        return [img for img, score in self.scores.items() 
                if self.min_score < score < self.max_score]
    
    def get_random_pair(self) -> Optional[Tuple[str, str]]:
        """Sélectionne une paire aléatoire d'images disponibles."""
        available_images = self._get_available_images()
        
        if len(available_images) < 2:
            return None
        
        pair = random.sample(available_images, 2)
        self.current_pair = tuple(pair)
        self._save_state()
        return self.current_pair
    
    def get_stats(self) -> str:
        """Retourne les statistiques actuelles."""
        total_images = len(self.images)
        available_images = len(self._get_available_images())
        completed_comparisons = len(self.comparisons)
        
        # Distribution des scores
        score_dist = {}
        for score in self.scores.values():
            rounded_score = round(score, 1)
            score_dist[rounded_score] = score_dist.get(rounded_score, 0) + 1
        
        stats = f"""
        📊 **Statistiques de l'annotation**
        
        • Images totales : {total_images}
        • Images disponibles pour comparaison : {available_images}
        • Comparaisons effectuées : {completed_comparisons}
        • Images au score minimum ({self.min_score}) : {sum(1 for s in self.scores.values() if s <= self.min_score)}
        • Images au score maximum ({self.max_score}) : {sum(1 for s in self.scores.values() if s >= self.max_score)}
        
        **Distribution des scores :**
        """
        
        for score, count in sorted(score_dist.items()):
            stats += f"\n• Score {score}: {count} images"
        
        return stats

def create_interface(annotator: CondorcetImageAnnotator):
    """Crée l'interface Gradio."""
    
    def is_valid_image(img):
        """Vérifie si l'image est valide (pas None et pas un array vide)."""
        if img is None:
            return False
        if isinstance(img, np.ndarray):
            return img.size > 0
        if isinstance(img, str):
            return img.strip() != ""
        return True
    
    def load_new_pair():
        """Charge une nouvelle paire d'images."""
        pair = annotator.get_random_pair()
        if pair is None:
            return (
                None, None,
                "🎉 **Annotation terminée !**\n\nToutes les images ont été classées.",
                annotator.get_stats(),
                gr.update(interactive=False),
                gr.update(interactive=False)
            )
        
        return (
            pair[0], pair[1],
            "👆 **Choisissez l'image que vous préférez**",
            annotator.get_stats(),
            gr.update(interactive=True),
            gr.update(interactive=True)
        )
    
    def choose_image1():
        """L'utilisateur choisit la première image."""
        if annotator.current_pair is not None:
            img1, img2 = annotator.current_pair
            annotator._save_comparison(img1, img2, img1)
        return load_new_pair()
    
    def choose_image2():
        """L'utilisateur choisit la deuxième image."""
        if annotator.current_pair is not None:
            img1, img2 = annotator.current_pair
            annotator._save_comparison(img1, img2, img2)
        return load_new_pair()
    
    def export_results():
        """Exporte les résultats finaux."""
        # Créer un DataFrame avec les scores finaux
        results_df = pd.DataFrame([
            {
                'image': os.path.basename(img),
                'final_score': score,
                'rank': rank
            }
            for rank, (img, score) in enumerate(
                sorted(annotator.scores.items(), key=lambda x: x[1], reverse=True), 1
            )
        ])
        
        export_file = f"final_rankings_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(export_file, index=False)
        return f"✅ Résultats exportés dans : {export_file}"
    
    def skip_pair():
        """Passer la paire actuelle sans faire de choix."""
        return load_new_pair()
    
    # Interface Gradio
    with gr.Blocks(title="Annotation d'images - Vote de Condorcet", theme=gr.themes.Soft()) as iface:
        gr.Markdown("# 🖼️ Annotation d'images par vote de Condorcet")
        gr.Markdown("Comparez les images deux par deux pour créer un classement global.")
        
        with gr.Row():
            with gr.Column():
                image1 = gr.Image(label="Image A", height=400, type="filepath")
                btn1 = gr.Button("👍 Je préfère l'image A", variant="primary", size="lg")
            
            with gr.Column():
                image2 = gr.Image(label="Image B", height=400, type="filepath")
                btn2 = gr.Button("👍 Je préfère l'image B", variant="primary", size="lg")
        
        with gr.Row():
            instruction = gr.Markdown("Cliquez sur 'Nouvelle paire' pour commencer")
            
        with gr.Row():
            new_pair_btn = gr.Button("🔄 Nouvelle paire", variant="secondary")
            skip_btn = gr.Button("⏭️ Passer cette paire", variant="secondary")
            export_btn = gr.Button("💾 Exporter les résultats", variant="secondary")
        
        with gr.Row():
            stats = gr.Markdown("Chargement des statistiques...")
        
        with gr.Row():
            export_status = gr.Markdown("")
        
        # State pour stocker les chemins des images actuelles
        current_images_state = gr.State([None, None])
        
        # Événements
        new_pair_btn.click(
            fn=load_new_pair,
            outputs=[image1, image2, instruction, stats, btn1, btn2]
        )
        
        btn1.click(
            fn=choose_image1,
            outputs=[image1, image2, instruction, stats, btn1, btn2]
        )
        
        btn2.click(
            fn=choose_image2,
            outputs=[image1, image2, instruction, stats, btn1, btn2]
        )
        
        skip_btn.click(
            fn=skip_pair,
            outputs=[image1, image2, instruction, stats, btn1, btn2]
        )
        
        export_btn.click(
            fn=export_results,
            outputs=[export_status]
        )
        
        # Charger les statistiques initiales
        iface.load(
            fn=lambda: annotator.get_stats(),
            outputs=[stats]
        )
    
    return iface

def main():
    parser = argparse.ArgumentParser(description="Interface d'annotation d'images par vote de Condorcet")
    parser.add_argument("image_dir", help="Répertoire contenant les images à annoter")
    parser.add_argument("--results-file", default="annotations.csv", help="Fichier CSV pour sauvegarder les résultats")
    parser.add_argument("--state-file", default="annotation_state.json", help="Fichier JSON pour sauvegarder l'état")
    parser.add_argument("--port", type=int, default=7860, help="Port pour l'interface web")
    parser.add_argument("--share", action="store_true", help="Créer un lien public")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"Erreur : Le répertoire {args.image_dir} n'existe pas.")
        return
    
    # Créer l'annotateur
    annotator = CondorcetImageAnnotator(
        image_dir=args.image_dir,
        results_file=args.results_file,
        state_file=args.state_file
    )
    
    if not annotator.images:
        print(f"Erreur : Aucune image trouvée dans {args.image_dir}")
        return
    
    print(f"📁 Répertoire d'images : {args.image_dir}")
    print(f"📊 Nombre d'images trouvées : {len(annotator.images)}")
    print(f"💾 Fichier de résultats : {args.results_file}")
    print(f"🔄 Fichier d'état : {args.state_file}")
    
    # Créer et lancer l'interface
    iface = create_interface(annotator)
    iface.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
