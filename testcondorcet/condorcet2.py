import gradio as gr
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse

class CondorcetImageAnnotator:
    def __init__(self, image_dir: str, output_csv: str, state_file: str = "annotation_state.json"):
        self.image_dir = Path(image_dir)
        self.output_csv = output_csv
        self.state_file = state_file
        
        # Extensions d'images supportées
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # Charger les images
        self.images = self._load_images()
        
        # Initialiser ou charger l'état
        self.scores = {}
        self.raw_scores = {}  # Scores bruts avant normalisation
        self.comparisons_count = 0
        self.max_comparisons = 100  # Par défaut
        self.current_pair = None
        self.is_normalized = False  # Indicateur si les scores ont été normalisés
        
        self._load_state()
        
    def _load_images(self) -> List[str]:
        """Charge la liste des images du répertoire."""
        images = []
        for ext in self.image_extensions:
            images.extend(self.image_dir.glob(f"*{ext}"))
            images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        return [str(img) for img in images]
    
    def _load_state(self):
        """Charge l'état précédent si il existe."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.raw_scores = state.get('raw_scores', {})
                self.scores = state.get('scores', {})
                self.comparisons_count = state.get('comparisons_count', 0)
                self.max_comparisons = state.get('max_comparisons', 100)
                self.is_normalized = state.get('is_normalized', False)
                
                # Si pas de raw_scores sauvegardés (ancien format), utiliser scores comme raw_scores
                if not self.raw_scores and self.scores and not self.is_normalized:
                    self.raw_scores = self.scores.copy()
                
                print(f"État chargé: {self.comparisons_count} comparaisons effectuées")
            except Exception as e:
                print(f"Erreur lors du chargement de l'état: {e}")
                self._initialize_scores()
        else:
            self._initialize_scores()
    
    def _initialize_scores(self):
        """Initialise les scores des images."""
        # Score initial à 0 pour tous
        self.raw_scores = {img: 0.0 for img in self.images}
        self.scores = self.raw_scores.copy()
        self.is_normalized = False
    
    def _save_state(self):
        """Sauvegarde l'état actuel."""
        state = {
            'raw_scores': self.raw_scores,
            'scores': self.scores,
            'comparisons_count': self.comparisons_count,
            'max_comparisons': self.max_comparisons,
            'is_normalized': self.is_normalized
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _normalize_scores(self):
        """Normalise les scores entre 0 et 5 (appelé seulement à la fin)."""
        if not self.raw_scores:
            return
        
        scores_values = list(self.raw_scores.values())
        min_score = min(scores_values)
        max_score = max(scores_values)
        
        if max_score == min_score:
            # Tous les scores sont identiques
            for img in self.raw_scores:
                self.scores[img] = 2.5
        else:
            # Normalisation linéaire entre 0 et 5
            for img in self.raw_scores:
                normalized = 5 * (self.raw_scores[img] - min_score) / (max_score - min_score)
                self.scores[img] = normalized
        
        self.is_normalized = True
        print("Scores normalisés entre 0 et 5")
    
    def _save_results(self):
        """Sauvegarde les résultats dans un fichier CSV."""
        # Normaliser avant de sauvegarder si pas encore fait
        if not self.is_normalized:
            self._normalize_scores()
        
        df = pd.DataFrame([
            {
                'image': os.path.basename(img), 
                'image_path': img,
                'raw_score': self.raw_scores[img],
                'normalized_score': self.scores[img]
            } 
            for img in self.images
        ])
        df = df.sort_values('normalized_score', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        df.to_csv(self.output_csv, index=False)
        print(f"Résultats sauvegardés dans {self.output_csv}")
    
    def _get_available_images(self) -> List[str]:
        """Retourne toutes les images (pas d'exclusion pendant l'annotation)."""
        return list(self.images)
    
    def _select_random_pair(self) -> Optional[Tuple[str, str]]:
        """Sélectionne une paire d'images aléatoire."""
        available_images = self._get_available_images()
        
        if len(available_images) < 2:
            return None
        
        return random.sample(available_images, 2)
    
    def _update_scores(self, winner: str, loser: str):
        """Met à jour les scores bruts après une comparaison."""
        # Système de scoring simple : +1 pour le gagnant, -1 pour le perdant
        # Ou utiliser un système ELO modifié
        
        # Version ELO simplifiée avec les scores bruts
        current_winner = self.raw_scores[winner]
        current_loser = self.raw_scores[loser]
        
        # Probabilité attendue
        expected_winner = 1 / (1 + 10**((current_loser - current_winner) / 400))
        expected_loser = 1 - expected_winner
        
        # Facteur K (plus élevé car pas de normalisation intermédiaire)
        k = 32
        
        # Mise à jour
        self.raw_scores[winner] = current_winner + k * (1 - expected_winner)
        self.raw_scores[loser] = current_loser + k * (0 - expected_loser)
        
        # Les scores affichés restent les scores bruts pendant l'annotation
        if not self.is_normalized:
            self.scores = self.raw_scores.copy()
    
    def get_next_pair(self) -> Tuple[Optional[str], Optional[str], str, int, int]:
        """Retourne la prochaine paire d'images à comparer."""
        if self.comparisons_count >= self.max_comparisons:
            # Normaliser les scores à la fin
            if not self.is_normalized:
                self._normalize_scores()
            self._save_results()
            return None, None, "🎉 Annotation terminée ! Scores normalisés et résultats sauvegardés.", self.comparisons_count, len(self.images)
        
        pair = self._select_random_pair()
        if pair is None:
            if not self.is_normalized:
                self._normalize_scores()
            self._save_results()
            return None, None, "Plus assez d'images disponibles pour continuer.", self.comparisons_count, len(self.images)
        
        self.current_pair = pair
        status = f"Comparaison {self.comparisons_count + 1}/{self.max_comparisons}"
        if not self.is_normalized:
            status += " (scores bruts)"
        else:
            status += " (scores normalisés)"
        
        return pair[0], pair[1], status, self.comparisons_count, len(self.images)
    
    def vote_left(self):
        """Vote pour l'image de gauche."""
        if self.current_pair is None:
            return self.get_next_pair()
        
        if self.comparisons_count >= self.max_comparisons:
            return self.get_next_pair()
        
        winner, loser = self.current_pair
        self._update_scores(winner, loser)
        self.comparisons_count += 1
        self._save_state()
        
        return self.get_next_pair()
    
    def vote_right(self):
        """Vote pour l'image de droite."""
        if self.current_pair is None:
            return self.get_next_pair()
        
        if self.comparisons_count >= self.max_comparisons:
            return self.get_next_pair()
        
        loser, winner = self.current_pair
        self._update_scores(winner, loser)
        self.comparisons_count += 1
        self._save_state()
        
        return self.get_next_pair()
    
    def set_max_comparisons(self, n: int):
        """Définit le nombre maximum de comparaisons."""
        self.max_comparisons = max(1, n)
        self._save_state()
        return f"Nombre maximum de comparaisons défini à {self.max_comparisons}"
    
    def reset_annotation(self):
        """Remet à zéro l'annotation."""
        self._initialize_scores()
        self.comparisons_count = 0
        self.current_pair = None
        self.is_normalized = False
        self._save_state()
        return "Annotation remise à zéro"
    
    def force_normalize(self):
        """Force la normalisation des scores."""
        if self.raw_scores:
            self._normalize_scores()
            self._save_state()
            self._save_results()
            return "Scores normalisés et sauvegardés"
        return "Aucun score à normaliser"
    
    def get_current_rankings(self):
        """Retourne le classement actuel."""
        if not self.scores:
            return "Aucune donnée disponible"
        
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        
        score_type = "normalisés" if self.is_normalized else "bruts"
        rankings = f"🏆 Classement actuel (scores {score_type}):\n\n"
        
        for i, (img, score) in enumerate(sorted_scores[:15], 1):
            img_name = os.path.basename(img)
            rankings += f"{i:2d}. {img_name}: {score:.2f}\n"
        
        if len(sorted_scores) > 15:
            rankings += f"\n... et {len(sorted_scores) - 15} autres images"
        
        # Ajouter quelques statistiques
        scores_values = [s for _, s in sorted_scores]
        rankings += f"\n\n📊 Statistiques:\n"
        rankings += f"Min: {min(scores_values):.2f}\n"
        rankings += f"Max: {max(scores_values):.2f}\n"
        rankings += f"Moyenne: {np.mean(scores_values):.2f}\n"
        rankings += f"Écart-type: {np.std(scores_values):.2f}"
        
        return rankings

def create_interface(annotator: CondorcetImageAnnotator):
    """Crée l'interface Gradio."""
    
    with gr.Blocks(title="Annotation d'images - Vote de Condorcet") as interface:
        gr.Markdown("# 🖼️ Annotation d'images par vote de Condorcet")
        gr.Markdown("Choisissez l'image que vous préférez entre les deux proposées. Les scores seront normalisés à la fin de toutes les comparaisons.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ Configuration")
                max_comp_input = gr.Number(
                    value=annotator.max_comparisons, 
                    label="Nombre maximum de comparaisons",
                    precision=0
                )
                set_max_btn = gr.Button("Définir le maximum")
                max_comp_status = gr.Textbox(label="Status configuration", interactive=False)
                
                gr.Markdown("### 🔄 Actions")
                with gr.Row():
                    reset_btn = gr.Button("🔄 Remettre à zéro", variant="secondary")
                    normalize_btn = gr.Button("📊 Forcer la normalisation", variant="secondary")
                action_status = gr.Textbox(label="Status actions", interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Informations")
                status_text = gr.Textbox(label="Statut", interactive=False)
                progress_text = gr.Textbox(label="Progression", interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🆚 Comparaison")
                with gr.Row():
                    with gr.Column():
                        image_left = gr.Image(label="Image A", height=400)
                        vote_left_btn = gr.Button("👈 Choisir A", variant="primary", size="lg")
                    
                    with gr.Column():
                        image_right = gr.Image(label="Image B", height=400)
                        vote_right_btn = gr.Button("👉 Choisir B", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🏆 Classement actuel")
                rankings_text = gr.Textbox(label="Top 15", lines=20, interactive=False)
                refresh_rankings_btn = gr.Button("🔄 Actualiser le classement")
        
        # État initial
        def initialize():
            img1, img2, status, comp_count, total_images = annotator.get_next_pair()
            rankings = annotator.get_current_rankings()
            progress = f"{comp_count}/{annotator.max_comparisons} comparaisons effectuées"
            return img1, img2, status, progress, rankings
        
        # Fonctions de callback
        def on_vote_left():
            img1, img2, status, comp_count, total_images = annotator.vote_left()
            rankings = annotator.get_current_rankings()
            progress = f"{comp_count}/{annotator.max_comparisons} comparaisons effectuées"
            return img1, img2, status, progress, rankings
        
        def on_vote_right():
            img1, img2, status, comp_count, total_images = annotator.vote_right()
            rankings = annotator.get_current_rankings()
            progress = f"{comp_count}/{annotator.max_comparisons} comparaisons effectuées"
            return img1, img2, status, progress, rankings
        
        def on_set_max_comparisons(n):
            status = annotator.set_max_comparisons(int(n))
            return status
        
        def on_reset():
            status = annotator.reset_annotation()
            # Réinitialiser l'affichage
            img1, img2, main_status, comp_count, total_images = annotator.get_next_pair()
            rankings = annotator.get_current_rankings()
            progress = f"{comp_count}/{annotator.max_comparisons} comparaisons effectuées"
            return status, img1, img2, main_status, progress, rankings
        
        def on_normalize():
            status = annotator.force_normalize()
            rankings = annotator.get_current_rankings()
            return status, rankings
        
        def on_refresh_rankings():
            return annotator.get_current_rankings()
        
        # Connexions des événements
        vote_left_btn.click(
            on_vote_left,
            outputs=[image_left, image_right, status_text, progress_text, rankings_text]
        )
        
        vote_right_btn.click(
            on_vote_right,
            outputs=[image_left, image_right, status_text, progress_text, rankings_text]
        )
        
        set_max_btn.click(
            on_set_max_comparisons,
            inputs=[max_comp_input],
            outputs=[max_comp_status]
        )
        
        reset_btn.click(
            on_reset,
            outputs=[action_status, image_left, image_right, status_text, progress_text, rankings_text]
        )
        
        normalize_btn.click(
            on_normalize,
            outputs=[action_status, rankings_text]
        )
        
        refresh_rankings_btn.click(
            on_refresh_rankings,
            outputs=[rankings_text]
        )
        
        # Initialisation au chargement
        interface.load(
            initialize,
            outputs=[image_left, image_right, status_text, progress_text, rankings_text]
        )
    
    return interface

def main():
    parser = argparse.ArgumentParser(description="Interface d'annotation d'images par vote de Condorcet")
    parser.add_argument("--image_dir", type=str, required=True, help="Répertoire contenant les images")
    parser.add_argument("--output_csv", type=str, default="image_rankings.csv", help="Fichier CSV de sortie")
    parser.add_argument("--state_file", type=str, default="annotation_state.json", help="Fichier d'état de l'annotation")
    parser.add_argument("--max_comparisons", type=int, default=100, help="Nombre maximum de comparaisons")
    parser.add_argument("--port", type=int, default=7860, help="Port pour l'interface web")
    parser.add_argument("--share", action="store_true", help="Partager l'interface publiquement")
    
    args = parser.parse_args()
    
    # Vérifier que le répertoire d'images existe
    if not os.path.exists(args.image_dir):
        print(f"Erreur: Le répertoire {args.image_dir} n'existe pas.")
        return
    
    # Créer l'annotateur
    annotator = CondorcetImageAnnotator(args.image_dir, args.output_csv, args.state_file)
    annotator.set_max_comparisons(args.max_comparisons)
    
    if len(annotator.images) == 0:
        print(f"Erreur: Aucune image trouvée dans {args.image_dir}")
        return
    
    print(f"Chargé {len(annotator.images)} images depuis {args.image_dir}")
    
    # Créer et lancer l'interface
    interface = create_interface(annotator)
    interface.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
