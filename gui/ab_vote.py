"""
Interface Gradio d'évaluation d'images par vote comparatif A/B (Elo & Condorcet unifiés).

Ce module permet de réaliser des comparaisons par paires (pairwise comparisons) de manière
interactive. Deux images sont présentées à l'utilisateur, qui sélectionne sa préférée.
Deux méthodes de calcul de score sont disponibles :
1. Elo : Système compétitif non linéaire (K=32). Les scores bruts sont projetés
   sur une échelle de 0.0 à 5.0 en fin de session.
2. Condorcet : Système incrémental linéaire (+0.1/-0.1, borné entre 0.0 et 5.0).
   Les images ayant atteint les scores extrêmes de 0 ou 5 sont écartées des duels futurs.
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class PairwiseImageAnnotator:
    """
    Moteur unifié gérant les duels comparatifs (A/B) d'images, la persistance d'état
    et le calcul des scores esthétiques selon les algorithmes Condorcet et Elo.
    """

    def __init__(
        self,
        image_dir: str,
        output_file: str,
        state_file: str = "annotation_state.json",
        method: str = "elo",
        max_comparisons: int = 100
    ):
        """
        Initialise l'annotateur de comparaison par paires.

        Args:
            image_dir (str): Répertoire contenant les images à comparer.
            output_file (str): Chemin du CSV de sortie pour les classements/duels.
            state_file (str): Fichier JSON servant à persister l'état d'annotation.
            method (str): Méthode de vote ('elo' ou 'condorcet').
            max_comparisons (int): Nombre maximum de duels recommandés pour Elo.
        """
        self.image_dir = Path(image_dir)
        self.output_file = Path(output_file)
        self.state_file = Path(state_file)
        self.method = method.lower()
        self.max_comparisons = max_comparisons
        
        # Extensions d'images autorisées
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Chargement des images physiques
        self.images = self._load_images()
        
        # Paramètres généraux et structures d'annotation
        self.scores = {}            # Échelle standardisée (0.0 à 5.0)
        self.raw_scores = {}        # Scores bruts (Elo ou Condorcet linéaire)
        self.comparisons_count = 0  # Nombre de duels complétés
        self.current_pair = None    # Paire actuellement affichée
        self.is_normalized = False  # Indicateur si la normalisation Elo a été appliquée
        self.comparisons = []       # Historique des duels (surtout utile pour Condorcet)

        # Spécificités Condorcet
        self.min_score = 0.0
        self.max_score = 5.0
        self.initial_condorcet_score = 3.0
        
        # Restauration automatique si un état existe
        self._load_state()

    def _load_images(self) -> List[str]:
        """Scanne le dossier cible pour lister tous les chemins d'images valides."""
        images = []
        if not self.image_dir.exists():
            return []
        for ext in self.image_extensions:
            images.extend(self.image_dir.glob(f"*{ext}"))
            images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        # Élimination des doublons sur les systèmes de fichiers insensibles à la casse (Windows)
        return sorted(list(set([str(img) for img in images])))

    def _initialize_scores(self):
        """Initialise la table des scores en fonction de la méthode choisie."""
        if self.method == "condorcet":
            self.raw_scores = {img: self.initial_condorcet_score for img in self.images}
            self.scores = self.raw_scores.copy()
        else:  # Elo
            self.raw_scores = {img: 0.0 for img in self.images}
            self.scores = {img: 2.5 for img in self.images}
        self.comparisons_count = 0
        self.is_normalized = False
        self.current_pair = None
        self.comparisons = []

    def _load_state(self):
        """Restaure l'état d'annotation existant depuis le fichier JSON."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Détection de la méthode à la reprise
                saved_method = state.get('method', self.method)
                if saved_method != self.method:
                    print(f"[INFO] Passage automatique au mode sauvegarde : {saved_method.upper()}")
                    self.method = saved_method
                
                old_raw_scores = state.get('raw_scores', {})
                old_scores = state.get('scores', {})
                self.comparisons_count = state.get('comparisons_count', 0)
                self.max_comparisons = state.get('max_comparisons', self.max_comparisons)
                self.is_normalized = state.get('is_normalized', False)
                self.comparisons = state.get('comparisons', [])
                
                # Reconstruction intelligente des dictionnaires pour gérer le changement de répertoire des images
                self.raw_scores = {}
                self.scores = {}
                
                # Association par nom de fichier de base (basename) pour rétrocompatibilité totale
                basename_to_raw = {os.path.basename(k): v for k, v in old_raw_scores.items()}
                basename_to_norm = {os.path.basename(k): v for k, v in old_scores.items()}
                
                for img in self.images:
                    base = os.path.basename(img)
                    if base in basename_to_raw:
                        self.raw_scores[img] = basename_to_raw[base]
                    else:
                        if self.method == "condorcet":
                            self.raw_scores[img] = self.initial_condorcet_score
                        else:
                            self.raw_scores[img] = 0.0
                            
                    if base in basename_to_norm:
                        self.scores[img] = basename_to_norm[base]
                    else:
                        if self.method == "condorcet":
                            self.scores[img] = self.initial_condorcet_score
                        else:
                            self.scores[img] = 2.5
                
                print(f"[INFO] Etat restaure : {self.comparisons_count} duels effectues (Mode: {self.method.upper()})")
            except Exception as e:
                print(f"[ATTENTION] Echec de restauration de l'etat ({e}), reinitialisation...")
                self._initialize_scores()
        else:
            self._initialize_scores()

    def _save_state(self):
        """Sauvegarde l'état d'avancement et les scores dans le fichier JSON d'état."""
        # Création du dossier parent du fichier d'état s'il n'existe pas
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'method': self.method,
            'raw_scores': self.raw_scores,
            'scores': self.scores,
            'comparisons_count': self.comparisons_count,
            'max_comparisons': self.max_comparisons,
            'is_normalized': self.is_normalized,
            'comparisons': self.comparisons,
            'current_pair': self.current_pair
        }
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _normalize_elo_scores(self):
        """Projete linéairement les scores bruts ELO sur l'échelle de 0.0 à 5.0."""
        if not self.raw_scores:
            return
        
        scores_values = list(self.raw_scores.values())
        min_score = min(scores_values)
        max_score = max(scores_values)
        
        if max_score == min_score:
            for img in self.raw_scores:
                self.scores[img] = 2.5
        else:
            for img in self.raw_scores:
                normalized = 5.0 * (self.raw_scores[img] - min_score) / (max_score - min_score)
                self.scores[img] = round(normalized, 3)
        
        self.is_normalized = True

    def export_rankings(self) -> str:
        """Génère et exporte le classement ordonné complet dans le fichier CSV."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.method == "elo" and not self.is_normalized:
            self._normalize_elo_scores()

        # Tri décroissant selon le score normalisé (ou le score Condorcet)
        sorted_items = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        
        records = []
        for rank, (img, score) in enumerate(sorted_items, 1):
            records.append({
                'rank': rank,
                'image': os.path.basename(img),
                'image_path': img,
                'raw_score': round(self.raw_scores.get(img, 0.0), 3),
                'score_0_to_5': round(score, 3)
            })
            
        df = pd.DataFrame(records)
        df.to_csv(self.output_file, index=False, encoding='utf-8')
        
        # Enregistrement de l'historique des duels Condorcet si disponible
        if self.method == "condorcet" and self.comparisons:
            history_file = self.output_file.parent / f"history_duels_condorcet.csv"
            pd.DataFrame(self.comparisons).to_csv(history_file, index=False, encoding='utf-8')
            return f"[SUCCES] Classement exporte dans : {self.output_file} (Historique duels dans : {history_file})"
            
        return f"[SUCCES] Classement ordonne exporte dans : {self.output_file}"

    def _get_eligible_images(self) -> List[str]:
        """Filtre les images éligibles pour le duel en cours."""
        if self.method == "condorcet":
            # Exclusion des images ayant atteint 0.0 ou 5.0
            return [img for img, score in self.scores.items() 
                    if self.min_score < score < self.max_score]
        else:
            # En mode Elo, toutes les images restent en lice
            return list(self.images)

    def select_next_pair(self) -> Optional[Tuple[str, str]]:
        """Tire au sort deux images distinctes éligibles."""
        eligible = self._get_eligible_images()
        if len(eligible) < 2:
            return None
        
        pair = random.sample(eligible, 2)
        self.current_pair = tuple(pair)
        self._save_state()
        return self.current_pair

    def _update_condorcet_scores(self, winner: str, loser: str):
        """Mise à jour linéaire simple pour le mode Condorcet (+0.1 / -0.1)."""
        adjustment = 0.1
        
        # Enregistrement de l'historique avant modification
        duel = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'image1': os.path.basename(winner),
            'image2': os.path.basename(loser),
            'winner': os.path.basename(winner),
            'score1_before': self.scores[winner],
            'score2_before': self.scores[loser]
        }
        
        new_winner = min(self.max_score, self.scores[winner] + adjustment)
        new_loser = max(self.min_score, self.scores[loser] - adjustment)
        
        self.raw_scores[winner] = new_winner
        self.raw_scores[loser] = new_loser
        self.scores[winner] = new_winner
        self.scores[loser] = new_loser
        
        duel['score1_after'] = new_winner
        duel['score2_after'] = new_loser
        self.comparisons.append(duel)

    def _update_elo_scores(self, winner: str, loser: str):
        """Mise à jour non linéaire standardisée pour le mode ELO (K=32)."""
        current_winner_raw = self.raw_scores[winner]
        current_loser_raw = self.raw_scores[loser]
        
        # Calcul de l'espérance de gain
        expected_winner = 1.0 / (1.0 + 10.0**((current_loser_raw - current_winner_raw) / 400.0))
        expected_loser = 1.0 - expected_winner
        
        k = 32
        
        # Ajustement Elo des scores bruts
        self.raw_scores[winner] = current_winner_raw + k * (1.0 - expected_winner)
        self.raw_scores[loser] = current_loser_raw + k * (0.0 - expected_loser)
        
        # En mode ELO, les scores bruts sont recopiés et ne seront projetés qu'à la normalisation
        if not self.is_normalized:
            self.scores = self.raw_scores.copy()

    def process_vote(self, winner_idx: int) -> Tuple[Optional[str], Optional[str], str, int, int]:
        """
        Enregistre l'issue du duel et génère la paire suivante.
        
        Args:
            winner_idx (int): 0 si l'image A gagne, 1 si l'image B gagne.
        """
        if self.current_pair is not None:
            img_a, img_b = self.current_pair
            winner = img_a if winner_idx == 0 else img_b
            loser = img_b if winner_idx == 0 else img_a
            
            if self.method == "condorcet":
                self._update_condorcet_scores(winner, loser)
            else:
                self._update_elo_scores(winner, loser)
                
            self.comparisons_count += 1
            self._save_state()
            
        return self.get_ui_state()

    def get_ui_state(self) -> Tuple[Optional[str], Optional[str], str, int, int]:
        """Renvoie le tuple nécessaire pour configurer l'interface Gradio."""
        # Limite atteinte en mode Elo
        if self.method == "elo" and self.comparisons_count >= self.max_comparisons:
            self.export_rankings()
            status = f"[FIN] Session achevee ({self.comparisons_count}/{self.max_comparisons} duels). Classement exporte !"
            return None, None, status, self.comparisons_count, len(self.images)
            
        pair = self.select_next_pair()
        if pair is None:
            self.export_rankings()
            status = "[FIN] Plus assez d'images eligibles. Classement finalise et exporte !"
            return None, None, status, self.comparisons_count, len(self.images)
            
        status = f"Duel {self.comparisons_count + 1} en cours (Mode: {self.method.upper()})"
        if self.method == "elo":
            status += f" / Limite conseil : {self.max_comparisons}"
            
        return pair[0], pair[1], status, self.comparisons_count, len(self.images)

    def get_stats_markdown(self) -> str:
        """Calcule et affiche un résumé complet des statistiques pour l'UI."""
        total_images = len(self.images)
        comparisons_done = self.comparisons_count
        
        if total_images == 0:
            return "Aucune image chargee dans le module."

        stats = f"### 📊 Statistiques de la Session (Mode: **{self.method.upper()}**)\n"
        stats += f"- **Nombre total d'images chargées :** {total_images}\n"
        stats += f"- **Duels effectués jusqu'ici      :** {comparisons_done}\n"

        if self.method == "condorcet":
            eligible_count = len(self._get_eligible_images())
            excluded_low = sum(1 for s in self.scores.values() if s <= self.min_score)
            excluded_high = sum(1 for s in self.scores.values() if s >= self.max_score)
            
            stats += f"- **Images toujours actives         :** {eligible_count} / {total_images}\n"
            stats += f"- **Images classees au minimum (0)  :** {excluded_low}\n"
            stats += f"- **Images classees au maximum (5)  :** {excluded_high}\n\n"
            
            # Affichage de la distribution des notes
            stats += "**📈 Distribution des scores linéaires :**\n"
            score_dist = {}
            for score in self.scores.values():
                rounded = round(score, 1)
                score_dist[rounded] = score_dist.get(rounded, 0) + 1
                
            for score, count in sorted(score_dist.items()):
                bar = "█" * min(count, 15)
                stats += f"\n- **Score {score:.1f}** : {count:3d} images {bar}"
        
        else:  # Elo
            stats += f"- **Seuil cible de duels            :** {self.max_comparisons}\n\n"
            stats += "**🏆 TOP 15 - Classement Actuel (Scores normalisés de 0 à 5) :**\n"
            
            if self.is_normalized or comparisons_done > 0:
                if not self.is_normalized:
                    self._normalize_elo_scores()
                
                sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
                for idx, (img, score) in enumerate(sorted_scores[:15], 1):
                    img_name = os.path.basename(img)
                    raw_val = self.raw_scores.get(img, 0.0)
                    stats += f"\n{idx:2d}. **{img_name}** (Elo: {raw_val:.1f} -> **{score:.2f} / 5**)"
                
                if len(sorted_scores) > 15:
                    stats += f"\n\n*... et {len(sorted_scores) - 15} autres images.*"
            else:
                stats += "\nAucun duel effectue pour le moment. Le classement s'affichera des le premier vote."

        return stats

    def reset_session(self) -> str:
        """Remet à zéro l'ensemble des scores et réinitialise le fichier d'état."""
        self._initialize_scores()
        self._save_state()
        return "[SUCCES] Session de vote réinitialisée. Tous les scores sont remis à zéro !"

    def force_elo_normalization(self) -> str:
        """Force manuellement la normalisation et la mise à jour du CSV."""
        if self.method != "elo":
            return "[ERREUR] La normalisation ELO n'est pas applicable en mode Condorcet."
        self._normalize_elo_scores()
        self._save_state()
        self.export_rankings()
        return "[SUCCES] Normalisation appliquee et classement exporte avec succes."


def create_interface(annotator: PairwiseImageAnnotator) -> gr.Blocks:
    """Génère la structure de l'interface graphique Gradio interactive."""
    theme = gr.themes.Soft() if annotator.method == "condorcet" else gr.themes.Default()
    
    with gr.Blocks(title=f"Votes Comparatifs A/B - {annotator.method.upper()}", theme=theme) as iface:
        gr.Markdown(f"# 🆚 Evaluation Interactive d'Images - Mode **{annotator.method.upper()}**")
        gr.Markdown(
            "Visualisez deux imagescote-à-cote et sélectionnez celle qui présente la meilleure qualité "
            "visuelle ou esthétique pour mettre à jour la base de scores."
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Duel A / B")
                with gr.Row():
                    with gr.Column():
                        image_a = gr.Image(label="Image A", height=450, type="filepath", interactive=False)
                        btn_a = gr.Button("👍 Je préfère l'image A", variant="primary", size="lg")
                    with gr.Column():
                        image_b = gr.Image(label="Image B", height=450, type="filepath", interactive=False)
                        btn_b = gr.Button("👍 Je préfère l'image B", variant="primary", size="lg")
                
                with gr.Row():
                    status_lbl = gr.Textbox(label="Statut du Duel", interactive=False)
                
                with gr.Row():
                    skip_btn = gr.Button("⏭️ Passer ce duel (Tirage aléatoire)", variant="secondary")
                    export_btn = gr.Button("💾 Exporter le classement CSV", variant="secondary")
                    
                with gr.Row():
                    action_msg = gr.Markdown("")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistiques & Classement")
                stats_area = gr.Markdown("Chargement des statistiques initiales...")
                refresh_stats_btn = gr.Button("🔄 Actualiser les statistiques")
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Options globales")
                if annotator.method == "elo":
                    max_comparisons_input = gr.Number(
                        value=annotator.max_comparisons,
                        label="Nombre total de duels conseillés",
                        precision=0
                    )
                    apply_norm_btn = gr.Button("📊 Appliquer la normalisation ELO", variant="secondary")
                
                reset_btn = gr.Button("🗑️ Réinitialiser tous les scores", variant="stop")

        # Logique réactive des votes
        def handle_vote_a():
            img_a, img_b, status, comp_count, total = annotator.process_vote(0)
            return img_a, img_b, status, annotator.get_stats_markdown()

        def handle_vote_b():
            img_a, img_b, status, comp_count, total = annotator.process_vote(1)
            return img_a, img_b, status, annotator.get_stats_markdown()

        def handle_skip():
            pair = annotator.select_next_pair()
            if pair is None:
                return None, None, "[FIN] Plus assez d'images éligibles.", annotator.get_stats_markdown()
            img_a, img_b = pair
            status = f"Duel {annotator.comparisons_count + 1} en cours (Mode: {annotator.method.upper()})"
            return img_a, img_b, status, annotator.get_stats_markdown()

        def handle_export():
            msg = annotator.export_rankings()
            return msg

        def handle_reset():
            msg = annotator.reset_session()
            img_a, img_b, status, comp_count, total = annotator.get_ui_state()
            return img_a, img_b, status, msg, annotator.get_stats_markdown()

        # Connecteurs d'événements
        btn_a.click(
            fn=handle_vote_a,
            outputs=[image_a, image_b, status_lbl, stats_area]
        )
        
        btn_b.click(
            fn=handle_vote_b,
            outputs=[image_a, image_b, status_lbl, stats_area]
        )
        
        skip_btn.click(
            fn=handle_skip,
            outputs=[image_a, image_b, status_lbl, stats_area]
        )
        
        export_btn.click(
            fn=handle_export,
            outputs=[action_msg]
        )
        
        reset_btn.click(
            fn=handle_reset,
            outputs=[image_a, image_b, status_lbl, action_msg, stats_area]
        )
        
        refresh_stats_btn.click(
            fn=lambda: annotator.get_stats_markdown(),
            outputs=[stats_area]
        )

        if annotator.method == "elo":
            def update_max_comparisons(val):
                annotator.max_comparisons = max(1, int(val))
                annotator._save_state()
                return f"[INFO] Limite de duels mise à jour à : {annotator.max_comparisons}"
                
            max_comparisons_input.change(
                fn=update_max_comparisons,
                inputs=[max_comparisons_input],
                outputs=[action_msg]
            )
            
            apply_norm_btn.click(
                fn=lambda: annotator.force_elo_normalization(),
                outputs=[action_msg]
            )

        # Chargement initial au lancement de l'interface
        def init_load():
            img_a, img_b, status, comp_count, total = annotator.get_ui_state()
            return img_a, img_b, status, annotator.get_stats_markdown()

        iface.load(
            fn=init_load,
            outputs=[image_a, image_b, status_lbl, stats_area]
        )

    return iface


def main():
    """Point d'entrée du script CLI Gradio unifié de comparaisons par paires."""
    parser = argparse.ArgumentParser(
        description="Interface web d'annotation d'images par duels (unifiant Elo et Condorcet)."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/inputs/pairwise_voting",
        help="Chemin vers le répertoire d'images à évaluer"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/outputs/predictions/pairwise_rankings.csv",
        help="Chemin du CSV de classement final de sortie"
    )
    parser.add_argument(
        "--state_file",
        type=str,
        default="data/inputs/pairwise_voting/annotation_state.json",
        help="Chemin du fichier JSON de sauvegarde de l'état des scores"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["elo", "condorcet"],
        default="elo",
        help="Méthode de vote à utiliser (elo ou condorcet)"
    )
    parser.add_argument(
        "--max_comparisons",
        type=int,
        default=100,
        help="Nombre cible de duels recommandés pour le mode Elo"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port réseau d'écoute de l'application web"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Générer un lien d'accès public temporaire"
    )

    args = parser.parse_args()

    # Création du dossier d'images s'il n'existe pas
    if not os.path.exists(args.image_dir):
        print(f"[ATTENTION] Le répertoire spécifié '{args.image_dir}' n'existe pas. Création automatique...")
        os.makedirs(args.image_dir, exist_ok=True)

    # Initialisation de l'annotateur unifié
    annotator = PairwiseImageAnnotator(
        image_dir=args.image_dir,
        output_file=args.output_file,
        state_file=args.state_file,
        method=args.method,
        max_comparisons=args.max_comparisons
    )

    if not annotator.images:
        print(f"[ATTENTION] Aucune image valide détectée sous '{args.image_dir}'.")
        print("Veuillez y copier des images (.jpg, .jpeg, .png, .webp) pour pouvoir annoter.")

    print(f"|----------------------------------------------------")
    print(f"| 🆚 Lancement de l'Interface de Vote Comparatif A/B")
    print(f"|----------------------------------------------------")
    print(f"| - Méthode sélectionnée : {annotator.method.upper()}")
    print(f"| - Répertoire d'images   : {args.image_dir}")
    print(f"| - Fichier d'état JSON   : {args.state_file}")
    print(f"| - Classement final CSV  : {args.output_file}")
    print(f"| - Images détectées      : {len(annotator.images)}")
    print(f"|----------------------------------------------------")

    # Lancement de l'interface Gradio unifiée
    iface = create_interface(annotator)
    iface.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
