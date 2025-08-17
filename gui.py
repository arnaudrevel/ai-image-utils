import gradio as gr
import pandas as pd
import os
import json # Pour sérialiser/désérialiser le DataFrame complet

# Fonction pour lire le CSV et préparer les données pour l'affichage
def display_csv_results(csv_file):
    """
    Lit un fichier CSV généré par aesthetic_cli.py et prépare son contenu pour l'affichage Gradio.
    Retourne également le DataFrame complet en JSON pour le passer au gr.State.
    """
    if csv_file is None:
        # Retourne une valeur par défaut pour tous les outputs
        return (
            "Veuillez télécharger un fichier CSV pour afficher les résultats.", 
            pd.DataFrame(), # DataFrame vide
            None, # Image preview (aucun chemin)
            None # État JSON vide
        )
    
    try:
        # Lire le fichier CSV
        df = pd.read_csv(csv_file.name)
        
        # --- Stocker le DataFrame original avant tout formatage pour le selecteur d'image ---
        # Convertir en JSON string pour stocker dans un gr.State
        # Utiliser `orient='records'` pour une conversion facile en liste de dictionnaires
        full_df_json = df.to_json(orient='records') 
        
        # --- Formattage des colonnes pour une meilleure lisibilité dans le DataFrame affiché ---
        display_df = df.copy() # Travailler sur une copie pour le display

        # Formatter la colonne 'confidence' en pourcentage
        if 'confidence' in display_df.columns:
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else None)
        
        # Formatter les colonnes de probabilités en pourcentages
        for i in range(6): # Pour les qualités de 0 à 5
            prob_col = f'prob_quality_{i}'
            if prob_col in display_df.columns:
                display_df[prob_col] = display_df[prob_col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else None)

        # Remplacer les valeurs booléennes par des emojis pour 'success'
        if 'success' in display_df.columns:
            display_df['success'] = display_df['success'].apply(lambda x: "✅ Succès" if x else "❌ Échec")
        
        # --- Génération du résumé statistique ---
        total_images = len(df) # Utiliser le DF original pour les calculs
        successful_predictions = df[df['success'] == True].shape[0] # Utiliser le booléen original
        failed_predictions = total_images - successful_predictions
        
        summary_text = f"""
        ### 📊 Résumé des Résultats
        - **Total d'images analysées:** {total_images}
        - **Prédictions réussies:** {successful_predictions}
        - **Prédictions échouées:** {failed_predictions}
        """
        
        # Calculer la qualité moyenne uniquement sur les prédictions réussies
        if successful_predictions > 0:
            numeric_quality = pd.to_numeric(df[df['success'] == True]['predicted_quality'], errors='coerce').dropna()
            if not numeric_quality.empty:
                avg_quality = numeric_quality.mean()
                summary_text += f"\n- **Qualité moyenne (sur les succès):** `{avg_quality:.2f}/5`"
            else:
                summary_text += f"\n- *Impossible de calculer la qualité moyenne (pas de données numériques valides).* "

        # Retourne le résumé, le DataFrame formaté pour l'affichage, 
        # une image par défaut (la première si disponible, sinon None), et le DF complet en JSON
        first_image_path = None
        if not df.empty and 'image_path' in df.columns:
            # S'assurer que le chemin est valide avant de le retourner
            if os.path.exists(df.iloc[0]['image_path']):
                first_image_path = df.iloc[0]['image_path']

        return summary_text, display_df, first_image_path, full_df_json

    except Exception as e:
        return (
            f"❌ Erreur lors de la lecture du fichier CSV: {e}\nAssurez-vous que le fichier est au bon format.", 
            pd.DataFrame(),
            None,
            None
        )

# Fonction pour afficher l'image sélectionnée dans le DataFrame
def show_selected_image(evt: gr.SelectData, full_df_json: str):
    """
    Callback déclenché lors de la sélection d'une ligne dans le DataFrame.
    Affiche l'image correspondant à la ligne sélectionnée.
    """
    if evt.index is None or full_df_json is None:
        return None # Rien n'est sélectionné ou pas de données
    
    # Charger le DataFrame complet à partir de l'état JSON
    full_df = pd.read_json(full_df_json, orient='records')
    
    selected_row_index = evt.index[0] # L'index de la ligne sélectionnée
    
    if selected_row_index < 0 or selected_row_index >= len(full_df):
        return None # Index hors limites
    
    image_path = full_df.iloc[selected_row_index]['image_path']
    
    # Vérifier si le fichier image existe
    if os.path.exists(image_path) and os.path.isfile(image_path):
        return image_path
    else:
        print(f"Image non trouvée au chemin : {image_path}")
        return None # Ou un chemin vers une image par défaut "non trouvée"

# Création de l'interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Visualisation des Résultats de Qualité Esthétique")
    gr.Markdown(
        "Bienvenue ! Téléchargez un fichier CSV généré par le script `aesthetic_cli.py` "
        "pour afficher et explorer les résultats de l'analyse de qualité d'image."
    )

    with gr.Row():
        # Composant pour le téléchargement du fichier
        csv_upload_input = gr.File(
            label="📤 Télécharger le fichier CSV des résultats", 
            type="filepath", 
            file_types=[".csv"]
        )
    
    # Composant pour afficher le résumé
    summary_output = gr.Markdown(
        "..." # Texte initial
    )

    # État caché pour stocker le DataFrame complet sans formatage (pour récupérer les chemins d'image originaux)
    full_dataframe_state = gr.State(value=None) 

    with gr.Row():
        # Composant pour afficher le tableau des résultats détaillés
        dataframe_output = gr.DataFrame(
            label="📋 Résultats détaillés des images (cliquez sur une ligne pour visualiser l'image)",
            interactive=True, # Permet le tri et la recherche dans le tableau
            wrap=True, # Enroule le texte dans les cellules
            # Ajustez column_widths si vous avez beaucoup de colonnes pour améliorer la lisibilité
            # column_widths=["1fr", "auto", "auto", "auto", "auto", "auto", "auto", "auto", "auto", "auto", "auto", "auto"],
        )

        # Composant pour afficher la prévisualisation de l'image sélectionnée
        image_preview_output = gr.Image(
            label="Visualisation de l'image sélectionnée", 
            type="filepath", # Indique que l'entrée est un chemin de fichier
            height=400, # Hauteur fixe pour une meilleure présentation
            width=600,  # Largeur fixe
            interactive=False # L'utilisateur ne peut pas éditer l'image ici
        )

    # Lier le téléchargement du fichier à la fonction de traitement
    # On met à jour le summary, le dataframe affiché et l'état caché avec le DF complet
    csv_upload_input.change(
        fn=display_csv_results,
        inputs=csv_upload_input,
        outputs=[summary_output, dataframe_output, image_preview_output, full_dataframe_state]
    )

    # Lier la sélection d'une ligne dans le DataFrame à la fonction d'affichage d'image
    dataframe_output.select(
        fn=show_selected_image,
        inputs=[full_dataframe_state], # Passe l'état contenant le DataFrame complet
        outputs=[image_preview_output],
        queue=False # Ne pas mettre en file d'attente pour une réactivité immédiate
    )

# Lancer l'application Gradio
if __name__ == "__main__":
    demo.launch(
        share=False, # Mettez à True pour créer un lien public temporaire (utile pour le partage)
        inbrowser=True # Ouvre l'application automatiquement dans votre navigateur
    )
