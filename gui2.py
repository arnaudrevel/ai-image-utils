import gradio as gr
import pandas as pd
import os
import json 
from datetime import datetime # Pour le résumé de la qualité moyenne

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
        full_df_json = df.to_json(orient='records') 
        
        # --- Préparation du DataFrame pour l'affichage : Sélection et renommage des colonnes ---
        # S'assurer que les colonnes existent avant de les sélectionner
        required_cols = ['image_path', 'quality_label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"La colonne '{col}' est manquante dans le fichier CSV. "
                                 "Veuillez vous assurer que le CSV contient au moins 'image_path' et 'quality_label'.")

        # Sélectionner les colonnes nécessaires et faire une copie
        display_df = df[['image_path', 'quality_label']].copy()
        
        # Renommer les colonnes pour l'affichage
        display_df.rename(columns={
            'image_path': 'Nom du fichier', 
            'quality_label': 'Qualité esthétique'
        }, inplace=True)

        # --- AJOUT DE LA COLONNE MINIATURE ---
        # Insérer la nouvelle colonne 'Miniature' à la deuxième position (index 1)
        # après 'Nom du fichier'
        display_df.insert(1, 'Miniature', '') 
        
        for index, row in display_df.iterrows():
            # Utiliser le nouveau nom de colonne 'Nom du fichier' pour récupérer le chemin
            img_path = str(row['Nom du fichier']) 
            if pd.notna(img_path) and os.path.exists(img_path) and os.path.isfile(img_path):
                # Crée une balise HTML <img> pour la miniature
                display_df.at[index, 'Miniature'] = f'<img src="{img_path}" width="80" height="80" style="object-fit: contain; border-radius: 4px;" title="Miniature - Cliquez sur la ligne pour agrandir">'
            else:
                display_df.at[index, 'Miniature'] = "N/A"
        # --- FIN AJOUT DE LA COLONNE MINIATURE ---

        # --- Génération du résumé statistique ---
        # Le résumé utilise toujours le DataFrame 'df' original pour ses calculs complets
        total_images = len(df) 
        successful_predictions = df[df['success'] == True].shape[0] 
        failed_predictions = total_images - successful_predictions
        
        summary_text = f"""
        ### 📊 Résumé des Résultats
        - **Total d'images analysées:** {total_images}
        - **Prédictions réussies:** {successful_predictions}
        - **Prédictions échouées:** {failed_predictions}
        """
        
        if successful_predictions > 0:
            numeric_quality = pd.to_numeric(df[df['success'] == True]['predicted_quality'], errors='coerce').dropna()
            if not numeric_quality.empty:
                avg_quality = numeric_quality.mean()
                summary_text += f"\n- **Qualité moyenne (sur les succès):** `{avg_quality:.2f}/5`"
            else:
                summary_text += f"\n- *Impossible de calculer la qualité moyenne (pas de données numériques valides).* "

        first_image_path = None
        if not df.empty and 'image_path' in df.columns:
            if pd.notna(df.iloc[0]['image_path']) and os.path.exists(df.iloc[0]['image_path']):
                first_image_path = df.iloc[0]['image_path']

        return summary_text, display_df, first_image_path, full_df_json

    except Exception as e:
        return (
            f"❌ Erreur lors de la lecture du fichier CSV: {e}\nAssurez-vous que le fichier est au bon format "
            f"et que les chemins d'images sont accessibles par le serveur Gradio (idéalement relatifs).", 
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
        return None 
    
    full_df = pd.read_json(full_df_json, orient='records')
    
    selected_row_index = evt.index[0] 
    
    if selected_row_index < 0 or selected_row_index >= len(full_df):
        return None 
    
    image_path = full_df.iloc[selected_row_index]['image_path']
    
    if pd.notna(image_path) and os.path.exists(image_path) and os.path.isfile(image_path):
        return image_path
    else:
        print(f"Image non trouvée au chemin : {image_path}")
        return None 

# Création de l'interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Visualisation des Résultats de Qualité Esthétique")
    gr.Markdown(
        "Bienvenue ! Téléchargez un fichier CSV généré par le script `aesthetic_cli.py` "
        "pour afficher et explorer les résultats de l'analyse de qualité d'image."
        "\n\n**Note sur les miniatures :** Pour que les miniatures s'affichent, les chemins d'images "
        "dans le CSV doivent être accessibles par le serveur Gradio (idéalement, placez vos images "
        "dans le même répertoire que ce script ou un sous-répertoire)."
    )

    with gr.Row():
        csv_upload_input = gr.File(
            label="📤 Télécharger le fichier CSV des résultats", 
            type="filepath", 
            file_types=[".csv"]
        )
    
    summary_output = gr.Markdown("...")

    full_dataframe_state = gr.State(value=None) 

    with gr.Row():
        dataframe_output = gr.DataFrame(
            label="📋 Résultats (cliquez sur une ligne pour visualiser l'image)",
            interactive=True,
            wrap=True,
            # render_as="html",  # Cette option n'existe pas
            column_widths=["1fr", "100px", "auto"],
            # Autres options disponibles :
            headers=None,  # Liste des en-têtes de colonnes
            datatype="str",  # Type de données par défaut
        )

        image_preview_output = gr.Image(
            label="Visualisation de l'image sélectionnée", 
            type="filepath", 
            height=400, 
            width=600,  
            interactive=False 
        )

    csv_upload_input.change(
        fn=display_csv_results,
        inputs=csv_upload_input,
        outputs=[summary_output, dataframe_output, image_preview_output, full_dataframe_state]
    )

    dataframe_output.select(
        fn=show_selected_image,
        inputs=[full_dataframe_state],
        outputs=[image_preview_output],
        queue=False 
    )

if __name__ == "__main__":
    demo.launch(
        share=False, 
        inbrowser=True 
    )
