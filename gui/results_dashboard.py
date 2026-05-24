"""
Dashboard Gradio unifié pour la visualisation et l'analyse de qualité esthétique d'images.

Ce script lance une interface web Gradio permettant de charger un fichier CSV de résultats
généré par les scripts de prédiction CLI et d'explorer les scores esthétiques de manière
interactive. Deux modes d'affichage sont disponibles via un bouton de bascule :
- Mode Standard : affiche toutes les colonnes du CSV avec formatage des pourcentages.
- Mode Miniatures : affiche uniquement les colonnes essentielles avec des aperçus d'images
  intégrés directement dans le tableau grâce à l'injection de balises HTML <img>.
"""

import gradio as gr
import pandas as pd
import os


def display_csv_results(csv_file, show_thumbnails: bool):
    """
    Lit un fichier CSV de résultats, génère un résumé statistique et prépare
    le DataFrame pour l'affichage selon le mode sélectionné.

    Args:
        csv_file: Chemin du fichier CSV téléversé via l'interface Gradio.
        show_thumbnails (bool): Si True, injecte des miniatures HTML dans le tableau
            en n'affichant que les colonnes image_path et quality_label.
            Si False, affiche toutes les colonnes avec formatage lisible (pourcentages, emojis).

    Returns:
        tuple: (Texte résumé en Markdown, DataFrame formaté, Premier chemin d'image, État JSON brut)
    """
    if csv_file is None:
        return (
            "Veuillez télécharger un fichier CSV de résultats pour afficher les statistiques.",
            pd.DataFrame(),
            None,
            None,
        )

    try:
        # 1. Lecture du fichier CSV
        df = pd.read_csv(csv_file)

        # 2. Stockage du DataFrame original non altéré (pour la navigation par clic de ligne)
        full_df_json = df.to_json(orient="records")

        # 3. Validation de la présence des colonnes requises
        required_cols = ["image_path", "quality_label"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"La colonne requise '{col}' est manquante dans le fichier CSV. "
                    "Assurez-vous que le fichier contient 'image_path' et 'quality_label'."
                )

        # 4. Préparation du DataFrame d'affichage selon le mode actif
        if show_thumbnails:
            # Mode Miniatures : colonnes réduites + injection de balises HTML <img>
            display_df = df[["image_path", "quality_label"]].copy()
            display_df.rename(
                columns={"image_path": "Nom du fichier", "quality_label": "Qualité esthétique"},
                inplace=True,
            )

            # Insertion d'une colonne 'Miniature' entre le nom et la qualité
            display_df.insert(1, "Miniature", "")
            for index, row in display_df.iterrows():
                img_path = str(row["Nom du fichier"])
                if pd.notna(img_path) and os.path.exists(img_path) and os.path.isfile(img_path):
                    display_df.at[index, "Miniature"] = (
                        f'<img src="{img_path}" width="80" height="80" '
                        f'style="object-fit: contain; border-radius: 4px;" '
                        f'title="Aperçu — Cliquez sur la ligne pour agrandir">'
                    )
                else:
                    display_df.at[index, "Miniature"] = "N/A"
        else:
            # Mode Standard : toutes les colonnes avec formatage lisible
            display_df = df.copy()

            if "confidence" in display_df.columns:
                display_df["confidence"] = display_df["confidence"].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else None
                )
            for i in range(6):
                prob_col = f"prob_quality_{i}"
                if prob_col in display_df.columns:
                    display_df[prob_col] = display_df[prob_col].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else None
                    )
            if "success" in display_df.columns:
                display_df["success"] = display_df["success"].apply(
                    lambda x: "✅ Succès" if x else "❌ Échec"
                )

        # 5. Génération du résumé statistique
        total_images = len(df)
        has_success_col = "success" in df.columns
        successful_predictions = (
            df[df["success"] == True].shape[0] if has_success_col else total_images
        )
        failed_predictions = total_images - successful_predictions

        summary_text = f"""
### 📊 Résumé Statistique
- **Images totales analysées :** {total_images}
- **Analyses validées        :** {successful_predictions}
- **Échecs détectés          :** {failed_predictions}
        """

        if successful_predictions > 0 and "predicted_quality" in df.columns:
            quality_source = df[df["success"] == True] if has_success_col else df
            numeric_quality = pd.to_numeric(
                quality_source["predicted_quality"], errors="coerce"
            ).dropna()
            if not numeric_quality.empty:
                summary_text += (
                    f"\n- **Qualité moyenne (sur les succès) :** `{numeric_quality.mean():.2f}/5`"
                )
            else:
                summary_text += "\n- *Note moyenne indisponible.*"

        # 6. Image de prévisualisation par défaut (première ligne valide)
        first_image_path = None
        if not df.empty and "image_path" in df.columns:
            path = df.iloc[0]["image_path"]
            if pd.notna(path) and os.path.exists(path) and os.path.isfile(path):
                first_image_path = path

        return summary_text, display_df, first_image_path, full_df_json

    except Exception as e:
        return (
            f"❌ Erreur lors de la lecture du fichier CSV : {e}\n"
            "Vérifiez que les chemins d'images sont corrects et accessibles.",
            pd.DataFrame(),
            None,
            None,
        )


def show_selected_image(evt: gr.SelectData, full_df_json: str):
    """
    Callback déclenché lors du clic sur une ligne du tableau.
    Retrouve et renvoie le chemin physique de l'image correspondante.

    Args:
        evt (gr.SelectData): Données événementielles de cellule (contient l'index de ligne).
        full_df_json (str): État JSON contenant le DataFrame brut non altéré.

    Returns:
        str: Chemin absolu de l'image à afficher, ou None si introuvable.
    """
    if evt.index is None or full_df_json is None:
        return None

    full_df = pd.read_json(full_df_json, orient="records")
    selected_row_index = evt.index[0]

    if selected_row_index < 0 or selected_row_index >= len(full_df):
        return None

    image_path = full_df.iloc[selected_row_index]["image_path"]

    if pd.notna(image_path) and os.path.exists(image_path) and os.path.isfile(image_path):
        return image_path
    else:
        print(f"Avertissement : Fichier introuvable sur le disque : {image_path}")
        return None


# --- Interface Gradio ---
with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Tableau de Bord Esthétique")
    gr.Markdown(
        "Téléchargez un fichier CSV généré par un script de prédiction CLI pour explorer "
        "interactivement les scores et visualiser les images associées.\n\n"
        "*Activez le mode Miniatures pour afficher des aperçus directement dans le tableau.*"
    )

    with gr.Row():
        csv_upload_input = gr.File(
            label="📤 Charger le fichier CSV des résultats",
            type="filepath",
            file_types=[".csv"],
        )
        thumbnails_toggle = gr.Checkbox(
            label="🖼️ Afficher les miniatures",
            value=False,
            info="Active l'affichage d'aperçus d'images dans le tableau (colonnes réduites à l'essentiel).",
        )

    summary_output = gr.Markdown("...")

    # État mémoire persistant : stocke le DataFrame brut pour la navigation par clic
    full_dataframe_state = gr.State(value=None)

    with gr.Row():
        dataframe_output = gr.DataFrame(
            label="📋 Résultats d'analyse (Cliquez sur une ligne pour l'agrandir à droite)",
            interactive=True,
            wrap=True,
            datatype="str",  # Requis pour le rendu HTML des miniatures
        )
        image_preview_output = gr.Image(
            label="Image en taille réelle",
            type="filepath",
            height=450,
            width=650,
            interactive=False,
        )

    shared_outputs = [summary_output, dataframe_output, image_preview_output, full_dataframe_state]

    # Chargement d'un nouveau fichier CSV
    csv_upload_input.change(
        fn=display_csv_results,
        inputs=[csv_upload_input, thumbnails_toggle],
        outputs=shared_outputs,
    )

    # Bascule du mode miniatures (utilise le fichier déjà chargé)
    thumbnails_toggle.change(
        fn=display_csv_results,
        inputs=[csv_upload_input, thumbnails_toggle],
        outputs=shared_outputs,
    )

    # Clic sur une ligne du tableau → affichage de l'image en grand
    dataframe_output.select(
        fn=show_selected_image,
        inputs=[full_dataframe_state],
        outputs=[image_preview_output],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        inbrowser=True,
    )
