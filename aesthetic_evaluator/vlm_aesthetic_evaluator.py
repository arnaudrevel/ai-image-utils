"""
Analyseur esthétique multicritère basé sur un grand modèle de vision local (VLM).

Ce script exploite la bibliothèque LangChain et l'API locale d'Ollama pour exécuter
le modèle de vision 'qwen2.5vl:7b'. Il convertit une image locale en chaîne Base64,
la lie au contexte d'entrée du VLM, et lui soumet une grille d'évaluation esthétique
hautement détaillée (Composition, Couleurs/Lumières, Textures, Émotion, Technique).
Le modèle produit une critique textuelle argumentée et une note synthétique sur 10.
"""

import click
import PIL.Image
import base64
import io
import pathlib
import loguru
from langchain_ollama import OllamaLLM

# Initialisation du modèle VLM via Ollama avec une température de 0 pour assurer la reproductibilité des analyses
vlmmodel = OllamaLLM(model="qwen2.5vl:7b", temperature=0)


def encode_image_to_base64(img: PIL.Image.Image) -> str:
    """
    Encode une image chargée par Pillow en une chaîne brute Base64.
    Cette étape est indispensable pour transmettre l'image sous forme textuelle au VLM via JSON.

    Args:
        img (PIL.Image.Image): Objet image chargé avec Pillow.

    Returns:
        str: Représentation encodée en base64 de l'image.
    """
    buffered = io.BytesIO()
    # Sauvegarde temporaire de l'image en mémoire sous format PNG
    img.save(buffered, format="PNG")
    # Encodage binaire en base64 et conversion en chaîne de caractères UTF-8
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def extract_details_from_image(img: PIL.Image.Image) -> str:
    """
    Injecte l'image dans le contexte du VLM et déclenche l'analyse qualitative.

    Args:
        img (PIL.Image.Image): L'image à étudier.

    Returns:
        str: Critique détaillée et notation structurée renvoyée par le modèle.
    """
    # Liaison (binding) de la chaîne base64 de l'image aux paramètres d'appel du modèle Ollama
    vlm_with_image_context = vlmmodel.bind(images=[encode_image_to_base64(img)])
    
    # Prompt rigoureux et descriptif dictant la structure exacte de la réponse attendue du VLM
    vlm_prompt = """
Analyse l'image fournie en fonction des critères esthétiques suivants, en justifiant chaque point avec des éléments visuels précis. Utilise une échelle de notation (ex. : 1 à 5) pour chaque critère et propose une note globale. Structure ta réponse comme suit :

Composition :
- Équilibre des éléments (règle des tiers, symétrie, etc.).
- Utilisation de l'espace négatif.
- Cohérence des lignes directrices ou des points focaux.

Couleurs et Lumière :
- Harmonie des couleurs (palette, contrastes, saturation).
- Éclairage (naturel/artificiel, ombres, luminosité).
- Effets visuels (flou, profondeur de champ, etc.).

Texture et Détails :
- Qualité des textures (réalisme, abstraction).
- Netteté et résolution des détails.
- Cohérence des matériaux (ex. : métal, tissu, peau).

Émotion et Narratif :
- Impact émotionnel (ton, ambiance).
- Clarté du message ou de l'intention artistique.
- Originalité ou créativité.

Technique et Exécution :
- Maîtrise des outils (photographie, dessin, 3D, etc.).
- Absence de défauts techniques (artefacts, distorsions).
- Cohérence du style (réaliste, surréaliste, minimaliste, etc.).

Donne une note globale sur 10 et propose des suggestions d'amélioration si nécessaire. Sois précis et évite les jugements subjectifs non étayés.
"""
    
    # Invocation synchrone du VLM
    res = vlm_with_image_context.invoke(vlm_prompt)

    # Journalisation du retour brut dans les logs de debug
    loguru.logger.debug(f"Retour du VLM : {res}")
    return res


# Définition de la commande CLI à l'aide de la bibliothèque Click
@click.command(help="Génère une critique et une note esthétique détaillée sur une image via VLM local.")
@click.option(
    "-i", "--input", "img_path", 
    required=True, 
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    help="Chemin de l'image à étudier"
)
def main(img_path: str):
    """
    Point d'entrée du script CLI. Charge l'image et affiche l'analyse textuelle du VLM.
    """
    print(f"Chargement de l'image : {img_path}")
    try:
        # Ouverture du fichier image
        pil_img = PIL.Image.open(img_path)
        # Lancement de l'analyse
        critique = extract_details_from_image(pil_img)
        print("\n=== CRITIQUE ESTHÉTIQUE DE L'IMAGE ===")
        print(f"Image : {img_path}")
        print(critique)
    except Exception as e:
        print(f"❌ Erreur lors du traitement de l'image : {e}")


if __name__ == "__main__":
    main()
