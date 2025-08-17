import click
import PIL.Image
import base64
import io
import pathlib
import loguru
from langchain_ollama import OllamaLLM
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import CommaSeparatedListOutputParser
# import polars as pl

vlmmodel = OllamaLLM(model="qwen2.5vl:7b", temperature=0)
# llmmodel = OllamaLLM(model="gemma3:12b", temperature=0)

def encode_image_to_base64(img:PIL.Image.Image)->str:
    """
        Encode une image en chaîne base64 pour envoyer au VLM
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def extract_details_from_image(img:PIL.Image.Image)->str:
    vlm_with_image_context = vlmmodel.bind(images=[encode_image_to_base64(img)])
    
    vlm_prompt = """
                Analyse l'image fournie en fonction des critères esthétiques suivants, en justifiant chaque point avec des éléments visuels précis. Utilise une échelle de notation (ex. : 1 à 5) pour chaque critère et propose une note globale. Structure ta réponse comme suit :

Composition :

Équilibre des éléments (règle des tiers, symétrie, etc.).
Utilisation de l'espace négatif.
Cohérence des lignes directrices ou des points focaux.


Couleurs et Lumière :

Harmonie des couleurs (palette, contrastes, saturation).
Éclairage (naturel/artificiel, ombres, luminosité).
Effets visuels (flou, profondeur de champ, etc.).


Texture et Détails :

Qualité des textures (réalisme, abstraction).
Netteté et résolution des détails.
Cohérence des matériaux (ex. : métal, tissu, peau).


Émotion et Narratif :

Impact émotionnel (ton, ambiance).
Clarté du message ou de l'intention artistique.
Originalité ou créativité.


Technique et Exécution :

Maîtrise des outils (photographie, dessin, 3D, etc.).
Absence de défauts techniques (artefacts, distorsions).
Cohérence du style (réaliste, surréaliste, minimaliste, etc.).



Donne une note globale sur 10 et propose des suggestions d'amélioration si nécessaire. Sois précis et évite les jugements subjectifs non étayés.
                """
    
    res = vlm_with_image_context.invoke(vlm_prompt)

    loguru.logger.debug(f"{res}")
    return res

@click.command(help="Note esthétique sur une image")
@click.option("-i","--input","img",required=True, help="Image à étudier")
def main(img:str):
    print(img,":",extract_details_from_image(PIL.Image.open(img)))

if __name__ == "__main__":
    main()