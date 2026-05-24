"""
Interface Graphique Gradio pour l'Annotation Manuelle de la Qualité d'Images.

Ce script lance un serveur web Gradio interactif local. Il permet à l'utilisateur
de passer en revue des images situées dans un dossier donné, de leur attribuer une
note de qualité esthétique de 0 à 5, et d'enregistrer ces annotations de manière
incrémentale et thread-safe dans un fichier CSV structuré.
"""

import argparse
import csv
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Set

import gradio as gr

# Classes de qualité prédéfinies pour l'annotation
CLASSES = {
    0: "0_VeryPoor",
    1: "1_Poor",
    2: "2_Medium",
    3: "3_Good",
    4: "4_VeryGood",
    5: "5_Excellent",
}

# Extensions d'images par défaut prises en charge
DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Verrou (threading Lock) pour protéger l'écriture concurrente dans le fichier CSV
csv_lock = threading.Lock()


def find_images(
    image_dir: Path,
    recursive: bool = False,
    exts: Set[str] = None,
    shuffle: bool = False,
) -> List[Path]:
    """
    Parcourt le répertoire fourni pour trouver tous les fichiers d'images pris en charge.

    Args:
        image_dir (Path): Chemin du dossier source.
        recursive (bool): Si True, scanne récursivement tous les sous-dossiers.
        exts (Set[str]): Ensemble d'extensions à autoriser. Si None, utilise DEFAULT_EXTS.
        shuffle (bool): Si True, mélange aléatoirement l'ordre des images trouvées.

    Returns:
        List[Path]: Liste triée ou mélangée des chemins de fichiers d'images trouvés.
    """
    exts = {e.lower() for e in (exts or DEFAULT_EXTS)}
    
    # Recherche récursive ou non des fichiers d'images
    if recursive:
        files = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    else:
        files = [p for p in image_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
        
    # Tri ou mélange de la liste
    if shuffle:
        import random
        random.shuffle(files)
    else:
        files.sort()
        
    return files


def load_existing_annotations(csv_path: Path) -> dict:
    """
    Charge les annotations existantes depuis le fichier CSV pour éviter les doublons.

    Args:
        csv_path (Path): Chemin vers le fichier d'annotations CSV.

    Returns:
        dict: Dictionnaire associant le chemin de l'image (str) à son label entier (int).
    """
    ann = {}
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Lecture des colonnes clés
                    ip = row.get("image_path")
                    lab = row.get("label")
                    if ip is None or lab is None:
                        continue
                    try:
                        ann[ip] = int(lab)
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Attention lors de la lecture des annotations existantes : {e}")
    return ann


def write_annotation(csv_path: Path, image_path: Path, label: int):
    """
    Écrit une nouvelle ligne d'annotation dans le CSV de manière thread-safe.
    Crée automatiquement les répertoires et l'entête du CSV si nécessaire.

    Args:
        csv_path (Path): Chemin vers le fichier d'annotations de sortie.
        image_path (Path): Chemin absolu de l'image annotée.
        label (int): Entier de qualité (0 à 5).
    """
    # Création automatique du dossier parent s'il n'existe pas
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    
    # Utilisation du verrou pour empêcher deux requêtes HTTP rapides d'altérer le fichier simultanément
    with csv_lock:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "image_path", "label", "class_name"]
            )
            if write_header:
                writer.writeheader()
            
            # Écriture de l'enregistrement avec horodatage UTC ISO
            writer.writerow(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "image_path": str(image_path),
                    "label": int(label),
                    "class_name": CLASSES[int(label)],
                }
            )


def next_unannotated_index(files: List[Path], annotated_set: Set[str], start_idx: int) -> int:
    """
    Recherche le premier index d'image non annotée à partir d'un index de départ donné.

    Args:
        files (List[Path]): Liste complète des fichiers d'images.
        annotated_set (Set[str]): Ensemble des chemins de fichiers déjà annotés.
        start_idx (int): Index de départ pour la recherche.

    Returns:
        int: Index de la prochaine image non annotée, ou -1 s'il n'en reste plus.
    """
    n = len(files)
    i = start_idx
    while i < n:
        if str(files[i]) not in annotated_set:
            return i
        i += 1
    return -1


def previous_index(current_idx: int) -> int:
    """
    Calcule l'index précédent avec une borne inférieure de sécurité à 0.

    Args:
        current_idx (int): Index courant.

    Returns:
        int: Index précédent ou 0.
    """
    return max(0, current_idx - 1)


def build_ui(
    files: List[Path],
    csv_path: Path,
    existing: dict,
) -> gr.Blocks:
    """
    Construit la structure de l'interface utilisateur Gradio.

    Args:
        files (List[Path]): Liste de toutes les images chargées.
        csv_path (Path): Fichier CSV d'enregistrement des résultats.
        existing (dict): Annotations précédemment chargées.

    Returns:
        gr.Blocks: Instance de l'interface Gradio Blocks prête à être lancée.
    """
    annotated_set = set(existing.keys())

    # Création du bloc principal de l'UI avec thème et titre
    with gr.Blocks(title="Annotation d'images - Qualité") as demo:
        gr.Markdown(
            f"# 🖼️ Annotation d'images (Qualité Esthétique)\n"
            f"Sélectionnez la note de qualité correspondante à l'image affichée, puis cliquez sur **Valider**.\n\n"
            f"**Échelle de notation** : {', '.join([f'{k}: {v}' for k, v in CLASSES.items()])}\n\n"
            f"💾 Fichier d'annotations : `{csv_path}`"
        )

        # États internes Gradio servant à maintenir le statut entre les requêtes asynchrones
        state_files = gr.State([str(p) for p in files])  # Liste de chaînes de caractères de tous les chemins d'images
        state_idx = gr.State(0)                          # Index de l'image actuellement affichée
        state_annotated = gr.State(annotated_set)        # Ensemble des chemins déjà annotés

        # Layout graphique
        with gr.Row():
            # Composant d'affichage d'image
            image_comp = gr.Image(type="filepath", label="Image à annoter", height=512)
            with gr.Column():
                # Boutons radio de notation 0 à 5
                label_comp = gr.Radio(
                    choices=[f"{k}: {CLASSES[k]}" for k in CLASSES],
                    label="Niveau de qualité",
                    value=None,
                )
                info_txt = gr.Markdown("")
                # Ligne de contrôle
                with gr.Row():
                    btn_back = gr.Button("⬅️ Retour", variant="secondary")
                    btn_skip = gr.Button("⏭️ Ignorer", variant="secondary")
                    btn_validate = gr.Button("✅ Valider", variant="primary")

        def show_index(files_list: List[str], idx: int, annotated: Set[str]) -> Tuple[str, str, int]:
            """Calcule et formate les données de l'image courante à afficher."""
            if not files_list:
                return "", "Aucune image trouvée dans le répertoire.", idx

            # Avancer automatiquement si l'image à cet index a déjà été annotée lors d'une session précédente
            n = len(files_list)
            i = idx
            while i < n and files_list[i] in annotated:
                i += 1

            # Si toutes les images ont été passées en revue
            if i >= n:
                return (
                    "",
                    f"✔️ Terminé. Toutes les {n} images ont été annotées.",
                    n - 1 if n > 0 else 0,
                )

            img = files_list[i]
            done = len(annotated)
            total = n
            msg = f"Image **{i+1}/{total}** — Annotées : **{done}/{total}**"
            return img, msg, i

        def on_load(files_list: List[str], idx: int, annotated: Set[str]):
            """Handler exécuté lors du chargement initial de la page."""
            img, msg, new_idx = show_index(files_list, idx, annotated)
            return img, gr.update(value=None), msg, new_idx

        # Chargement automatique au démarrage du navigateur
        demo.load(
            on_load,
            [state_files, state_idx, state_annotated],
            [image_comp, label_comp, info_txt, state_idx],
        )

        def validate(label_value: str, files_list: List[str], idx: int, annotated: Set[str]):
            """Enregistre le choix utilisateur, écrit dans le CSV et passe à l'image suivante."""
            if not files_list:
                return "", gr.update(value=None), "Aucune image.", idx, annotated

            # Vérification qu'un bouton radio a été coché
            if not label_value:
                img, msg, new_idx = show_index(files_list, idx, annotated)
                return img, gr.update(), "⚠️ Veuillez sélectionner une classe, puis cliquez sur Valider.", new_idx, annotated

            try:
                # Extraction de l'entier (le chiffre avant les deux points)
                label_int = int(label_value.split(":")[0].strip())
            except Exception:
                label_int = None

            if label_int is None or label_int not in CLASSES:
                img, msg, new_idx = show_index(files_list, idx, annotated)
                return img, gr.update(), "⚠️ Label invalide. Veuillez réessayer.", new_idx, annotated

            current_img = files_list[idx]

            # Écriture asynchrone sécurisée dans le fichier CSV
            write_annotation(csv_path, Path(current_img), label_int)

            # Enregistrement dans la liste des images traitées pour cette session
            annotated.add(current_img)

            # Recherche de la prochaine image non annotée
            n = len(files_list)
            next_idx = idx + 1
            while next_idx < n and files_list[next_idx] in annotated:
                next_idx += 1

            # Si on a atteint la fin
            if next_idx >= n:
                return (
                    "",
                    gr.update(value=None),
                    f"✔️ Terminé. Toutes les {n} images ont été annotées !",
                    n - 1 if n > 0 else 0,
                    annotated,
                )

            next_img = files_list[next_idx]
            msg = f"Image **{next_idx+1}/{n}** — Annotées : **{len(annotated)}/{n}**"
            return next_img, gr.update(value=None), msg, next_idx, annotated

        # Clic sur le bouton de validation
        btn_validate.click(
            validate,
            [label_comp, state_files, state_idx, state_annotated],
            [image_comp, label_comp, info_txt, state_idx, state_annotated],
        )

        def skip(files_list: List[str], idx: int, annotated: Set[str]):
            """Permet d'ignorer l'image courante sans enregistrer de note dans le CSV."""
            n = len(files_list)
            next_idx = idx + 1
            while next_idx < n and files_list[next_idx] in annotated:
                next_idx += 1
            if next_idx >= n:
                return "", "Plus d'images à afficher. Vous pouvez fermer l'interface.", idx
            next_img = files_list[next_idx]
            msg = f"Image **{next_idx+1}/{n}** — Annotées : **{len(annotated)}/{n}** (précédente ignorée)"
            return next_img, msg, next_idx

        btn_skip.click(
            skip,
            [state_files, state_idx, state_annotated],
            [image_comp, info_txt, state_idx],
        )

        def go_back(files_list: List[str], idx: int, annotated: Set[str]):
            """Permet de revenir à l'image précédente pour réévaluation."""
            if not files_list:
                return "", "Aucune image.", idx
            prev_idx = previous_index(idx)
            prev_img = files_list[prev_idx]
            n = len(files_list)
            msg = f"Image **{prev_idx+1}/{n}** — Annotées : **{len(annotated)}/{n}**"
            return prev_img, msg, prev_idx

        btn_back.click(
            go_back,
            [state_files, state_idx, state_annotated],
            [image_comp, info_txt, state_idx],
        )

    return demo


def parse_args():
    """Configure et extrait les paramètres d'exécution CLI."""
    parser = argparse.ArgumentParser(
        description="Interface interactive Gradio pour l'annotation qualitative rapide d'images."
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Chemin absolue du répertoire contenant les images à annoter.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Chemin optionnel du fichier CSV de sortie (par défaut: annotations.csv dans le dossier des images).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourir récursivement les sous-répertoires à la recherche d'images.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mélanger l'ordre d'affichage des images pour casser les séquences temporelles.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=None,
        help="Extensions de fichiers autorisées séparées par des virgules (ex: jpg,png,webp).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port du serveur local Gradio (par défaut: 7860).",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Adresse IP d'écoute réseau du serveur (ex: '0.0.0.0' pour écouter sur tout le réseau local).",
    )
    return parser.parse_args()


def main():
    """Point d'entrée principal pour démarrer l'application Gradio."""
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Répertoire introuvable: {image_dir}")

    exts = None
    if args.exts:
        # Normalisation automatique des extensions (ex: 'jpg' -> '.jpg')
        exts = {e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in args.exts.split(",")}

    # Scanner les images
    files = find_images(image_dir, recursive=args.recursive, exts=exts, shuffle=args.shuffle)
    if not files:
        print("⚠️ Aucune image trouvée. Vérifiez le répertoire ou spécifiez les extensions / l'option récursive.")
        return

    # Fichier CSV de sortie par défaut
    csv_path = Path(args.output_csv) if args.output_csv else (image_dir / "annotations.csv")

    # Charger les annotations déjà réalisées lors de précédentes sessions
    existing = load_existing_annotations(csv_path)

    # Initialisation et démarrage du serveur Gradio
    demo = build_ui(files, csv_path, existing)
    print(f"Démarrage de l'interface d'annotation sur http://{args.server_name}:{args.server_port}/")
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
