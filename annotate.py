import argparse
import csv
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Set

import gradio as gr

# Classes de qualité
CLASSES = {
    0: "0_VeryPoor",
    1: "1_Poor",
    2: "2_Medium",
    3: "3_Good",
    4: "4_VeryGood",
    5: "5_Excellent",
}

# Extensions d'images supportées
DEFAULT_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Verrou pour l'écriture concurrente dans le CSV
csv_lock = threading.Lock()


def find_images(
    image_dir: Path,
    recursive: bool = False,
    exts: Set[str] = None,
    shuffle: bool = False,
) -> List[Path]:
    exts = {e.lower() for e in (exts or DEFAULT_EXTS)}
    if recursive:
        files = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    else:
        files = [p for p in image_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    if shuffle:
        import random

        random.shuffle(files)
    else:
        files.sort()
    return files


def load_existing_annotations(csv_path: Path) -> dict:
    """
    Retourne un dict {image_path_str: label_int} pour les lignes déjà annotées.
    """
    ann = {}
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Compatibilité: attend les colonnes image_path et label
                ip = row.get("image_path")
                lab = row.get("label")
                if ip is None or lab is None:
                    continue
                try:
                    ann[ip] = int(lab)
                except ValueError:
                    continue
    return ann


def write_annotation(csv_path: Path, image_path: Path, label: int):
    """
    Écrit une ligne d'annotation dans le CSV (entête auto si besoin).
    Colonnes: timestamp, image_path, label, class_name
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_lock:  # protège en cas de multiples événements rapides
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "image_path", "label", "class_name"]
            )
            if write_header:
                writer.writeheader()
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
    Renvoie l'index de la prochaine image non annotée à partir de start_idx
    ou -1 s'il n'y en a plus.
    """
    n = len(files)
    i = start_idx
    while i < n:
        if str(files[i]) not in annotated_set:
            return i
        i += 1
    return -1


def previous_index(current_idx: int) -> int:
    return max(0, current_idx - 1)


def build_ui(
    files: List[Path],
    csv_path: Path,
    existing: dict,
):
    annotated_set = set(existing.keys())

    with gr.Blocks(title="Annotation d'images - Qualité") as demo:
        gr.Markdown(
            f"# Annotation d'images (qualité)\n"
            f"Sélectionnez une classe pour chaque image, puis cliquez sur Valider.\n\n"
            f"Classes: {', '.join([f'{k}: {v}' for k, v in CLASSES.items()])}\n\n"
            f"Fichier d'annotations: {csv_path}"
        )

        # État interne
        state_files = gr.State([str(p) for p in files])  # List[str]
        state_idx = gr.State(0)  # int index courant
        state_annotated = gr.State(annotated_set)  # Set[str]

        # Composants
        with gr.Row():
            image_comp = gr.Image(type="filepath", label="Image à annoter", height=512)
            with gr.Column():
                label_comp = gr.Radio(
                    choices=[f"{k}: {CLASSES[k]}" for k in CLASSES],
                    label="Classe",
                    value=None,
                )
                info_txt = gr.Markdown("")
                with gr.Row():
                    btn_back = gr.Button("⬅️ Retour", variant="secondary")
                    btn_skip = gr.Button("⏭️ Ignorer", variant="secondary")
                    btn_validate = gr.Button("✅ Valider", variant="primary")

        def show_index(files_list: List[str], idx: int, annotated: Set[str]) -> Tuple[str, str, int]:
            if not files_list:
                return "", "Aucune image trouvée dans le répertoire.", idx

            # Avancer à la prochaine non annotée si l'index courant est déjà annoté
            n = len(files_list)
            i = idx
            while i < n and files_list[i] in annotated:
                i += 1

            if i >= n:
                return (
                    "",
                    f"✔️ Terminé. Toutes les {n} images sont annotées.",
                    n - 1 if n > 0 else 0,
                )

            img = files_list[i]
            done = len(annotated)
            total = n
            msg = f"Image {i+1}/{total} — Annotées: {done}/{total}"
            return img, msg, i

        def on_load(files_list: List[str], idx: int, annotated: Set[str]):
            img, msg, new_idx = show_index(files_list, idx, annotated)
            return img, gr.update(value=None), msg, new_idx

        demo.load(
            on_load,
            [state_files, state_idx, state_annotated],
            [image_comp, label_comp, info_txt, state_idx],
        )

        def validate(label_value: str, files_list: List[str], idx: int, annotated: Set[str]):
            if not files_list:
                return "", gr.update(value=None), "Aucune image.", idx, annotated

            # Extraire l'entier du label "k: name"
            if not label_value:
                # pas de label sélectionné
                img, msg, new_idx = show_index(files_list, idx, annotated)
                return img, gr.update(), "Veuillez sélectionner une classe, puis Valider.", new_idx, annotated

            try:
                label_int = int(label_value.split(":")[0].strip())
            except Exception:
                label_int = None

            if label_int is None or label_int not in CLASSES:
                img, msg, new_idx = show_index(files_list, idx, annotated)
                return img, gr.update(), "Label invalide. Réessayez.", new_idx, annotated

            current_img = files_list[idx]

            # Écriture CSV
            write_annotation(csv_path, Path(current_img), label_int)

            # Mettre à jour l'ensemble annoté
            annotated.add(current_img)

            # Aller à la prochaine image non annotée
            n = len(files_list)
            next_idx = idx + 1
            while next_idx < n and files_list[next_idx] in annotated:
                next_idx += 1

            # Si plus d'images restantes
            if next_idx >= n:
                return (
                    "",
                    gr.update(value=None),
                    f"✔️ Terminé. Toutes les {n} images sont annotées.",
                    n - 1 if n > 0 else 0,
                    annotated,
                )

            next_img = files_list[next_idx]
            msg = f"Image {next_idx+1}/{n} — Annotées: {len(annotated)}/{n}"
            return next_img, gr.update(value=None), msg, next_idx, annotated

        btn_validate.click(
            validate,
            [label_comp, state_files, state_idx, state_annotated],
            [image_comp, label_comp, info_txt, state_idx, state_annotated],
        )

        def skip(files_list: List[str], idx: int, annotated: Set[str]):
            # Passe à l'image suivante sans enregistrer
            n = len(files_list)
            next_idx = idx + 1
            while next_idx < n and files_list[next_idx] in annotated:
                next_idx += 1
            if next_idx >= n:
                return "", "Plus d'images à afficher. Vous pouvez fermer l'interface.", idx
            next_img = files_list[next_idx]
            msg = f"Image {next_idx+1}/{n} — Annotées: {len(annotated)}/{n}"
            return next_img, msg, next_idx

        btn_skip.click(
            skip,
            [state_files, state_idx, state_annotated],
            [image_comp, info_txt, state_idx],
        )

        def go_back(files_list: List[str], idx: int, annotated: Set[str]):
            # Revient à l'image précédente (ne modifie pas le CSV)
            if not files_list:
                return "", "Aucune image.", idx
            prev_idx = previous_index(idx)
            prev_img = files_list[prev_idx]
            n = len(files_list)
            msg = f"Image {prev_idx+1}/{n} — Annotées: {len(annotated)}/{n}"
            return prev_img, msg, prev_idx

        btn_back.click(
            go_back,
            [state_files, state_idx, state_annotated],
            [image_comp, info_txt, state_idx],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interface Gradio pour annoter des images par niveau de qualité."
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Répertoire contenant les images à annoter.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Chemin du fichier CSV de sortie (par défaut: <image_dir>/annotations.csv).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourir récursivement le répertoire.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Mélanger l'ordre des images.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=None,
        help="Extensions autorisées séparées par des virgules (ex: .jpg,.png,.webp).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port du serveur Gradio (défaut: 7860).",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Adresse d'écoute du serveur (ex: 0.0.0.0 pour accès réseau).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Répertoire introuvable: {image_dir}")

    exts = None
    if args.exts:
        exts = {e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in args.exts.split(",")}

    files = find_images(image_dir, recursive=args.recursive, exts=exts, shuffle=args.shuffle)
    if not files:
        print("Aucune image trouvée. Vérifiez le répertoire, les extensions, ou utilisez --recursive.")
    csv_path = Path(args.output_csv) if args.output_csv else (image_dir / "annotations.csv")

    existing = load_existing_annotations(csv_path)

    # Filtrer les images déjà annotées uniquement si vous souhaitez ne pas les revoir au démarrage
    # Ici on laisse la logique dans l'UI (skip automatique quand on avance).
    demo = build_ui(files, csv_path, existing)
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
