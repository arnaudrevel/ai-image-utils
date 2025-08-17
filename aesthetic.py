from argparse import ArgumentParser
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import tqdm

from PIL import Image

from aesthetic_predictor import predict_aesthetic


def main():
    current_dir = Path(__file__).parent
    parser = ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=current_dir / "cc0")
    parser.add_argument("--output", type=Path, default=current_dir / "scores.html")
    args = parser.parse_args()
    root_dir: Path = args.root_dir

    files = list(root_dir.glob("**/*.png"))
    images = []
    b64_images = []

    for file in tqdm.tqdm(files):
        images.append(Image.open(file))
        b64_images.append(b64encode(file.read_bytes()).decode("utf-8"))
    scores = predict_aesthetic(images).numpy().ravel()

    print(scores.min(),scores.mean())

if __name__ == "__main__":
    main()