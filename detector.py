"""Классификация JPEG по гистограммам яркостных коэффициентов ДКП."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn
from torch import nn

ROOT = Path(__file__).resolve().parent
FREQUENCIES = ((0, 1), (1, 0), (2, 0), (1, 1), (0, 2),
               (0, 3), (1, 2), (2, 1), (3, 0))
N_BINS = 123
N_HIST = len(FREQUENCIES) * N_BINS
CLASSES = ("однократное сжатие", "двукратное сжатие")


def features(source):
    """Путь или JPEG в BytesIO → 9 гистограмм и 64 шага квантования."""
    with Image.open(source) as image:
        if image.format != "JPEG":
            raise ValueError("требуется файл JPEG")
        if image.mode not in ("L", "RGB") or (image.mode == "RGB" and image.layer[0][0] != 1):
            raise ValueError("поддерживаются JPEG в градациях серого и YCbCr")
        w, h = image.size
        if min(w, h) < 256:
            raise ValueError("размер изображения должен быть не меньше 256 × 256")
        table = image.quantization[image.layer[0][3]]
        if max(table) > 255:
            raise ValueError("поддерживаются шаги квантования до 255")
        image.draft("L", (w, h))  # Получаем Y из JPEG без пересчёта через RGB.
        x, y = (w - 256) // 16 * 8, (h - 256) // 16 * 8
        pixels = np.asarray(image.crop((x, y, x + 256, y + 256)), dtype=np.float32)
    blocks = (pixels - 128).reshape(32, 8, 32, 8)
    coeff = dctn(blocks, axes=(1, 3), norm="ortho")
    rows, cols = zip(*FREQUENCIES)
    values = coeff[:, rows, :, cols].reshape(len(rows), -1)
    bins = np.floor(values + 0.5).clip(-61, 61).astype(np.int32) + 61
    bins += (np.arange(len(rows)) * N_BINS)[:, None]
    counts = np.bincount(bins.ravel(), minlength=N_HIST)
    return np.concatenate((counts / 1024, np.asarray(table) / 255)).astype(np.float32)


class HistogramCNN(nn.Module):
    def __init__(self, channels=(16, 32)):
        super().__init__()
        first, second = channels
        self.convolutions = nn.Sequential(
            nn.Conv1d(9, first, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(first, second, 3, padding=1), nn.ReLU(), nn.AvgPool1d(2),
        )
        self.classifier = nn.Sequential(nn.Linear(second * 30 + 64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        histograms = x[:, :N_HIST].reshape(-1, 9, N_BINS)
        patterns = self.convolutions(histograms).flatten(1)
        return self.classifier(torch.cat((patterns, x[:, N_HIST:]), dim=1))


def make_model(name, dropout=0.0, channels=(16, 32)):
    if name == "CNN":
        return HistogramCNN(channels)
    if name == "MLP":
        layers = []
        for inputs, outputs in ((N_HIST + 64, 64), (64, 32)):
            layers.extend([nn.Linear(inputs, outputs), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers, nn.Linear(32, 2))
    raise ValueError(f"неизвестная архитектура {name}")


def load_detector(weights=None):
    if weights is None:
        choice = json.loads((ROOT / "weights" / "choice.json").read_text())
        weights = ROOT / "weights" / choice["file"]
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    model = make_model(saved["model"], dropout=saved.get("dropout", 0.0))
    model.load_state_dict(saved["state"])
    return model.eval(), saved["mean"], saved["scale"]


@torch.inference_mode()
def classify(source, detector):
    model, mean, scale = detector
    x = (torch.from_numpy(features(source)) - mean) / scale
    score = model(x[None]).softmax(dim=1)[0, 1].item()
    return int(score >= 0.5), score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JPEG-файл или каталог с фотографиями")
    parser.add_argument("--weights", type=Path, help="путь к обученным весам")
    parser.add_argument("--csv", type=Path, help="сохранить результаты в CSV")
    args = parser.parse_args()
    directory = args.path.is_dir()
    if directory:
        photos = sorted(p for p in args.path.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
        if not photos:
            parser.exit(1, "В каталоге нет JPEG-файлов.\n")
    else:
        photos = [args.path]
    torch.set_num_threads(1)
    detector = load_detector(args.weights)
    rows = []
    for photo in photos:
        try:
            label, score = classify(photo, detector)
            print(f"{photo}: {CLASSES[label]} (оценка класса «двукратное» {score:.3f})")
            rows.append((photo, label, score, ""))
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            print(f"{photo}: {exc}")
            rows.append((photo, "", "", str(exc)))
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("path", "class", "score_double", "error"))
            writer.writerows(rows)
    return int(all(row[3] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
