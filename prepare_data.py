"""Подготовить воспроизводимые подвыборки из архивов DJPEG и UCID."""
import argparse
import csv
import hashlib
import io
import re
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from detector import features

DATA = Path("data")
PARK_BYTES = 26180076217
PARK_SHA = "6e58894a5752fbc9afece9b0daacb01f63a6d0704350846105caea8e3512e35a"
QUALITY = (55, 65, 75, 85, 95)
SIZES = {"train": 15000, "val": 2500, "test": 2500}  # Число пар.


def wanted_ids():
    rng = np.random.default_rng(2026)
    ids = np.arange(1, 570216)
    groups = rng.permutation(np.unique((ids - 1) // 1000))
    parts = np.split(groups, [428, 500])
    selected = {}
    for (name, count), groups in zip(SIZES.items(), parts):
        pool = ids[np.isin((ids - 1) // 1000, groups)]
        for rank, block in enumerate(rng.choice(pool, count + 100, replace=False)):
            selected[int(block)] = (name, rank)
    return selected


def write_manifest(archive):
    chosen = wanted_ids()
    images = DATA / "djpeg"
    images.mkdir(exist_ok=True)
    entries = {}
    pattern = re.compile(r"jpeg_data/(single|double)/(\d{2})/([SD])_(\d+)_(\d+)(?:_(\d+))?\.jpg")
    if archive.stat().st_size != PARK_BYTES:
        raise ValueError("архив DJPEG скачан не полностью")
    with archive.open("rb") as raw:
        if hashlib.file_digest(raw, "sha256").hexdigest() != PARK_SHA:
            raise ValueError("контрольная сумма DJPEG не совпала")
    tables = loadmat(DATA / "q_table_list.mat")["q_table"][:, 0]
    with tarfile.open(archive, "r|gz", bufsize=2**20) as tar:
        for member in tar:
            tar.members.clear()
            match = pattern.fullmatch(member.name)
            if not match:
                continue
            folder, group, prefix, block, first, last = match.groups()
            block = int(block)
            if block not in chosen:
                continue
            label = int(folder == "double")
            assert (prefix == "D") == bool(label) == (last is not None)
            first, final = int(first), int(last or first)
            content = tar.extractfile(member).read()
            with Image.open(io.BytesIO(content)) as image:
                assert image.format == "JPEG" and image.size == (256, 256)
                table = image.quantization[image.layer[0][3]]
                assert np.array_equal(table, tables[final - 1].ravel())
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            path = images / Path(member.name).name
            path.write_bytes(content)
            key = (block, label)
            assert key not in entries
            entries[key] = dict(path=path.as_posix(), label=label, split=chosen[block][0],
                                block=block, folder=group, first_table=first, final_table=final,
                                sha256=hashlib.sha256(content).hexdigest(), pixel_sha256=pixel_hash)
    rows = []
    previous_pixels = set()
    for split, count in SIZES.items():
        pairs = [b for b in chosen if chosen[b][0] == split and (b, 0) in entries and (b, 1) in entries]
        pairs.sort(key=lambda b: chosen[b][1])
        pairs = [b for b in pairs if all(entries[b, label]["pixel_sha256"] not in previous_pixels for label in (0, 1))]
        assert len(pairs) >= count, (split, len(pairs))
        for block in pairs[:count]:
            single, double = entries[block, 0], entries[block, 1]
            rows.extend((single, double))
        previous_pixels.update(row["pixel_sha256"] for row in rows if row["split"] == split)
    with (DATA / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"DJPEG: {len(rows)} файлов", flush=True)


def cache_park():
    with (DATA / "manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with ThreadPoolExecutor(max_workers=6) as workers:
        matrix = np.stack(list(workers.map(features, [r["path"] for r in rows])))
    np.save(DATA / "djpeg-features.npy", matrix)
    print("Признаки DJPEG:", matrix.shape, flush=True)


def encode(image, quality, sampling):
    output = io.BytesIO()
    image.save(output, "JPEG", quality=quality, subsampling=sampling)
    output.seek(0)
    return output


def ucid_photo(item):
    number, content = item
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB")
    w, h = image.size
    x, y = (w - 256) // 32 * 16, (h - 256) // 32 * 16
    patch = image.crop((x, y, x + 256, y + 256))
    rows, vectors = [], []
    for sampling in (0, 2):  # Pillow: 4:4:4 и 4:2:0.
        for q1 in QUALITY:
            rows.append((number, sampling, 0, 0, 0, q1))
            vectors.append(features(encode(patch, q1, sampling)))
            for offset in (0, 4):
                # Второй JPEG всегда содержит одни и те же 256 × 256 пикселей сцены.
                margin = 16 + offset
                left, top = x - margin, y - margin
                frame = image.crop((left, top, left + 288, top + 288))
                with Image.open(encode(frame, q1, sampling)) as stored:
                    decoded = stored.convert("RGB").crop((margin, margin, margin + 256, margin + 256))
                for q2 in QUALITY:
                    rows.append((number, sampling, offset, 1, q1, q2))
                    vectors.append(features(encode(decoded, q2, sampling)))
    return rows, vectors


def cache_ucid(archive):
    chosen = set(np.random.default_rng(2026).choice(np.arange(1, 1339), 300, replace=False))
    sources = {}
    with tarfile.open(archive, "r|gz", bufsize=2**20) as tar:
        for member in tar:
            if "/UCID-1338/" in member.name and member.name.endswith(".tif"):
                number = int(Path(member.name).stem)
                if number in chosen:
                    sources[number] = tar.extractfile(member).read()
    assert set(sources) == chosen
    rows, matrix = [], []
    with ThreadPoolExecutor(max_workers=6) as workers:
        for labels, vectors in workers.map(ucid_photo, sorted(sources.items())):
            rows.extend(labels)
            matrix.extend(vectors)
    with (DATA / "ucid.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("photo", "sampling", "offset", "label", "q1", "q2"))
        writer.writerows(rows)
    np.save(DATA / "ucid-features.npy", np.stack(matrix))
    print("UCID:", len(chosen), "фотографий,", len(rows), "JPEG", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("djpeg", "ucid"))
    args = parser.parse_args()
    if args.dataset == "djpeg":
        write_manifest(DATA / "jpeg_data.tar.gz")
        cache_park()
    else:
        cache_ucid(DATA / "ucid-source.tar.gz")
