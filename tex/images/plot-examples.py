"""Построить иллюстрации по данным DJPEG и UCID, подготовленным prepare_data.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

folder = Path(__file__).resolve().parent
data = folder.parents[1] / "data"
rows = pd.read_csv(data / "ucid.csv")
features = np.load(data / "ucid-features.npy", mmap_mode="r")
settings = (rows.sampling == 2) & (rows.offset == 0) & (rows.q2 == 95)
centers = np.arange(-30, 31)

fig, ax = plt.subplots(figsize=(6.3, 2.7), layout="constrained")
for q1, label, color, line in [(0, "Одно сжатие, Q = 95", "#263b45", "--"),
                               (55, "Два сжатия, 55 → 95", "#a34426", "-")]:
    values = features[settings & (rows.q1 == q1), :123]
    assert values.shape == (300, 123)
    ax.plot(centers, 100 * values.mean(axis=0)[61 + centers],
            color=color, linestyle=line, linewidth=1.6, label=label)
ax.set(xlabel="Округлённый коэффициент ДКП (0, 1)", ylabel="Доля блоков, %", xlim=(-30, 30), ylim=(0, 20))
ax.set_yticks([0, 5, 10, 15, 20])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=.25)
ax.legend(frameon=False, fontsize=9)
fig.savefig(folder / "ucid-histogram.pdf")

plt.close(fig)

# Воспроизводимая иллюстрация примера из раздела о командном интерфейсе.
pair = ("S_363219_68.jpg", "D_363219_68_1078.jpg")
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), layout="constrained")
luminance = []
for ax, name, title in zip(axes[:2], pair, ("Одно сжатие", "Два сжатия")):
    path = data / "djpeg" / name
    with Image.open(path) as image:
        ax.imshow(np.asarray(image.convert("RGB")), interpolation="nearest")
    with Image.open(path) as image:
        image.draft("L", image.size)
        luminance.append(np.array(image, dtype=np.int16))
    ax.set_title(title, fontsize=11)

difference = np.abs(luminance[0] - luminance[1])
im = axes[2].imshow(difference, cmap="gray", vmin=0, vmax=difference.max(), interpolation="nearest")
axes[2].set_title("Разность яркостей |ΔY|", fontsize=11)
fig.colorbar(im, ax=axes[2], ticks=[0, 2, 4, 7], shrink=.9, pad=.03)
for ax in axes:
    ax.set_axis_off()
fig.savefig(folder / "djpeg-pair.pdf", dpi=240)
