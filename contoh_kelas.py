"""Buat gambar contoh 8 foto per kelas -> results/contoh_kelas.png"""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "dataset_raw" / "metadata.csv")
kelas = sorted(df.kelas.unique())
fig, ax = plt.subplots(len(kelas), 8, figsize=(14, 2 * len(kelas)), squeeze=False)
for i, k in enumerate(kelas):
    fl = df[(df.kelas == k) & (df.split == "train")].nama_file.tolist()[::12][:8]
    for j in range(8):
        ax[i][j].axis("off")
        if j < len(fl):
            ax[i][j].imshow(Image.open(ROOT / "dataset_raw" / fl[j]).convert("RGB"))
            if j == 0: ax[i][j].set_title(k, loc="left")
plt.tight_layout(); plt.savefig(ROOT / "results" / "contoh_kelas.png", dpi=130)
print("Tersimpan: results/contoh_kelas.png")
