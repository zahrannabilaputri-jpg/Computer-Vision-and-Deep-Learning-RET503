"""Rangkum hasil semua run: tabel, grafik, confusion matrix, isi README, ekspor model terbaik."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, MDL, README = ROOT / "results", ROOT / "models", ROOT / "README.md"
MODES = ["feature", "partial", "scratch"]
START, END = "<!-- HASIL:START -->", "<!-- HASIL:END -->"

def c(x): return str(x).replace(".", ",")
def pm(v): return c(f"{np.mean(v)*100:.1f} ± {np.std(v)*100:.1f}%")
def num(n): return f"{n:,}".replace(",", ".")

runs = []
for d in sorted(RES.glob("resnet18_*_s*")):
    if (d / "summary.json").exists():
        s = json.load(open(d / "summary.json")); s["dir"] = d; runs.append(s)
assert runs, "Belum ada hasil di results/. Jalankan scripts/train.py dulu."
kelas = runs[0]["kelas"]; ep = runs[0]["epochs"]
seeds = sorted({r["seed"] for r in runs})

rows, per_seed, agg = [], [], {}
for m in MODES:
    rs = sorted([r for r in runs if r["mode"] == m], key=lambda r: r["seed"])
    if not rs: continue
    f1 = [r["epoch_val90_first"] for r in rs if r["epoch_val90_first"]]
    st = [r["epoch_val90_stable"] for r in rs if r["epoch_val90_stable"]]
    ep90 = f"{c(f'{np.mean(f1):.1f}')} / {c(f'{np.mean(st):.1f}')}" if f1 and st else "tidak tercapai"
    ntr, nall = rs[0]["n_trainable"], rs[0]["n_params"]
    agg[m] = dict(val=np.mean([r["best_val_acc"] for r in rs]), ntr=ntr, rs=rs)
    rows.append(f"| `{m}` | {num(ntr)} ({c(f'{ntr/nall*100:.2f}' if ntr/nall < 0.01 else f'{ntr/nall*100:.1f}')}%) | {pm([r['best_val_acc'] for r in rs])} | "
                f"{pm([r['test_acc'] for r in rs])} | {ep90} | {np.mean([r['train_time_s'] for r in rs]):.0f} s |")
    per_seed.append(f"`{m}` " + " / ".join(c(f"{r['test_acc']*100:.1f}") for r in rs) + "%")

# mode terpilih: akurasi VALIDASI tertinggi (bukan test); seri -> parameter dilatih lebih sedikit
best_mode = sorted(agg, key=lambda m: (-round(agg[m]["val"], 6), agg[m]["ntr"]))[0]

# grafik per epoch
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m in agg:
    H = pd.concat([pd.read_csv(r["dir"] / "history.csv") for r in agg[m]["rs"]]).groupby("epoch").mean()
    ax[0].plot(H.index, H.val_acc * 100, marker="o", ms=3, label=m)
    ax[1].plot(H.index, H.val_loss, marker="o", ms=3, label=m)
ax[0].set_title("Akurasi validasi (rata-rata seed)"); ax[0].set_ylabel("%")
ax[1].set_title("Loss validasi (rata-rata seed)")
for a in ax: a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(RES / "akurasi_per_epoch_resnet18.png", dpi=150); plt.close()

# confusion matrix test, dijumlah seluruh seed pada mode terpilih
cm = sum(pd.read_csv(r["dir"] / "confusion_test.csv", index_col=0).values for r in agg[best_mode]["rs"])
fig, a = plt.subplots(figsize=(5, 4.5)); a.imshow(cm, cmap="Blues")
a.set_xticks(range(len(kelas))); a.set_xticklabels(kelas, rotation=30)
a.set_yticks(range(len(kelas))); a.set_yticklabels(kelas)
a.set_xlabel("Tebakan model"); a.set_ylabel("Kelas sebenarnya")
a.set_title(f"Confusion matrix test ({best_mode}, {len(agg[best_mode]['rs'])} seed dijumlah)")
for i in range(len(kelas)):
    for j in range(len(kelas)):
        a.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
               color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig(RES / "confusion_resnet18.png", dpi=150); plt.close()
recall = {k: cm[i, i] / cm[i].sum() for i, k in enumerate(kelas)}

r0 = runs[0]
satu = c(f"{100 / r0['n_test']:.1f}")
blok = "\n".join([
    f"ResNet-18, {ep} epoch, seed {', '.join(map(str, seeds))} per mode; rata-rata ± simpangan baku. "
    f"Set valid {r0['n_valid']} citra, test {r0['n_test']} citra, perangkat: {r0['device']}.", "",
    "| Mode | Param dilatih | Akurasi val terbaik | Akurasi test | Epoch val ≥ 90% (pertama / stabil) | Waktu latih |",
    "| --- | --- | --- | --- | --- | --- |", *rows, "",
    "Akurasi test per seed: " + "; ".join(per_seed) + f". Satu citra salah di test setara {satu}%.", "",
    "![Akurasi dan loss validasi per epoch](results/akurasi_per_epoch_resnet18.png)", "",
    "![Confusion matrix test](results/confusion_resnet18.png)", "",
    f"Mode terpilih berdasarkan akurasi validasi: **`{best_mode}`**. Recall per kelas pada test (mode ini): "
    + ", ".join(f"{k} {c(f'{v*100:.1f}')}%" for k, v in recall.items()) + ".", "",
    "Data mentah tiap run ada di `results/resnet18_<mode>_s<seed>/` (`history.csv`, `summary.json`, `confusion_test.csv`).",
])
(RES / "tabel_ringkasan.md").write_text(blok + "\n")
json.dump(dict(mode_terpilih=best_mode, recall=recall,
               ringkasan={m: dict(val=agg[m]["val"], test=float(np.mean([r["test_acc"] for r in agg[m]["rs"]])))
                          for m in agg}), open(RES / "summary.json", "w"), indent=2)

txt = README.read_text()
if START in txt and END in txt:
    txt = txt[:txt.find(START) + len(START)] + "\n" + blok + "\n" + txt[txt.find(END):]
else:
    txt += f"\n## Hasil\n{START}\n{blok}\n{END}\n"
README.write_text(txt)

# ekspor model terbaik (fp16 ~22 MB, muat di batas upload browser GitHub 25 MB)
try:
    import torch
    br = max(agg[best_mode]["rs"], key=lambda r: r["best_val_acc"])
    sd = torch.load(MDL / f"resnet18_{best_mode}_s{br['seed']}.pt", map_location="cpu")
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(dict(state_dict=sd, kelas=kelas, mode=best_mode, seed=br["seed"]), MDL / "resnet18_terbaik_fp16.pt")
    print("Model terbaik disimpan: models/resnet18_terbaik_fp16.pt")
except Exception as e:
    print("Ekspor model dilewati:", e)
print("Selesai. Mode terpilih:", best_mode)
