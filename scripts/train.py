"""Transfer learning ResNet-18 pada dataset_raw (ball, rim, backboard, latar).

Tiga mode dibandingkan:
  feature : bobot ImageNet, hanya layer fc baru yang dilatih
  partial : bobot ImageNet, layer4 + fc dilatih (fine-tuning)
  scratch : bobot acak, semua layer dilatih (pembanding)

Contoh:
  python scripts/train.py --modes feature partial scratch --seeds 0 1 2
"""
import argparse, json, random, time, copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset_raw"
RES = ROOT / "results"
MDL = ROOT / "models"

# Foto yang salah folder (dilihat manual): isinya bola / papan putih, bukan ring.
EXCLUDE = [
    "rim/rim_20260924_netral_004.jpg",
    "rim/rim_20260924_netral_020.jpg",
    "rim/rim_20260924_netral_021.jpg",
]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        img, y = self.items[i]
        return self.tf(img), y


def load_items(kelas, exclude):
    df = pd.read_csv(RAW / "metadata.csv")
    df = df[df.kelas.isin(kelas)]
    if exclude:
        df = df[~df.nama_file.isin(EXCLUDE)]
    idx = {k: i for i, k in enumerate(kelas)}
    data = {}
    for split in ["train", "valid", "test"]:
        d = df[df.split == split]
        data[split] = [(Image.open(RAW / r.nama_file).convert("RGB"), idx[r.kelas])
                       for r in d.itertuples()]
    return data


def build(mode, n_cls):
    m = models.resnet18(weights=None if mode == "scratch" else models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    if mode == "feature":
        for p in m.parameters(): p.requires_grad = False
        for p in m.fc.parameters(): p.requires_grad = True
        groups = [{"params": m.fc.parameters(), "lr": 1e-3}]
    elif mode == "partial":
        for p in m.parameters(): p.requires_grad = False
        for p in list(m.layer4.parameters()) + list(m.fc.parameters()): p.requires_grad = True
        groups = [{"params": m.layer4.parameters(), "lr": 1e-4},
                  {"params": m.fc.parameters(), "lr": 1e-3}]
    else:
        groups = [{"params": m.parameters(), "lr": 1e-3}]
    return m, groups


def set_train(m):
    m.train()
    for l in m.modules():   # BatchNorm pada layer beku tetap eval()
        if isinstance(l, nn.BatchNorm2d) and not any(p.requires_grad for p in l.parameters()):
            l.eval()


@torch.no_grad()
def evaluate(m, dl, dev, loss_fn):
    m.eval(); tot = ok = n = 0; P, Y = [], []
    for x, y in dl:
        x, y = x.to(dev), y.to(dev); o = m(x)
        tot += loss_fn(o, y).item() * len(y); p = o.argmax(1)
        ok += (p == y).sum().item(); n += len(y); P += p.cpu().tolist(); Y += y.cpu().tolist()
    return tot / n, ok / n, P, Y


def run(mode, seed, a, data, kelas, dev):
    out = RES / f"resnet18_{mode}_s{seed}"
    if (out / "summary.json").exists() and not a.force:
        print(f"[lewati] {out.name} sudah ada"); return
    out.mkdir(parents=True, exist_ok=True); MDL.mkdir(exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_ev = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    g = torch.Generator().manual_seed(seed)
    dl = {"train": DataLoader(DS(data["train"], tf_tr), a.batch, shuffle=True, generator=g),
          "valid": DataLoader(DS(data["valid"], tf_ev), a.batch),
          "test":  DataLoader(DS(data["test"], tf_ev), a.batch)}

    model, groups = build(mode, len(kelas)); model.to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(groups); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss()

    hist, best_acc, best_loss, best_ep, best_state = [], -1, 9e9, 0, None
    t0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        set_train(model); tl = ta = n = 0
        for x, y in dl["train"]:
            x, y = x.to(dev), y.to(dev)
            o = model(x); loss = loss_fn(o, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(y); ta += (o.argmax(1) == y).sum().item(); n += len(y)
        sch.step()
        vl, va, _, _ = evaluate(model, dl["valid"], dev, loss_fn)
        hist.append(dict(epoch=ep, train_loss=tl/n, train_acc=ta/n, val_loss=vl, val_acc=va))
        print(f"{mode:8s} s{seed} ep{ep:2d} | train {ta/n:.3f} | val {va:.3f} (loss {vl:.3f})")
        if va > best_acc or (va == best_acc and vl < best_loss):
            best_acc, best_loss, best_ep = va, vl, ep; best_state = copy.deepcopy(model.state_dict())
    t_train = time.perf_counter() - t0

    model.load_state_dict(best_state)
    _, te, P, Y = evaluate(model, dl["test"], dev, loss_fn)
    cm = np.zeros((len(kelas), len(kelas)), int)
    for y, p in zip(Y, P): cm[y, p] += 1
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    pd.DataFrame(cm, index=kelas, columns=kelas).to_csv(out / "confusion_test.csv")
    va = [h["val_acc"] for h in hist]
    first = next((i + 1 for i, v in enumerate(va) if v >= .9), None)
    stable = next((e for e in range(1, len(va) + 1) if all(v >= .9 for v in va[e - 1:])), None)
    json.dump(dict(mode=mode, seed=seed, epochs=a.epochs, best_val_acc=best_acc, best_epoch=best_ep,
                   test_acc=te, train_time_s=t_train, n_trainable=n_tr, n_params=n_all,
                   epoch_val90_first=first, epoch_val90_stable=stable,
                   n_train=len(data["train"]), n_valid=len(data["valid"]), n_test=len(data["test"]),
                   kelas=kelas, device=torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"),
              open(out / "summary.json", "w"), indent=2)
    torch.save(best_state, MDL / f"resnet18_{mode}_s{seed}.pt")
    print(f"==> {out.name}: val {best_acc:.3f} | test {te:.3f} | {t_train:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["feature", "partial", "scratch"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--kelas", nargs="+", default=["backboard", "ball", "latar", "rim"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-exclude", action="store_true", help="jangan buang 3 foto salah label")
    ap.add_argument("--force", action="store_true", help="ulangi run yang sudah ada")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kelas = sorted(a.kelas)
    RES.mkdir(exist_ok=True); json.dump(kelas, open(RES / "kelas.json", "w"))
    data = load_items(kelas, not a.no_exclude)
    print({k: len(v) for k, v in data.items()}, "| kelas:", kelas, "| perangkat:", dev)
    for mode in a.modes:
        for seed in a.seeds:
            run(mode, seed, a, data, kelas, dev)
