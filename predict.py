"""Tebak kelas dari foto, atau dari webcam (mis. Logitech).

  python scripts/predict.py foto1.jpg foto2.jpg
  python scripts/predict.py --webcam 0        # 0 = kamera pertama; coba 1 atau 2 untuk kamera eksternal
"""
import argparse
from pathlib import Path
import torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parents[1]
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def load(path):
    ck = torch.load(path, map_location="cpu")
    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, len(ck["kelas"]))
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()})
    return m.eval(), ck["kelas"]

def tebak(m, kelas, img):
    with torch.no_grad(): p = torch.softmax(m(tf(img.convert("RGB")).unsqueeze(0)), 1)[0]
    return kelas[int(p.argmax())], float(p.max())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fotos", nargs="*"); ap.add_argument("--webcam", type=int)
    ap.add_argument("--model", default=str(ROOT / "models" / "resnet18_terbaik_fp16.pt"))
    a = ap.parse_args(); m, kelas = load(a.model)
    for f in a.fotos:
        k, p = tebak(m, kelas, Image.open(f)); print(f"{f}: {k} ({p:.1%})")
    if a.webcam is not None:
        import cv2
        cap = cv2.VideoCapture(a.webcam); print("Tekan Q untuk keluar")
        while True:
            ok, fr = cap.read()
            if not ok: break
            k, p = tebak(m, kelas, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            cv2.putText(fr, f"{k} {p:.0%}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("predict", fr)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release(); cv2.destroyAllWindows()
