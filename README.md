# Computer-Vision-and-Deep-Learning-RET503
# Transfer Learning ResNet-18 untuk Klasifikasi Objek Basket (backboard, ball, rim, latar)

Tugas RET503 Computer Vision and Deep Learning: Transfer Learning dan Fine-Tuning.
Penulis: **(isi nama dan NIM)**. Proyek: penglihatan robot MicroDuck.

**Model:** ResNet-18 pretrained ImageNet (torchvision). **Mode yang dibandingkan:** `feature`, `partial`, dan `scratch`.

## Isi repositori

```
├── README.md
├── docs/catatan.md             # catatan keputusan dan masalah data
├── dataset_raw/                # dataset klasifikasi: <kelas>/*.jpg + metadata.csv
├── colab/jalankan_di_colab.ipynb
├── scripts/
│   ├── train.py                # ResNet-18 x 3 mode x 3 seed
│   ├── report.py               # tabel + grafik per epoch + confusion matrix + isi README
│   ├── predict.py              # tebak dari foto atau webcam
│   └── contoh_kelas.py         # gambar contoh tiap kelas
├── results/                    # history.csv, summary.json, tabel, grafik
├── models/                     # resnet18_terbaik_fp16.pt (dibuat oleh report.py)
└── requirements.txt
```

## Data

> **Sumber data:** dataset ini **tidak diambil sendiri oleh penulis**. Berkas yang sama (kelas, jumlah foto, `metadata.csv`)
> juga terdapat pada repositori https://github.com/AkbarL090/ret503-transfer-learning
> **(isi/ubah keterangan sumber ini sesuai kenyataan: dari siapa dataset diperoleh)**.

Menurut `dataset_raw/metadata.csv`, foto berasal dari kamera robot (e-con See3CAM_CU135 di kepala robot),
sesi BRAIL-24Sep, 24 September 2026, cahaya netral. Tiap foto adalah potongan (crop) satu objek dari frame video.
Pembagian train/valid/test mengikuti kolom `split` pada `metadata.csv`; frame sumber yang sama tidak terbagi ke split berbeda.

| Kelas | Train | Valid | Test | Total (setelah pembersihan) |
| --- | --- | --- | --- | --- |
| backboard | 109 | 20 | 25 | 154 |
| ball | 115 | 15 | 26 | 156 |
| latar | 124 | 23 | 26 | 173 |
| rim | 115 | 21 | 25 | 161 |

Setiap kelas melebihi syarat minimal 50 citra. **Pembersihan:** 3 foto pada folder `rim` dibuang dari training karena
salah label (`rim_20260924_netral_020` dan `_021` berisi bola, `_004` berisi papan putih); file aslinya tetap ada di `dataset_raw/`.
Untuk memakai data tanpa pembersihan: `python scripts/train.py --no-exclude`.

## Metode

Praproses: ubah ke 224 × 224, RGB, normalisasi mean/std ImageNet. Augmentasi train: RandomResizedCrop (skala 0,6-1,0),
HorizontalFlip, ColorJitter. Optimizer Adam, CosineAnnealingLR, 10 epoch, batch 32. BatchNorm pada layer beku dijaga di mode `eval()`.
Model terbaik per run dipilih dari akurasi validasi, lalu diuji sekali pada set test.

| Mode | Bobot awal | Yang dilatih | Learning rate |
| --- | --- | --- | --- |
| `feature` | ImageNet | layer `fc` baru saja | 1e-3 |
| `partial` | ImageNet | `layer4` + `fc` | 1e-4 / 1e-3 |
| `scratch` | acak | semua layer | 1e-3 |

Setiap mode dijalankan dengan 3 seed (0, 1, 2) karena set valid dan test kecil.

## Cara menjalankan

```
pip install -r requirements.txt
python scripts/contoh_kelas.py
python scripts/train.py --modes feature partial scratch --seeds 0 1 2
python scripts/report.py            # mengisi bagian Hasil di README ini
python scripts/predict.py foto.jpg  # atau: --webcam 0
```

Semua langkah juga ada di `colab/jalankan_di_colab.ipynb` (pilih GPU T4).

## Hasil

<!-- HASIL:START -->
*(Bagian ini terisi otomatis setelah `python scripts/report.py` dijalankan.)*
<!-- HASIL:END -->

## Analisis

*(Tulis analisismu sendiri setelah melihat hasil di atas. Pertanyaan pemandu:)*

1. Apakah bobot pretrained (`feature`, `partial`) belajar lebih cepat dan lebih akurat daripada `scratch`? Bukti apa dari tabel dan grafik?
2. Apakah fine-tuning `layer4` (`partial`) memberi keuntungan dibanding `feature`? Perhatikan simpangan baku dan jumlah citra test.
3. Kelas mana yang paling sering salah pada confusion matrix, dan kenapa kira-kira?
4. Mode mana yang dipilih, dan apa alasannya?
5. Mengapa angka yang sangat tinggi perlu dibaca hati-hati pada data dari satu sesi?

## Keterbatasan

- Seluruh data dari satu sesi, satu lokasi, satu kamera, dan cahaya netral; hasil belum membuktikan kinerja di kondisi lain.
- Set valid (sekitar 79 citra) dan test (sekitar 100 citra) kecil, sehingga selisih beberapa persen antar mode bisa hanya derau.
- Potongan objek kecil (rata-rata sekitar 46 × 64 px) diperbesar ke 224 × 224.
- Data berasal dari kamera e-con; kamera Logitech yang akan dipakai kemudian belum diuji dan bisa menghasilkan akurasi berbeda.
- Ini klasifikasi potongan objek, bukan deteksi posisi objek pada frame penuh.
