# Transfer Learning ResNet-18 untuk Klasifikasi Objek Basket

**Kelas:** `backboard`, `ball`, `rim`, `latar` · **Mata kuliah:** RET503 Computer Vision and Deep Learning · **Proyek:** penglihatan robot MicroDuck

| | |
| --- | --- |
| **Penulis** | Zahra Nabila Putri (4222401028) |
| **Tugas** | Transfer Learning dan Fine-Tuning |
| **Model** | ResNet-18 pretrained ImageNet (`torchvision`) |
| **Mode yang dibandingkan** | `feature`, `partial`, `scratch` |
| **Framework** | PyTorch + torchvision |

---

## Daftar isi

1. [Ringkasan](#ringkasan)
2. [Struktur repositori](#struktur-repositori)
3. [Data](#data)
4. [Metode](#metode)
5. [Cara menjalankan](#cara-menjalankan)
6. [Keluaran yang dihasilkan](#keluaran-yang-dihasilkan)
7. [Hasil](#hasil)
8. [Analisis](#analisis)
9. [Keterbatasan](#keterbatasan)
10. [Pemecahan masalah](#pemecahan-masalah)
11. [Sumber dan kredit](#sumber-dan-kredit)

---

## Ringkasan

Proyek ini mengklasifikasikan potongan (crop) citra dari kamera robot menjadi empat kelas: **backboard**, **ball**, **rim**, dan **latar** (bukan ketiganya). Tujuannya membandingkan tiga strategi pelatihan pada ResNet-18:

1. **`feature`**: bobot ImageNet dibekukan, hanya layer `fc` baru yang dilatih (feature extraction).
2. **`partial`**: `layer4` dan `fc` dilatih, layer lain dibekukan (fine-tuning sebagian).
3. **`scratch`**: bobot acak, semua layer dilatih (tanpa transfer learning, sebagai pembanding).

Setiap mode dijalankan dengan 3 seed (0, 1, 2) karena set valid dan test kecil, sehingga hasil dilaporkan sebagai rata-rata ± simpangan baku, bukan satu angka tunggal.

---

## Struktur repositori

```
Computer-Vision-and-Deep-Learning-RET503/
├── README.md
├── requirements.txt
├── docs/
│   └── catatan.md                  # catatan keputusan dan masalah data
├── dataset_raw/                    # dataset klasifikasi
│   ├── backboard/*.jpg
│   ├── ball/*.jpg
│   ├── latar/*.jpg
│   ├── rim/*.jpg
│   └── metadata.csv                # kolom split: train / valid / test
├── colab/
│   └── jalankan_di_colab.ipynb     # seluruh langkah di Google Colab (GPU T4)
├── scripts/
│   ├── train.py                    # ResNet-18 x 3 mode x 3 seed
│   ├── report.py                   # tabel, grafik per epoch, confusion matrix, isi bagian Hasil
│   ├── predict.py                  # prediksi dari foto atau webcam
│   └── contoh_kelas.py             # gambar contoh tiap kelas
├── results/                        # history.csv, summary.json, tabel, grafik
└── models/
    └── resnet18_terbaik_fp16.pt    # dibuat oleh report.py
```

> `results/` dan `models/` terisi setelah `train.py` dan `report.py` dijalankan.

---

## Data

> **Sumber data:** dataset ini **tidak diambil sendiri oleh penulis**. Berkas yang sama (kelas, jumlah foto, `metadata.csv`)
> juga terdapat pada repositori <https://github.com/AkbarL090/ret503-transfer-learning>.
> **TODO: ganti keterangan ini dengan sumber sebenarnya (dari siapa dataset diperoleh) sebelum dikumpulkan.**

### Asal foto

Menurut `dataset_raw/metadata.csv`, foto berasal dari kamera robot (e-con See3CAM_CU135 di kepala robot), sesi **BRAIL-24Sep**, 24 September 2026, cahaya netral. Tiap foto adalah **potongan satu objek** dari frame video, bukan frame penuh.

### Pembagian data

Pembagian train/valid/test mengikuti kolom `split` pada `metadata.csv`. Frame sumber yang sama **tidak** terbagi ke split berbeda, sehingga tidak ada kebocoran data (data leakage) antar split.

| Kelas | Train | Valid | Test | Total (setelah pembersihan) |
| --- | ---: | ---: | ---: | ---: |
| backboard | 109 | 20 | 25 | 154 |
| ball | 115 | 15 | 26 | 156 |
| latar | 124 | 23 | 26 | 173 |
| rim | 115 | 21 | 25 | 161 |

Setiap kelas melebihi syarat minimal 50 citra.

### Pembersihan data

3 foto pada folder `rim` dibuang dari training karena salah label:

| Berkas | Isi sebenarnya |
| --- | --- |
| `rim_20260924_netral_020` | bola |
| `rim_20260924_netral_021` | bola |
| `rim_20260924_netral_004` | papan putih |

File aslinya tetap ada di `dataset_raw/`. Untuk memakai data **tanpa** pembersihan:

```bash
python scripts/train.py --no-exclude
```

Catatan keputusan dan masalah data lainnya ada di `docs/catatan.md`.

---

## Metode

### Praproses

- Ubah ukuran ke **224 × 224**, format RGB
- Normalisasi mean/std ImageNet

### Augmentasi (hanya train)

- `RandomResizedCrop` (skala 0,6–1,0)
- `HorizontalFlip`
- `ColorJitter`

### Pelatihan

| Pengaturan | Nilai |
| --- | --- |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR |
| Epoch | 10 |
| Batch size | 32 |
| Seed | 0, 1, 2 |
| BatchNorm pada layer beku | dijaga di mode `eval()` |

Model terbaik per run dipilih dari **akurasi validasi**, lalu diuji **sekali** pada set test.

### Tiga mode

| Mode | Bobot awal | Yang dilatih | Learning rate |
| --- | --- | --- | --- |
| `feature` | ImageNet | layer `fc` baru saja | 1e-3 |
| `partial` | ImageNet | `layer4` + `fc` | 1e-4 (layer4) / 1e-3 (fc) |
| `scratch` | acak | semua layer | 1e-3 |

---

## Cara menjalankan

### Prasyarat

- Python 3.9 atau lebih baru
- GPU disarankan (CPU tetap bisa, lebih lambat)

### 1. Persiapan

```bash
git clone https://github.com/zahrannabilaputri-jpg/Computer-Vision-and-Deep-Learning-RET503.git
cd Computer-Vision-and-Deep-Learning-RET503
pip install -r requirements.txt
```

### 2. Lihat contoh tiap kelas

```bash
python scripts/contoh_kelas.py
```

### 3. Latih model

```bash
python scripts/train.py --modes feature partial scratch --seeds 0 1 2
```

Ini menjalankan 3 mode × 3 seed = 9 run dan menyimpan `history.csv` serta `summary.json` ke `results/`.

### 4. Buat laporan

```bash
python scripts/report.py
```

Skrip ini membuat tabel, grafik per epoch, confusion matrix, menyimpan model terbaik ke `models/`, dan **mengisi bagian [Hasil](#hasil)** di README ini.

### 5. Prediksi

```bash
python scripts/predict.py foto.jpg      # dari berkas foto
python scripts/predict.py --webcam 0    # dari webcam
```

### Alternatif: Google Colab

Semua langkah di atas tersedia di `colab/jalankan_di_colab.ipynb`. Pilih runtime **GPU T4** (Runtime → Change runtime type).

---

## Keluaran yang dihasilkan

| Berkas / folder | Isi |
| --- | --- |
| `results/history.csv` | loss dan akurasi per epoch untuk tiap mode dan seed |
| `results/summary.json` | ringkasan akurasi akhir per run |
| `results/` (tabel, grafik) | tabel perbandingan, kurva per epoch, confusion matrix |
| `models/resnet18_terbaik_fp16.pt` | bobot model terbaik (presisi fp16) |

---

## Hasil

<!-- HASIL:START -->
*(Bagian ini terisi otomatis setelah `python scripts/report.py` dijalankan.)*
<!-- HASIL:END -->

---

## Analisis

> **TODO:** tulis analisis sendiri setelah hasil di atas terisi. Jawab dengan bukti dari tabel dan grafik, bukan dugaan. Pertanyaan panduan:

### 1. Pretrained vs scratch
Apakah `feature` dan `partial` belajar lebih cepat dan lebih akurat daripada `scratch`? Bukti apa dari tabel dan grafik?

*(jawaban)*

### 2. Fine-tuning `layer4`
Apakah `partial` memberi keuntungan dibanding `feature`? Perhatikan simpangan baku dan jumlah citra test (sekitar 100).

*(jawaban)*

### 3. Kesalahan pada confusion matrix
Kelas mana yang paling sering salah, dan kenapa kira-kira?

*(jawaban)*

### 4. Mode terpilih
Mode mana yang dipilih, dan apa alasannya?

*(jawaban)*

### 5. Membaca akurasi tinggi dengan hati-hati
Mengapa angka yang sangat tinggi perlu dibaca hati-hati pada data dari satu sesi?

*(jawaban)*

---

## Keterbatasan

- Seluruh data dari **satu sesi, satu lokasi, satu kamera, dan cahaya netral**; hasil belum membuktikan kinerja di kondisi lain.
- Set valid (sekitar 79 citra) dan test (sekitar 100 citra) kecil, sehingga selisih beberapa persen antar mode bisa hanya derau.
- Potongan objek kecil (rata-rata sekitar 46 × 64 px) diperbesar ke 224 × 224.
- Data berasal dari kamera e-con; kamera Logitech yang akan dipakai kemudian belum diuji dan bisa menghasilkan akurasi berbeda.
- Ini **klasifikasi potongan objek**, bukan deteksi posisi objek pada frame penuh.

---

## Pemecahan masalah

| Masalah | Penyebab umum | Solusi |
| --- | --- | --- |
| `ModuleNotFoundError` | dependensi belum terpasang | `pip install -r requirements.txt` |
| Training sangat lambat | berjalan di CPU | pakai GPU atau Colab T4 |
| `CUDA out of memory` | VRAM tidak cukup | jalankan di Colab T4 atau kurangi batch size di `train.py` |
| Bagian Hasil masih kosong | `report.py` belum dijalankan | jalankan `train.py` dulu, lalu `report.py` |
| Webcam tidak terbuka | indeks kamera salah | coba `--webcam 1` |
| Push ke GitHub ditolak | berkas lebih dari 100 MB | periksa ukuran `dataset_raw/` dan `models/`, kecilkan atau pakai Git LFS |

---

## Sumber dan kredit

- Model: ResNet-18 pretrained ImageNet dari `torchvision`
- Data: lihat bagian [Data](#data) (isi sumber sebenarnya)
- Mata kuliah: RET503 Computer Vision and Deep Learning
