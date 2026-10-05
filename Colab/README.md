# Membedakan Bola dan Ring Basket dengan ResNet-18 (Transfer Learning)

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zahrannabilaputri-jpg/Computer-Vision-and-Deep-Learning-RET503/blob/main/notebook.ipynb)

**Mata kuliah:** RET503 Computer Vision and Deep Learning, topik Transfer Learning dan Fine-Tuning
**Penulis:** *(isi nama dan NIM)*
**Konteks:** dasar penglihatan untuk robot MicroDuck

## Ringkasan

Saya melatih ResNet-18 (bobot awal ImageNet) untuk membedakan dua kelas potongan gambar dari kamera robot basket:
**`ball`** (bola) dan **`rim`** (ring). Pelatihan dua tahap: lebih dulu hanya layer terakhir, lalu blok `layer4` ikut disesuaikan.

**Hasil:** akurasi test **100% (51 dari 51 foto benar)**. Angka ini perlu dibaca hati-hati karena set test kecil dan datanya dari
satu sesi; alasannya dijelaskan di bagian [Pembahasan](#pembahasan).

## Dataset

> **Sumber data:** dataset ini **bukan hasil pengambilan sendiri**. Berkas yang sama juga ada di
> https://github.com/AkbarL090/ret503-transfer-learning.
> **(Ubah catatan ini sesuai kenyataan: dari siapa dataset saya peroleh.)**
> Foto dari kamera Logitech milik saya belum dipakai untuk pelatihan; lihat [Rencana lanjutan](#rencana-lanjutan).

Menurut `dataset_raw/metadata.csv`, foto berasal dari kamera e-con See3CAM_CU135 (kepala robot), sesi BRAIL-24Sep,
24 September 2026, cahaya netral. Tiap foto adalah potongan satu objek dari frame video. Zip asli memuat empat kelas
(`ball`, `rim`, `backboard`, `latar`); untuk tugas ini saya memakai dua kelas saja. Pembagian train, valid, dan test
mengikuti kolom `split` pada `metadata.csv`, dan frame sumber yang sama tidak terbagi ke split berbeda.

| Kelas | Train | Valid | Test | Jumlah |
| --- | --- | --- | --- | --- |
| ball | 115 | 15 | 26 | 156 |
| rim | 115 | 21 | 25 | 161 |
| **Total** | 230 | 36 | 51 | 317 |

Kedua kelas melebihi syarat minimal 50 citra. Sebelum training saya membuang tiga foto di folder `rim` yang salah label
(`rim_20260924_netral_020` dan `_021` berisi bola, `_004` berisi papan putih); file aslinya tetap ada di `dataset_raw/`.

![Contoh foto tiap kelas](results/contoh_data.png)

## Metode

| Komponen | Pengaturan |
| --- | --- |
| Model | ResNet-18 pretrained ImageNet, layer `fc` diganti menjadi 2 keluaran |
| Masukan | diubah ke 224 × 224, RGB, normalisasi mean/std ImageNet |
| Augmentasi (train saja) | HorizontalFlip, rotasi 10°, ColorJitter (0,3) |
| Tahap A (epoch 1–10) | hanya `fc` dilatih, Adam, learning rate 1e-3 |
| Tahap B (epoch 11–20) | `layer4` dan `fc` dilatih, Adam, learning rate 1e-4 |
| Lain-lain | batch 16, loss CrossEntropy, BatchNorm layer beku dijaga di mode `eval()`, seed 42 |
| Pemilihan model | epoch dengan akurasi validasi tertinggi (seri: loss validasi lebih rendah); set test dipakai sekali di akhir |

Seluruh kode dan keluaran lengkap ada di [`notebook.ipynb`](notebook.ipynb).

## Hasil

### Proses belajar

![Kurva loss dan akurasi](results/training_curves.png)

Garis putus-putus menandai pergantian dari tahap A ke tahap B. Data per epoch ada di [`results/history.csv`](results/history.csv).

| Epoch (global) | Fase | Akurasi train | Akurasi valid | Loss valid |
| --- | --- | --- | --- | --- |
| 1 | A | 88,7% | 80,6% | 0,333 |
| 2 | A | 98,7% | 97,2% | 0,105 |
| 3 | A | 99,6% | 100% | 0,079 |
| 10 | A | 100% | 100% | 0,019 |
| 11 | B | 100% | 100% | 0,000 |
| 20 | B | 100% | 100% | 0,000 |

Loss valid "0,000" berarti nilainya di bawah 0,0005 (dibulatkan tiga desimal oleh notebook).

### Ujian akhir (set test, 51 foto)

| Kelas | Precision | Recall | F1 | Jumlah |
| --- | --- | --- | --- | --- |
| ball | 1,000 | 1,000 | 1,000 | 26 |
| rim | 1,000 | 1,000 | 1,000 | 25 |
| **Akurasi** | | | **1,000** | 51 |

![Confusion matrix test](results/confusion_matrix.png)

Tidak ada foto test yang salah ditebak. Laporan lengkap: [`results/classification_report.txt`](results/classification_report.txt).

## Pembahasan

1. **Transfer learning membuat model cepat belajar.** Pada tahap A hanya layer terakhir yang dilatih, tetapi akurasi validasi sudah
   80,6% pada epoch 1, 97,2% pada epoch 2, dan 100% pada epoch 3. Akurasi train juga sudah 98,7% pada epoch 2. Artinya fitur umum
   hasil ImageNet sudah cukup untuk memisahkan bola dan ring tanpa melatih backbone.
2. **Manfaat fine-tuning (tahap B) tidak dapat diukur di sini.** Akurasi validasi sudah 100% sebelum tahap B, sehingga tidak ada
   ruang untuk naik. Yang berubah hanya loss validasi, dari 0,019 menjadi hampir nol. Dengan 36 foto validasi, penurunan itu tidak
   cukup untuk menyimpulkan bahwa fine-tuning memberi keuntungan; ini efek langit-langit (*ceiling effect*), bukan bukti.
3. **Tidak ada tanda *overfitting* pada kurva.** Kurva train dan valid berdekatan di semua epoch, tanpa celah yang melebar.
   Namun karena akurasi keduanya sudah mentok 100%, kurva ini tidak bisa membedakan model yang benar-benar umum dari model yang
   sekadar cocok dengan data yang mudah.
4. **Akurasi test 100% (51/51) tidak berarti model sempurna.** Dengan 51 foto, hasil tanpa kesalahan hanya menunjukkan bahwa laju
   kesalahan sebenarnya kemungkinan kecil. Perkiraan kasar (aturan tiga) menaruh batas atas sekitar 3/51 ≈ 6% pada tingkat
   kepercayaan 95%. Karena tidak ada kesalahan, saya juga tidak dapat menganalisis jenis kesalahan model.
5. **Tugasnya memang mudah.** Bola berwarna kuning terang dan ring berwarna merah dengan bentuk silinder, jadi kedua kelas berbeda
   jelas dalam warna dan bentuk. Ditambah semua foto berasal dari satu sesi, satu kamera, dan satu cahaya, sehingga latar dan posisi
   robot pada test kemungkinan mirip dengan train. Hasil ini kemungkinan lebih optimis daripada kinerja di lingkungan baru.
6. **Kualitas label memengaruhi hasil.** Saya menemukan tiga foto salah folder pada kelas `rim` (isinya bola dan papan putih).
   Foto seperti ini membingungkan model dan, bila jatuh di set test, akan tampak sebagai kesalahan model padahal kesalahannya di data.
7. **Satu kali percobaan.** Seluruh hasil berasal dari satu seed (42) tanpa pengulangan, jadi variasi antar-percobaan belum diukur.

## Keterbatasan

- Semua data dari satu sesi, satu lokasi, satu kamera (e-con), dan cahaya netral; kinerja di kondisi lain belum terbukti.
- Set valid (36 foto) dan test (51 foto) kecil, dan akurasi sudah menyentuh 100%, sehingga perbedaan antar pengaturan tidak dapat dibedakan.
- Hanya dua kelas yang mudah dibedakan; kelas `backboard` dan `latar` pada dataset belum dipakai.
- Potongan objek kecil (rata-rata sekitar 46 × 64 piksel) diperbesar ke 224 × 224.
- Ini klasifikasi potongan objek, bukan deteksi posisi objek di frame penuh.
- Satu seed saja, tanpa perbandingan dengan pelatihan dari nol atau model lain.
- Label salah selain tiga foto di atas mungkin masih ada; pemeriksaan dilakukan manual dan tidak menyeluruh.

## Rencana lanjutan

- Mengambil foto dengan kamera Logitech pada cahaya dan lokasi berbeda sebagai test set tambahan untuk menguji ketahanan model.
- Mencoba keempat kelas (`ball`, `rim`, `backboard`, `latar`) dan beberapa seed, termasuk pembanding dari nol.
- Menambahkan tahap deteksi atau ROI agar model dapat dipakai pada frame penuh di robot MicroDuck.

## Isi repositori

```
├── README.md
├── notebook.ipynb            # kode dan keluaran lengkap dari Google Colab (GPU T4)
├── dataset_raw/              # <kelas>/*.jpg dan metadata.csv
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── contoh_data.png
│   ├── history.csv
│   ├── classification_report.txt
│   └── metrics.json
└── requirements.txt
```

## Cara menjalankan ulang

1. Klik tombol **Buka di Colab** di atas, lalu pilih **Runtime → Change runtime type → T4 GPU**.
2. Unduh folder `dataset_raw` dalam bentuk zip (`dataset_raw.zip`), lalu upload pada Tahap 2 di notebook.
3. Jalankan sel dari atas ke bawah. Berkas bobot model tidak disertakan di repositori ini.
