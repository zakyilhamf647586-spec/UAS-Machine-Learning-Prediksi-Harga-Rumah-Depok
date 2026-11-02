<div align="center">

# 🏠 Prediksi Harga Rumah di Depok

### Machine Learning · Random Forest Regression · Streamlit

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Project](https://img.shields.io/badge/Type-Academic%20Project-273746)

**Proyek akademik untuk memperkirakan harga rumah di Depok berdasarkan karakteristik properti, mulai dari eksplorasi data dan pemodelan hingga aplikasi prediksi berbasis web.**

[**🚀 Demo Aplikasi**](https://uas-machine-learning-prediksi-harga-rumah-depok-akyvtuz6m8nj5n.streamlit.app/) · [**💻 Kode Aplikasi**](app.py) · [**📓 Notebook**](notebook/) · [**📂 Dataset**](dataset/)

</div>

---

## Ringkasan Proyek

Proyek ini mempelajari penggunaan **machine learning untuk memprediksi harga rumah di Kota Depok**, menggunakan informasi seperti luas bangunan, luas tanah, jumlah kamar, garasi, dan lokasi. Dua pendekatan regresi dibandingkan: **Linear Regression** sebagai model pembanding dan **Random Forest Regression** sebagai model utama.

Berdasarkan hasil evaluasi yang dilaporkan dalam proyek, Random Forest memberikan **R² = 0,797** dan **MAE = Rp266.100.206**, lebih baik daripada Linear Regression pada kedua metrik tersebut. Hasil tersebut merupakan estimasi berdasarkan dataset yang digunakan, **bukan appraisal atau penilaian resmi suatu properti**.

### Tujuan

- Mengeksplorasi faktor-faktor yang berkaitan dengan harga rumah pada data yang digunakan.
- Mempersiapkan data untuk pemodelan regresi melalui cleaning dan preprocessing.
- Membandingkan hasil Linear Regression dengan Random Forest Regression.
- Menyajikan estimasi harga melalui aplikasi interaktif berbasis **Streamlit**.

## Dataset

Dokumentasi proyek mencatat data properti rumah di Depok dengan ringkasan berikut:

| Informasi | Nilai |
|---|---|
| Jumlah data awal | **24.561** baris |
| Jumlah data setelah cleaning | **19.619** baris |
| Target prediksi | **Harga rumah** |
| Tipe permasalahan | **Regresi** |

**Variabel input yang didokumentasikan:**

| Fitur | Penjelasan |
|---|---|
| Jumlah kamar tidur | Banyaknya kamar tidur |
| Jumlah kamar mandi | Banyaknya kamar mandi |
| Jumlah garasi | Informasi kapasitas/jumlah garasi dalam dataset |
| Luas tanah | Luas bidang tanah |
| Luas bangunan | Luas bangunan rumah |
| Lokasi | Informasi area/lokasi properti |

> **Catatan sumber data:** README awal belum mencantumkan tautan sumber atau ketentuan penggunaan dataset. Informasi tersebut perlu dilengkapi berdasarkan sumber data yang sebenarnya sebelum dataset digunakan kembali atau didistribusikan lebih jauh. Nama kolom teknis dan satuan luas dapat diperiksa pada file dataset/notebook.

## Metodologi

Alur kerja proyek sebagaimana didokumentasikan:

```text
Dataset awal
    ↓
Data cleaning dan pemeriksaan data
    ↓
Penanganan missing values, duplikasi, dan outlier
    ↓
Exploratory Data Analysis (EDA)
    ↓
Encoding variabel lokasi
    ↓
Train–test split
    ↓
Training: Linear Regression & Random Forest Regression
    ↓
Evaluasi model menggunakan MAE dan R²
    ↓
Aplikasi prediksi dengan Streamlit
```

Detail implementasi (termasuk proporsi train–test split, teknik penanganan outlier, dan parameter model) sebaiknya dirujuk dari [notebook eksperimen](notebook/), karena belum dijelaskan secara spesifik dalam README awal.

## Model dan Hasil Evaluasi

| Model | MAE (lebih rendah lebih baik) | R² (lebih tinggi lebih baik) |
|---|---:|---:|
| Linear Regression | Rp326.133.837 | 0,725 |
| **Random Forest Regression** | **Rp266.100.206** | **0,797** |

**Model yang dipilih:** Random Forest Regression, berdasarkan perbandingan kedua metrik di atas.

**Cara membaca metrik:**

- **MAE (Mean Absolute Error)** menyatakan rata-rata besar selisih absolut antara harga aktual dan prediksi. Pada hasil yang dilaporkan, nilainya sekitar **Rp266,1 juta** untuk Random Forest.
- **R²** mengukur seberapa baik model menjelaskan variasi nilai target dibandingkan prediksi baseline berbasis rata-rata. Nilai **0,797** menunjukkan kesesuaian yang lebih baik dibandingkan model pembanding pada evaluasi yang didokumentasikan, tetapi **tidak berarti setiap harga rumah diprediksi dengan akurasi 79,7%**.

> Angka di atas disalin dari hasil yang tercantum pada dokumentasi awal; README ini tidak melakukan pelatihan atau pengujian ulang model. Untuk menilai kemampuan generalisasi, perlu ditinjau bagaimana data dibagi, bagaimana preprocessing diterapkan, dan apakah ada potensi data leakage.

### Feature Importance

Urutan fitur yang dilaporkan dalam dokumentasi proyek:

1. **Luas bangunan**
2. **Luas tanah**
3. **Kamar mandi**
4. **Kamar tidur**
5. **Lokasi**
6. **Garasi**

*Feature importance* menjelaskan kontribusi relatif fitur terhadap prediksi model menurut metode yang digunakan. Peringkat ini **bukan bukti hubungan sebab-akibat** atau jaminan bahwa urutannya selalu sama pada dataset lain.

## Aplikasi Streamlit

Aplikasi didokumentasikan memiliki fitur:

- Form input karakteristik rumah.
- Prediksi atau estimasi harga rumah.
- Informasi model.
- Ringkasan statistik dataset.
- Visualisasi feature importance.
- Ringkasan hasil prediksi.

**[Buka demo aplikasi Streamlit](https://uas-machine-learning-prediksi-harga-rumah-depok-akyvtuz6m8nj5n.streamlit.app/)**

> Jika demo sedang tidak aktif atau memerlukan waktu untuk memulai, proyek juga dapat dicoba secara lokal mengikuti instruksi di bawah. Ketersediaan live demo tidak diuji ulang dalam penyusunan dokumen ini.

## Struktur Repositori

Struktur tingkat atas berikut disesuaikan dengan tampilan repositori yang telah ditunjukkan:

```text
UAS-Machine-Learning-Prediksi-Harga-Rumah-Depok/
├── dataset/          # Berkas dataset
├── models/           # Berkas model yang disimpan
├── notebook/         # Notebook analisis dan pelatihan
├── app.py            # Aplikasi Streamlit
├── requirements.txt  # Dependensi Python
└── README.md         # Dokumentasi proyek
```

## Menjalankan Aplikasi Secara Lokal

### 1. Clone repositori

```bash
git clone https://github.com/zakyilhamf647586-spec/UAS-Machine-Learning-Prediksi-Harga-Rumah-Depok.git
cd UAS-Machine-Learning-Prediksi-Harga-Rumah-Depok
```

### 2. Buat virtual environment (direkomendasikan)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instal dependensi

```bash
python -m pip install -r requirements.txt
```

### 4. Jalankan Streamlit

```bash
python -m streamlit run app.py
```

Aplikasi biasanya dibuka melalui alamat lokal yang ditampilkan Streamlit di terminal. Pastikan berkas model dan dataset yang dibutuhkan oleh `app.py` tersedia pada path yang sesuai. Perintah di atas belum diuji ulang pada lingkungan baru dalam audit berbasis README ini.

## Teknologi

- **Pemrosesan data:** Python, Pandas, NumPy
- **Machine learning:** scikit-learn, Linear Regression, Random Forest Regression
- **Visualisasi:** Matplotlib
- **Aplikasi web:** Streamlit
- **Pengembangan dan versi kode:** VS Code, Git, GitHub

## Keterbatasan dan Pengembangan Selanjutnya

- Prediksi merupakan **estimasi statistik**, bukan nilai appraisal resmi atau rekomendasi transaksi.
- Performa yang dilaporkan bergantung pada data, pembagian evaluasi, serta proses preprocessing; belum tentu berlaku sama untuk properti atau periode lain.
- Sumber, izin penggunaan, cakupan waktu, dan representativitas dataset perlu didokumentasikan lebih lengkap.
- Detail validasi, parameter model, dan pencegahan **data leakage** perlu dikonfirmasi dari notebook sebelum klaim reproducibility yang lebih kuat dibuat.
- Pengembangan berikutnya dapat mencakup pengujian pada data baru, analisis error per rentang harga, dan penjelasan faktor prediksi yang lebih mendalam.

## Penulis

**Zaky Ilham Ferdiansyah**  
Program Studi Teknik Informatika · Universitas Dian Nuswantoro  
2026

[GitHub](https://github.com/zakyilhamf647586-spec) · [LinkedIn](https://www.linkedin.com/in/zaky-ilham-ferdiansyah/)
