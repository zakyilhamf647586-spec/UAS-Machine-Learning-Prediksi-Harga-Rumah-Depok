# 🏠 Prediksi Harga Rumah di Depok — Machine Learning

**Proyek akademik · Python · Scikit-learn · Random Forest · Streamlit**

> **Catatan:** Aplikasi ini merupakan eksperimen pembelajaran *regresi harga properti*, **bukan** jasa penilaian rumah resmi. Dataset, cakupan data, dan galat model membatasi keandalan hasilnya.

## Ringkasan

Proyek ini memprediksi harga rumah dari jumlah kamar tidur, kamar mandi, garasi, luas tanah, luas bangunan, dan lokasi di Depok. Model dibangun dengan pipeline preprocessing yang **sama pada training dan inference**, lalu dibandingkan pada data uji yang tidak digunakan untuk training.

- **Model terpilih:** Random Forest Regression (150 pohon, `max_depth=20`, `min_samples_leaf=2`).
- **Hasil pada holdout:** **R² = 0,839**; **MAE = Rp406,82 juta**; RMSE sekitar Rp712,10 juta.
- **Pembanding Linear Regression:** R² = 0,765; MAE = Rp516,60 juta.
- **Split:** 80%/20% berbasis *group* dari kombinasi fitur yang identik (18.670 baris latih, 4.622 baris uji), `random_state=42`.

Angka di atas berlaku untuk **eksperimen baru dengan cakupan data yang ditetapkan di bawah**, bukan pengganti langsung angka dari notebook UAS sebelumnya.

## Dataset dan ruang lingkup

Dataset yang diberikan memiliki **24.561 baris**, tujuh kolom, **37 nilai lokasi unik**, tanpa nilai kosong dan tanpa baris yang sepenuhnya identik. Namun, nilai ekstrem perlu ditinjau secara kritis; dataset mencantumkan harga maksimum Rp995 miliar dan luas tanah maksimum 818.801.098 m².

| Variabel | Peran | Keterangan |
|---|---|---|
| Harga | Target | Rupiah (Rp) |
| Kamar Tidur | Fitur | Jumlah |
| Kamar Mandi | Fitur | Jumlah |
| Garasi | Fitur | Jumlah |
| Luas Tanah | Fitur | m² |
| Luas Bangunan | Fitur | m² |
| Lokasi | Fitur | Kategori lokasi |

Untuk mengurangi pengaruh nilai ekstrem dalam **eksperimen ini**, kami menerapkan batas cakupan tetap **sebelum splitting**, bukan cutoff IQR yang dipelajari dari data uji:

- Harga: Rp100 juta–Rp10 miliar.
- Luas tanah dan bangunan: 20–1.500 m².
- Kamar tidur dan mandi: 1–15; garasi: 0–10.

Dengan batas tersebut, **23.292 baris termasuk cakupan**, sedangkan **1.269 dikeluarkan**. Batas ini merupakan keputusan pemodelan yang membatasi populasi yang diuji—bukan fakta universal tentang harga rumah di Depok dan bukan jaminan bahwa semua data yang tersisa bersih.

**Sumber awal, periode pengumpulan, status harga (iklan/transaksi), dan hak redistribusi data belum terdokumentasi.** Verifikasi sebelum menerbitkan atau menggunakan data untuk tujuan lain.

## Metodologi

1. Membaca dataset dan memeriksa kolom, nilai kosong, duplikat, serta nilai ekstrem.
2. Menentukan batas eksplisit untuk populasi analisis, lalu memfilter data.
3. Membagi data menggunakan `GroupShuffleSplit(test_size=0.2, random_state=42)` sehingga kombinasi input yang sama tidak tersebar di kedua subset. Ada **3.739 baris** di dalam cakupan yang memiliki kombinasi fitur identik dengan baris lain; ini tidak sama dengan membuktikan duplikasi properti.
4. Membentuk pipeline `ColumnTransformer`: lima fitur numerik diteruskan; `Lokasi` menggunakan `OneHotEncoder(handle_unknown='ignore')`, **fit hanya pada data latih**.
5. Melatih Linear Regression dan Random Forest; membandingkan pada holdout yang sama.
6. Menyimpan pipeline final beserta `metrics.json`; aplikasi Streamlit memuat keduanya tanpa melakukan encoding lokasi secara terpisah.

## Evaluasi model

| Model | MAE pada data uji | RMSE pada data uji | R² data uji |
|---|---:|---:|---:|
| Baseline (median harga latih) | Rp1.175,83 juta | Rp1.889,75 juta | -0,131 |
| Linear Regression | Rp516,60 juta | Rp862,03 juta | 0,765 |
| **Random Forest (terpilih)** | **Rp406,82 juta** | **Rp712,10 juta** | **0,839** |

**MAE** adalah rata-rata besar selisih absolut antara harga aktual dan harga prediksi. Nilai Rp406,82 juta masih merupakan ketidakpastian yang besar untuk keputusan finansial.

### Error per rentang harga aktual — Random Forest

| Rentang harga aktual | Jumlah data uji | MAE |
|---|---:|---:|
| Hingga Rp1 miliar | 1.745 | Rp184,75 juta |
| Di atas Rp1 miliar hingga Rp3 miliar | 2.003 | Rp367,59 juta |
| Lebih dari Rp3 miliar | 874 | Rp940,09 juta |

**Perlu dicatat:** hasil di atas berasal dari **satu split**; split tersebut juga digunakan untuk membandingkan dua kandidat model, sehingga bukan test set independen setelah pemilihan model. Belum ada cross-validation, evaluasi temporal, atau validasi data eksternal. R² training sebesar 0,954 dibandingkan R² testing 0,839 mengindikasikan selisih kinerja yang perlu dianalisis sebelum menyatakan model siap dipakai secara luas.

## Jalankan proyek

**Lingkungan training yang diuji:** Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, Scikit-learn 1.8.0. Eksekusi UI Streamlit pada server publik belum diverifikasi dalam audit ini.

```bash
git clone https://github.com/zakyilhamf647586-spec/UAS-Machine-Learning-Prediksi-Harga-Rumah-Depok.git
cd UAS-Machine-Learning-Prediksi-Harga-Rumah-Depok
python -m venv .venv
```

Aktivasi: Windows PowerShell: `.venv\Scripts\Activate.ps1` atau Linux/macOS: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

`train_model.py` membutuhkan dataset pada `dataset/Cleaned_Final_Rumah.csv`. File model dan metrik terbaru (`models/house_price_pipeline.joblib`, `metrics.json`) dapat dihasilkan ulang dengan script tersebut.

> **Keamanan:** file `.joblib`/`.pkl` merupakan format pickle yang dapat menjalankan kode pada saat dimuat. Muat hanya artefak dari training Anda atau sumber terpercaya. Hindari artefak asing.

## Struktur proyek versi ini

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── metrics.json
├── models/
│   └── house_price_pipeline.joblib
├── dataset/
│   └── Cleaned_Final_Rumah.csv
├── notebook/
│   └── Validasi_Model_Depok.ipynb
├── tests/
│   └── test_model.py
├── AUDIT_VALIDASI.md
└── README.md
```

## Testing

```bash
pip install pytest
python -m pytest tests/test_model.py -q
```

Pengujian mencakup prediksi untuk setiap lokasi, pemisahan kombinasi fitur identik antara data latih/uji, serta konsistensi angka metrik dari model yang benar-benar disimpan.

## Perubahan dibanding notebook UAS sebelumnya

Notebook historis menggunakan beberapa tahap filtering dan *retraining*. Nilai R² **0,797** / MAE **Rp266,10 juta** adalah hasil RF pada subset **19.619 baris** di tengah notebook; eksperimen terakhir notebook menyaring dataset menjadi **20.821 baris** dan menyimpan Random Forest berbeda. Kedua hasil tidak boleh dicampur. Lihat `AUDIT_VALIDASI.md` untuk rekonstruksi eksperimentasinya.

## Keterbatasan dan tindak lanjut

- Kualitas dan sumber data (termasuk apakah harga merupakan penawaran atau harga transaksi) belum diverifikasi.
- Aturan batas data adalah keputusan lingkup eksperimen yang mungkin mengecualikan segmen pasar tertentu.
- Variabel penting seperti usia bangunan, kondisi, sertifikat, akses, serta koordinat lokasi belum tersedia.
- Pengujian baru memanfaatkan satu pembagian holdout; tambahkan *group cross-validation*, uji kestabilan waktu, pemeriksaan kesalahan menurut lokasi, dan pemeriksaan bias harga.
- Jangan memasarkan hasil sebagai penilaian properti resmi; perlu pengujian eksternal dan pembaruan data.

**Author:** Zaky Ilham Ferdiansyah — Informatics Engineering, Universitas Dian Nuswantoro. [LinkedIn](https://www.linkedin.com/in/zaky-ilham-ferdiansyah/)
