# Laporan Audit & Validasi — Machine Learning Harga Rumah Depok

## 1. Berkas yang diperiksa

`Cleaned_Final_Rumah.csv`, `UAS_Prediksi_Harga_Rumah.ipynb`, `app (1).py`, `requirements(20261102-100242).txt`, README awal, dan metadata/model pickle yang dilampirkan. Kode lama diperiksa tanpa mendeserialisasi pickle yang diunggah. Pipeline **baru** dibuat dan diuji dari training script yang diperiksa.

## 2. Fakta dataset (hasil inspeksi langsung)

- 24.561 baris × 7 kolom (`Harga`, `Kamar Tidur`, `Kamar Mandi`, `Garasi`, `Luas Tanah`, `Luas Bangunan`, `Lokasi`).
- Tidak ada missing value atau baris duplikat penuh; terdapat 37 kategori lokasi.
- Ada 3.948 baris yang memiliki kombinasi fitur input identik dengan baris lain. Kombinasi fitur sama bisa mewakili properti berbeda; ini bukan bukti duplikasi record.
- Nilai ekstrem: `Harga` Rp995 miliar, `Luas Tanah` dan `Luas Bangunan` hingga 818.801.098 m², `Kamar Tidur` hingga 284, `Kamar Mandi` hingga 180.
- Sumber/perizinan dataset tidak tersedia di file yang diberikan. Penamaan `Cleaned_Final` tidak membuktikan kualitas dataset final.

## 3. Rekonstruksi dua eksperimen lama

### Eksperimen bertahap (cell 39 dan 55–69)

- Filter `Luas Tanah <5000`, `Luas Bangunan <5000`: 24.539 baris.
- IQR `Harga`: 22.351 baris.
- IQR `Luas Tanah`: 20.415 baris.
- IQR `Luas Bangunan`: 19.619 baris.
- Output notebook tersimpan: Linear Regression MAE Rp326.133.837,07 dan R² 0,7250656; RF (100 pohon) MAE Rp266.100.206,97 dan R² 0,7974713. **Ini output historis dari eksperimen bertahap, bukan hasil model yang terakhir disimpan**.

### Eksperimen akhir (cell 78–80)

- Notebook memuat ulang CSV dan menerapkan batas atas IQR `Harga`, `Luas Tanah` dan `Luas Bangunan` secara bersamaan: **20.821 baris**.
- Konfigurasi `rf_final`: 50 pohon, `max_depth=20`, `random_state=42`; terakhir disimpan sebagai `random_forest_model.pkl`.
- **Rekonstruksi kode di lingkungan ini** dengan filter akhir dan split yang sama: 16.656 baris latih, 4.165 baris uji; RF 50 pohon memiliki **MAE Rp293.617.579,12; R² 0,799710**. RF 100 pohon pada subset sama memiliki MAE Rp291.440.206,94 dan R² 0,799697.
- Rekonstruksi menguji konfigurasi yang sama, **bukan** memvalidasi isi persis pickle lama atau deployment yang berjalan. Versi library juga dapat mengubah hasil numerik.
- Model lama memakai `pd.get_dummies` pada satu baris input dengan `drop_first=True`, lalu `reindex`. Semua indikator lokasi menjadi nol; bug encoding ini dapat menghasilkan prediksi tak sensitif terhadap pilihan lokasi.

## 4. Desain eksperimen baru (dengan batas dan hasil yang transparan)

Dalam paket baru, data difilter dengan batas tetap: `Harga` Rp100 juta–Rp10 miliar, kedua luas 20–1.500 m², kamar tidur/mandi 1–15, garasi 0–10. **Batas ini bukan aturan pasar objektif**; sengaja mendefinisikan ruang lingkup terbatas untuk studi akademik.

- Dalam cakupan: 23.292; dikeluarkan: 1.269.
- GroupShuffleSplit 80/20 random_state=42 berdasarkan kombinasi seluruh fitur input, 18.670 train dan 4.622 test. Tidak ada *exact input-feature groups* bersama kedua subset.
- `ColumnTransformer` + `OneHotEncoder(handle_unknown='ignore')` di-fit hanya pada train. Penanganan lokasi identik saat training dan inference.
- Pembanding: median harga latih, Linear Regression, RF 150 trees / depth 20 / min_samples_leaf 2 / random_state 42.
- Model terpilih berdasar MAE holdout: Random Forest; **dipakai model hasil fit train saja** (tidak melakukan fit ulang pada test), disimpan bersama metadata metrik dari split yang sama.

| Model | MAE holdout | RMSE holdout | R² holdout |
|---|---:|---:|---:|
| Baseline median | Rp1.175.833.405,45 | Rp1.889.753.573,27 | -0,130886 |
| Linear Regression | Rp516.601.471,83 | Rp862.026.217,78 | 0,764685 |
| Random Forest (baru) | **Rp406.816.836,52** | **Rp712.101.584,62** | **0,839420** |

Train RF: R² 0,954023 dan MAE Rp228.111.933,53. Selisih dengan holdout mengindikasikan kebutuhan memeriksa overfitting.

| Harga aktual di test | n | MAE |
|---|---:|---:|
| ≤ Rp1 miliar | 1.745 | Rp184.753.433,25 |
| > Rp1 miliar s.d. Rp3 miliar | 2.003 | Rp367.585.576,20 |
| > Rp3 miliar | 874 | Rp940.090.123,83 |

Catatan: `pd.cut` menempatkan tepat Rp1 miliar dalam kelompok pertama dan tepat Rp3 miliar dalam kelompok kedua. Model juga dipilih berdasarkan holdout yang dipakai melaporkan performa; ini bukan evaluasi terakhir pada test set independen setelah pemilihan model. Pernyataan model lebih unggul **berlaku untuk holdout dan cakupan tersebut**, bukan penilaian semua properti di Depok.

## 5. Pemeriksaan kode yang benar-benar dilakukan

- Model baru dilatih dan disimpan di `models/house_price_pipeline.joblib`.
- Model tersimpan dibuka kembali dan diuji pada contoh lokasi Cinere, Beji, Sawangan: seluruhnya menghasilkan angka valid; unit test juga menguji **semua 37 lokasi**.
- Tiga pengujian pytest lulus: input lokasi, tidak ada persilangan exact feature groups, dan metrik model persisted cocok dengan nilai laporan.
- UI Streamlit baru telah dicek sintaks Python, **belum diuji lewat server Streamlit** karena paket Streamlit tidak tersedia di runtime audit. Testing aplikasi di lingkungan deployment masih diperlukan.
- Belum dilakukan group cross-validation multi-fold atau verifikasi sumber data; jangan memakai hasil sebagai jaminan akurasi produksi.

## 6. Rencana implementasi ke GitHub

1. Backup repositori/branch lama, jangan menimpa bukti eksperimen UAS awal.
2. Buat branch baru misalnya `improve/validated-pipeline` dan salin `train_model.py`, `app.py`, `requirements.txt`, `metrics.json`, `models/house_price_pipeline.joblib`, notebook, tests, `AUDIT_VALIDASI.md`, `README.md`.
3. Pastikan CSV ada di `dataset/Cleaned_Final_Rumah.csv`. Verifikasi izin publikasi dataset sebelum commit.
4. Install requirements dalam lingkungan yang sesuai dan jalankan `python train_model.py` (opsional untuk reproduksi) serta `python -m pytest tests/test_model.py -q` setelah memasang pytest.
5. Jalankan `streamlit run app.py` dan uji semua input serta tampilan, terutama lokasi dan perbedaan estimasi. Kemudian deploy dari branch yang sudah diverifikasi.
6. Setelah audit stabil, merge ke main; baru ubah angka metrik pada LinkedIn/README bila merujuk **eksperimen baru** dan jelaskan perubahan cakupan evaluasi.
