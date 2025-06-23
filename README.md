# Sistem Prediksi Capstone - FlowlyHub

## Deskripsi Singkat

Sistem Prediksi Capstone - aplikasi web berbasis Streamlit yang mengimplementasikan model machine learning untuk memprediksi keterlambatan karyawan dan status stok bahan. Prediksi keterlambatan mempertimbangkan jadwal, waktu kedatangan, hari, dan kondisi cuaca dengan toleransi otomatis. Prediksi stok menganalisis stok awal, barang masuk/keluar, dan periode untuk menentukan status (Aman, Stabil, atau Berisiko).

## Fitur Utama

- 📊 Dashboard interaktif dengan navigasi sidebar
- 🕒 Prediksi keterlambatan karyawan dengan toleransi berdasarkan cuaca
- 📦 Prediksi status stok bahan dengan tingkat risiko

## Model Machine Learning

### Model Absensi

- Input: jadwal masuk, waktu kedatangan, hari, kondisi cuaca
- Output: status keterlambatan (TEPAT WAKTU/TERLAMBAT)
- Toleransi: 5 menit untuk cuaca hujan/badai, 1 menit untuk cerah/berawan

### Model Stok

- Input: stok awal, barang masuk, barang keluar, bulan, satuan
- Output: status stok (Aman/Stabil/Berisiko)
- Estimasi: perhitungan hari hingga stok habis

## Teknologi

- **Frontend**: Streamlit
- **Model**: TensorFlow, Scikit-learn
- **Data Processing**: Pandas, NumPy, Joblib

## Cara Menjalankan

1. Install dependensi:
   ```
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```

## Struktur Folder

```
.
├── absensi/
│   ├── model/
│   │   ├── absensi_model.h5
│   │   ├── scaler.joblib
│   │   └── model_metadata.json
│   └── bismillah_[CAPSTONE]_absensi.ipynb
├── stok/
│   ├── model/
│   │   ├── stok_model.h5
│   │   ├── scaler.joblib
│   │   └── model_metadata.json
│   └── bismillah_[CAPSTONE]_stok.ipynb
├── app.py
└── requirements.txt
```

## Tim Machine Learning FlowlyHub

- **Dewi Safira Permata Sari** - MC009D5X0787
- **Erisa Putri Nabila** - MC229D5X0818
- **Firda Humaira** - MC009D5X0441

## Capstone Project - DBS Academy x Dicoding - 2025
