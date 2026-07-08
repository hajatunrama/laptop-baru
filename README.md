# 💻 Laptop Price Predictor

Proyek Machine Learning untuk memprediksi harga laptop berdasarkan spesifikasi
(CPU, GPU, RAM, storage, dll), lengkap dengan dashboard interaktif berbasis
**Streamlit**.

Tiga algoritma dilatih dan dibandingkan secara otomatis:
- **Linear Regression**
- **Random Forest Regressor**
- **XGBoost Regressor**

Model dengan skor R² tertinggi dipilih otomatis sebagai model produksi.

## 📁 Struktur Folder

```
laptop-price-ml/
├── app/
│   └── streamlit_app.py     # Dashboard Streamlit (EDA + prediksi)
├── data/
│   └── dataset_laptop_realistis.csv
├── models/
│   ├── best_model.pkl        # Model terbaik (hasil training)
│   ├── metrics.json          # Perbandingan metrik semua model
│   └── feature_meta.json     # Metadata fitur untuk form input
├── train_model.py            # Script training & evaluasi model
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan Secara Lokal

1. Clone repo ini:
   ```bash
   git clone https://github.com/<username>/<repo-name>.git
   cd laptop-price-ml
   ```

2. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Opsional) Latih ulang model:
   ```bash
   python train_model.py
   ```

5. Jalankan dashboard Streamlit:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🌐 Deploy ke Streamlit Community Cloud (gratis)

1. Push project ini ke repo GitHub kamu (lihat langkah di bawah).
2. Buka https://share.streamlit.io/ lalu login dengan akun GitHub.
3. Klik **"New app"**, pilih repo `laptop-price-ml`.
4. Isi:
   - **Main file path**: `app/streamlit_app.py`
   - **Branch**: `main`
5. Klik **Deploy**. Streamlit akan otomatis install dari `requirements.txt`.

## 🔗 Menghubungkan ke GitHub

Dari folder project ini, jalankan:

```bash
git init
git add .
git commit -m "Initial commit: laptop price prediction ML + Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

Ganti `<username>` dan `<repo-name>` sesuai akun & nama repo GitHub kamu.

## 📊 Fitur Dashboard

- **Ringkasan Model**: perbandingan R², MAE, RMSE antar algoritma.
- **Eksplorasi Data**: distribusi harga, boxplot per tipe penggunaan,
  scatter performa vs harga, heatmap korelasi.
- **Prediksi Harga**: form interaktif untuk input spesifikasi laptop dan
  mendapatkan estimasi harga secara real-time.

## 🧠 Dataset

Dataset berisi 1.200 baris data laptop dengan fitur: Country, Brand, Model,
CPU/GPU spec, RAM, Storage, skor performa, dan harga (USD).
