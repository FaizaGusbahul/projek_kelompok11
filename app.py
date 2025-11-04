import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt

# ============================================================
# 1. SETUP PAGE
# ============================================================
st.set_page_config(page_title="Analisis Akses Air Bersih", layout="wide", page_icon="💧")
st.title("💧 Analisis dan Prediksi Akses Air Bersih di Indonesia")

st.markdown("""
Aplikasi ini dibuat untuk mendukung **SDG 6: Air Bersih dan Sanitasi Layak**.  
Kamu dapat mengunggah dataset baru atau menggunakan dataset bawaan untuk:
- 📊 Melihat tren dan analisis akses air bersih.  
- 🤖 Melakukan prediksi menggunakan model machine learning (Random Forest).  
- 🧩 Mengetahui faktor yang paling berpengaruh terhadap akses air bersih.
""")

# ============================================================
# 2. PILIH DATASET
# ============================================================
st.sidebar.header("📂 Pilih Dataset")
uploaded_file = st.sidebar.file_uploader("Upload dataset CSV kamu", type=["csv"])
default_path = "TUGAS_DATA_KELOMPOK11/DATA_WATER_SUPPLY_STATISTICS.csv"

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset berhasil diunggah!")
else:
    if os.path.exists(default_path):
        df_raw = pd.read_csv(default_path)
        st.sidebar.info("ℹ️ Menggunakan dataset default dari folder proyek.")
    else:
        st.error("❌ Tidak ada dataset ditemukan. Upload file CSV terlebih dahulu.")
        st.stop()

# ============================================================
# 3. PEMBERSIHAN DATA OTOMATIS
# ============================================================
st.header("🧹 Pembersihan Data Otomatis")
st.write("Sistem akan membersihkan data secara otomatis (hapus duplikat, isi nilai kosong, normalisasi kolom).")

df = df_raw.copy()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.drop_duplicates(inplace=True)
df.dropna(axis=1, how='all', inplace=True)

num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

st.dataframe(df.head())

# ============================================================
# 4. PILIH MENU ANALISIS
# ============================================================
menu = st.sidebar.radio("Pilih Menu", ["📈 Tren Nasional", "🔮 Prediksi Provinsi", "📊 Analisis Faktor"])

# ============================================================
# 5. TREND NASIONAL
# ============================================================
if menu == "📈 Tren Nasional":
    st.header("📈 Tren Nasional Akses Air Bersih")
    if "tahun" in df.columns and "air_bersih" in df.columns:
        df_year = df.groupby("tahun")["air_bersih"].mean().reset_index()
        st.line_chart(df_year, x="tahun", y="air_bersih", use_container_width=True)
        st.markdown("Grafik menunjukkan rata-rata jumlah air bersih nasional per tahun.")
    else:
        st.warning("Kolom 'tahun' dan 'air_bersih' tidak ditemukan dalam dataset.")

# ============================================================
# 6. PREDIKSI PROVINSI
# ============================================================
elif menu == "🔮 Prediksi Provinsi":
    st.header("🔮 Prediksi Akses Air Bersih per Provinsi")

    model_path = "model.pkl"
    scaler_path = "scaler.pkl"
    features_path = "features.json"

    if not os.path.exists(model_path):
        st.error("❌ File `model.pkl` tidak ditemukan. Pastikan file model sudah ada di folder proyek.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    with open(features_path, "r") as f:
        features = json.load(f)

    st.markdown("Masukkan data indikator sesuai kebutuhan:")
    inputs = {}
    for feat in features:
        inputs[feat] = st.number_input(f"{feat}", value=0.0, step=0.01)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([inputs])
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df
        pred = model.predict(input_scaled)[0]
        st.success(f"💧 Prediksi jumlah air bersih: **{pred:.2f}**")

# ============================================================
# 7. ANALISIS FAKTOR
# ============================================================
elif menu == "📊 Analisis Faktor":
    st.header("📊 Analisis Faktor yang Mempengaruhi Akses Air Bersih")

    model_path = "model.pkl"
    features_path = "features.json"
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        with open(features_path, "r") as f:
            features = json.load(f)

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Fitur": features,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance["Fitur"], importance["Importance"])
            ax.set_xlabel("Tingkat Pengaruh")
            ax.set_ylabel("Fitur")
            ax.set_title("Faktor Paling Berpengaruh terhadap Akses Air Bersih")
            st.pyplot(fig)
            st.dataframe(importance)
        else:
            st.warning("Model tidak mendukung atribut `feature_importances_`.")
    else:
        st.error("❌ File model atau fitur tidak ditemukan.")

# ============================================================
# 8. PENJELASAN AKHIR
# ============================================================
st.markdown("---")
st.markdown("""
**Kesimpulan:**
Aplikasi ini membantu pemerintah dan masyarakat untuk:
- Menganalisis tren ketersediaan air bersih di berbagai wilayah.  
- Memprediksi dampak kebijakan terhadap akses air bersih.  
- Menentukan faktor penting dalam pengelolaan sumber daya air.

🧩 Semua langkah mendukung pencapaian **SDG 6 - Air Bersih dan Sanitasi Layak.**
""")
