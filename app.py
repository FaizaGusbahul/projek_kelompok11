# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import difflib

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
default_path = "DATA_WATER_SUPPLY_STATISTICS.csv"

# try reading CSV with a couple fallback options (basic)
def read_csv_with_fallback(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        try:
            # coba separator semicolon
            return pd.read_csv(path_or_buffer, sep=';')
        except Exception:
            # terakhir, coba engine python dan infer
            return pd.read_csv(path_or_buffer, engine='python')

if uploaded_file:
    df_raw = read_csv_with_fallback(uploaded_file)
    st.sidebar.success("✅ Dataset berhasil diunggah!")
else:
    if os.path.exists(default_path):
        df_raw = read_csv_with_fallback(default_path)
        st.sidebar.info("ℹ️ Menggunakan dataset default dari folder proyek.")
    else:
        st.error("❌ Tidak ada dataset ditemukan. Upload file CSV terlebih dahulu.")
        st.stop()

# ============================================================
# 3. PEMBERSIHAN DATA OTOMATIS (naming normalization + basic impute)
# ============================================================
st.header("🧹 Pembersihan Data Otomatis")
st.write("Sistem akan membersihkan data secara otomatis (hapus duplikat, isi nilai kosong, normalisasi kolom).")

# Copy original, keep original column names for mapping convenience
df = df_raw.copy()

# Keep an original copy of column names
original_columns = df.columns.tolist()

# Normalize columns for internal operations but keep mapping to original labels
normalized_map = {col: col.strip().lower().replace(" ", "_") for col in original_columns}
df.rename(columns=normalized_map, inplace=True)

# Remove duplicate rows and drop-fully-empty columns
df.drop_duplicates(inplace=True)
df.dropna(axis=1, how='all', inplace=True)

# Fill numeric columns' missing with median
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Show diagnostic info to user (very important to see what columns exist)
st.subheader("🔎 Informasi Kolom (diagnostik)")
st.write("Nama kolom setelah normalisasi:", df.columns.tolist())
st.write("Contoh 5 baris pertama:")
st.dataframe(df.head())

# ============================================================
# helpers: auto-detect columns (fuzzy) and cleaning numeric strings
# ============================================================
def find_best_column(col_names, candidates, cutoff=0.6):
    """
    col_names: list of normalized column names (lowercase underscore)
    candidates: list of candidate keys in normalized form
    returns the chosen normalized column name or None
    """
    col_lower = [c.lower() for c in col_names]
    # exact candidate match
    for cand in candidates:
        if cand in col_lower:
            return col_names[col_lower.index(cand)]
    # fuzzy match
    for cand in candidates:
        matches = difflib.get_close_matches(cand, col_lower, n=1, cutoff=cutoff)
        if matches:
            return col_names[col_lower.index(matches[0])]
    return None

def clean_numeric_series(ser):
    """Bersihkan string numeric: hapus % dan koma ribuan lalu ubah ke numeric"""
    s = ser.astype(str).str.strip()
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    # Hapus karakter non-digit kecuali minus dan titik
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# ============================================================
# 4. PILIH MENU ANALISIS + manuel mapping
# ============================================================
menu = st.sidebar.radio("Pilih Menu", ["📈 Tren Nasional", "🔮 Prediksi Provinsi", "📊 Analisis Faktor"])

# Sidebar manual mapping (berguna jika auto-detect salah)
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Pemetaan Kolom (opsional)")
manual_year = st.sidebar.selectbox("Pilih kolom Tahun (optional)", options=[None] + df.columns.tolist())
manual_water = st.sidebar.selectbox("Pilih kolom 'Air Bersih' (optional)", options=[None] + df.columns.tolist())

# ============================================================
# 5. TREND NASIONAL
# ============================================================
if menu == "📈 Tren Nasional":
    st.header("📈 Tren Nasional Akses Air Bersih")

    # kandidat nama kolom (dalam bentuk normalized): tambahkan kemungkinan lain jika dataset mu berbeda
    year_candidates = ["tahun", "year", "tahun_berdasarkan", "tahun_akses"]
    water_candidates = [
        "air_bersih", "airbersih", "access_to_clean_water", "clean_water", "persentase_air_bersih",
        "percent_access", "presentase_air_bersih", "akses_air_bersih", "persen_air_bersih"
    ]

    # gunakan manual jika user memilih
    year_col = manual_year if manual_year else find_best_column(df.columns.tolist(), year_candidates)
    water_col = manual_water if manual_water else find_best_column(df.columns.tolist(), water_candidates)

    st.write("Deteksi kolom -> tahun:", year_col, ", air_bersih:", water_col)

    if year_col and water_col:
        # bersihkan kolom tahun: ambil digit (misal '2020' dari string)
        try:
            df[year_col] = pd.to_numeric(df[year_col].astype(str).str.replace(r"\D+", "", regex=True), errors="coerce")
        except Exception:
            # fallback: biarkan apa adanya
            pass

        # bersihkan kolom air bersih jadi numeric
        df[water_col] = clean_numeric_series(df[water_col])

        df_year = df.groupby(year_col)[water_col].mean().reset_index().dropna()
        if df_year.empty:
            st.warning("Data tahun / air_bersih ada tetapi hasil agregasi kosong (cek nilai kosong atau format).")
        else:
            # Sort by year (if numeric)
            try:
                df_year = df_year.sort_values(by=year_col)
            except Exception:
                pass
            st.line_chart(df_year, x=year_col, y=water_col, use_container_width=True)
            st.markdown("Grafik menunjukkan rata-rata jumlah air bersih nasional per tahun.")
    else:
        st.warning("Kolom 'tahun' dan/or 'air_bersih' tidak ditemukan otomatis. Periksa daftar kolom di panel diagnostik atau pilih kolom secara manual di sidebar.")

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
    else:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # load features if exists; jika tidak, coba ambil numeric columns sebagai fallback
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                features = json.load(f)
        else:
            features = df.select_dtypes(include='number').columns.tolist()
            st.info("features.json tidak ditemukan — menggunakan kolom numerik dari dataset sebagai fitur fallback.")

        st.markdown("Masukkan data indikator sesuai kebutuhan:")
        inputs = {}
        for feat in features:
            # default 0.0, jika kolom ada di df, isi nilai median sebagai default
            default_val = 0.0
            if feat in df.columns:
                try:
                    default_val = float(pd.to_numeric(df[feat], errors="coerce").median(skipna=True))
                except Exception:
                    default_val = 0.0
            inputs[feat] = st.number_input(f"{feat}", value=float(default_val), step=0.01)

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
            }).sort_values(by="Importance", ascending=True)  # ascending so barh displays largest on top

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance["Fitur"], importance["Importance"])
            ax.set_xlabel("Tingkat Pengaruh")
            ax.set_ylabel("Fitur")
            ax.set_title("Faktor Paling Berpengaruh terhadap Akses Air Bersih")
            st.pyplot(fig)
            st.dataframe(importance.sort_values(by="Importance", ascending=False).reset_index(drop=True))
        else:
            st.warning("Model tidak mendukung atribut `feature_importances_`.")
    else:
        st.error("❌ File model atau fitur tidak ditemukan. Pastikan `model.pkl` dan `features.json` ada jika ingin analisis faktor.")

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
