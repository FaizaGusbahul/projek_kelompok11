import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

# ============================================================
# 1. LOAD MODEL DAN FILE PENDUKUNG
# ============================================================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_features():
    with open("features.json", "r") as f:
        features = json.load(f)
    return features

model = load_model()
scaler = load_scaler()
features = load_features()

# ============================================================
# 2. CONFIGURASI PAGE
# ============================================================
st.set_page_config(page_title="Prediksi Akses Air Bersih (SDG 6)", layout="wide")
st.title("💧 Prediksi Akses Air Bersih di Indonesia")
st.markdown("""
Aplikasi ini dibuat untuk mendukung *Tujuan Pembangunan Berkelanjutan (SDG 6)*: 
Menjamin ketersediaan dan pengelolaan air bersih yang berkelanjutan untuk semua.

Fungsi aplikasi:
- Menganalisis tren nasional air bersih
- Melakukan prediksi berdasarkan indikator tiap provinsi
- Melihat faktor paling berpengaruh terhadap akses air bersih
""")

# ============================================================
# 3. PILIH MENU
# ============================================================
menu = st.sidebar.radio("Navigasi", ["📈 Tren Nasional", "🔮 Prediksi Provinsi", "📊 Analisis Faktor"])

# ============================================================
# 4. UPLOAD DATASET
# ============================================================
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload dataset mentah (.csv)", type=["csv"])
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset berhasil diunggah")
else:
    st.sidebar.warning("Gunakan contoh dataset bawaan")
    df_raw = pd.read_csv("DATA_WATER_SUPPLY_STATISTICS.csv")

# ============================================================
# 5. CLEANING OTOMATIS
# ============================================================
def clean_dataset(df):
    df.columns = [c.strip().replace(" ", "").replace("-", "").lower() for c in df.columns]
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    # Pastikan kolom tahun bertipe numerik
    if "tahun" in df.columns:
        df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce")
    # Ganti nilai kosong dengan median
    df = df.fillna(df.median(numeric_only=True))
    return df

df = clean_dataset(df_raw)

# ============================================================
# 6. MENU: TREN NASIONAL
# ============================================================
if menu == "📈 Tren Nasional":
    st.header("📊 Tren Nasional Air Bersih")
    st.markdown("Menampilkan perkembangan rata-rata air bersih di seluruh provinsi dan prediksi 5 tahun ke depan.")

    # Pilih kolom target
    target_col = st.selectbox("Pilih kolom utama (misal jumlah_air_bersih)", df.columns)
    df_year = df.groupby("tahun")[target_col].mean().dropna()

    if not df_year.empty:
        df_year.index = pd.to_datetime(df_year.index, format="%Y")

        # Forecast dengan Holt-Winters
        model_hw = ExponentialSmoothing(df_year, trend='add', seasonal=None, damped_trend=True)
        fit = model_hw.fit(optimized=True)
        forecast = fit.forecast(steps=5)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_year.index, y=df_year.values, mode='lines+markers', name='Aktual', line=dict(color='steelblue')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines+markers', name='Prediksi', line=dict(color='orange', dash='dash')))
        fig.update_layout(title=f"Tren dan Prediksi Nasional: {target_col}", xaxis_title="Tahun", yaxis_title="Nilai Rata-rata")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Hasil Prediksi 5 Tahun ke Depan")
        st.dataframe(forecast.round(2).rename("Prediksi"))
    else:
        st.warning("Kolom target tidak memiliki data yang cukup.")

# ============================================================
# 7. MENU: PREDIKSI PROVINSI
# ============================================================
elif menu == "🔮 Prediksi Provinsi":
    st.header("🔮 Prediksi Jumlah Air Bersih per Provinsi")
    st.markdown("Masukkan indikator untuk memperkirakan jumlah air bersih yang dihasilkan provinsi.")

    # Input manual berdasar fitur model
    user_input = {}
    for feat in features:
        user_input[feat] = st.number_input(f"{feat}", value=0.0, step=0.1)

    if st.button("Prediksi Jumlah Air Bersih"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"💧 Prediksi Jumlah Air Bersih: *{prediction:.2f} unit*")

# ============================================================
# 8. MENU: ANALISIS FAKTOR
# ============================================================
elif menu == "📊 Analisis Faktor":
    st.header("📊 Faktor yang Mempengaruhi Akses Air Bersih")
    st.markdown("Visualisasi fitur yang paling berpengaruh dalam model Random Forest.")

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Fitur": features,
            "Pentingnya": model.feature_importances_
        }).sort_values(by="Pentingnya", ascending=False)

        st.bar_chart(importance.set_index("Fitur"))
        st.dataframe(importance)
    else:
        st.warning("Model tidak memiliki atribut feature_importances_.")

# ============================================================
# 9. FOOTER
# ============================================================
st.markdown("---")
st.caption("Aplikasi ini dikembangkan untuk mendukung *SDG 6: Air Bersih dan Sanitasi Layak* — oleh Tim Analisis Data Air Bersih Indonesia 💧")
