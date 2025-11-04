import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

# Load file dari Colab
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource
def load_features():
    with open('features.json', 'r') as f:
        features = json.load(f)
    return features

@st.cache_data
def load_data():
    df = pd.read_csv('DATA_WATER_SUPPLY_STATISTICS_clean.csv')
    return df

# Load semua
model = load_model()
scaler = load_scaler()
features = load_features()
df = load_data()

# Judul App
st.title("🌊 Prediksi Akses Air Bersih di Indonesia")
st.markdown("""
Aplikasi ini menggunakan machine learning untuk memprediksi dan menganalisis akses air bersih, mendukung **SDG 6 (Air Bersih dan Sanitasi Layak)**.
- **Tujuan**: Membantu pemerintah dan masyarakat memantau progres akses air minum yang aman, mengidentifikasi wilayah berisiko, dan merencanakan intervensi seperti peningkatan kapasitas produksi atau tenaga kerja.
- **Data**: Berdasarkan dataset statistik air bersih dari berbagai provinsi Indonesia.
""")

# Sidebar untuk navigasi
menu = st.sidebar.selectbox("Pilih Menu", ["Tren Nasional", "Prediksi Provinsi", "Analisis Faktor"])

if menu == "Tren Nasional":
    st.header("📈 Tren Nasional Akses Air Bersih")
    st.markdown("Pantau perkembangan rata-rata air bersih per tahun dan prediksi 5 tahun ke depan untuk mendukung SDG 6.1 (akses air minum layak).")
    
    # Agregasi per tahun
    df_year = df.groupby('tahun')['air_bersih'].mean()
    df_year.index = pd.to_datetime(df_year.index, format='%Y')
    
    # Model forecast
    model_hw = ExponentialSmoothing(df_year, trend='add', seasonal=None, damped_trend=True)
    fit = model_hw.fit(optimized=True)
    forecast = fit.forecast(steps=5)
    
    # Plot dengan Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_year.index, y=df_year.values, mode='lines+markers', name='Aktual', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines+markers', name='Prediksi', line=dict(color='darkorange', dash='dash')))
    fig.update_layout(title="Tren dan Prediksi Jumlah Air Bersih (Rata-rata Nasional)", xaxis_title="Tahun", yaxis_title="Jumlah Air Bersih")
    st.plotly_chart(fig)
    
    st.subheader("Hasil Prediksi 5 Tahun ke Depan")
    st.dataframe(forecast.round(2))

elif menu == "Prediksi Provinsi":
    st.header("🔮 Prediksi Akses Air Bersih per Provinsi")
    st.markdown("Masukkan data indikator untuk memprediksi jumlah air bersih. Ini membantu simulasi kebijakan daerah, seperti dampak peningkatan kapasitas terhadap akses air bersih (SDG 6.4: efisiensi air).")
    
    # Input form berdasarkan features
    inputs = {}
    for feat in features:
        inputs[feat] = st.number_input(f"{feat} (numerik)", value=0.0, step=0.01)
    
    if st.button("Prediksi"):
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        st.success(f"Prediksi Jumlah Air Bersih: {pred:.2f}")
        
        # Evaluasi sederhana (opsional, jika ada data test)
        # st.info("Model ini memiliki R² ~0.85 berdasarkan data training.")

elif menu == "Analisis Faktor":
    st.header("📊 Analisis Faktor Pengaruh")
    st.markdown("Lihat fitur yang paling berpengaruh terhadap akses air bersih. Ini membantu prioritas intervensi, seperti fokus pada biaya listrik atau tenaga kerja, untuk mencapai SDG 6.")
    
    # Ambil importance dari model (asumsikan Random Forest)
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Fitur': features, 'Importance': importance}).sort_values(by='Importance', ascending=False).head(10)
    
    # Plot
    fig, ax = plt.subplots()
    ax.barh(feat_imp['Fitur'][::-1], feat_imp['Importance'][::-1])
    ax.set_xlabel('Importance')
    ax.set_title('10 Fitur Terpenting')
    st.pyplot(fig)
    
    st.dataframe(feat_imp)

# Footer
st.markdown("---")
st.markdown("**Dikembangkan untuk mendukung SDG 6**. Jika ada pertanyaan, hubungi pengembang.")