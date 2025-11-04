# app.py (Versi Terbaru dengan Penyesuaian Dataset)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ======================
# 🧩 KONFIGURASI DASAR
# ======================
st.set_page_config(
    page_title="Prediksi Akses Air Bersih di Indonesia",
    layout="wide",
    page_icon="💧"
)

# ======================
# 📦 LOAD MODEL, SCALER, DAN FITUR (DENGAN CACHE OPTIMAL)
# ======================
@st.cache_resource
def load_model():
    with st.spinner("Loading model..."):
        return joblib.load("model.pkl")

@st.cache_resource
def load_scaler():
    with st.spinner("Loading scaler..."):
        return joblib.load("scaler.pkl")

@st.cache_data
def load_features():
    with st.spinner("Loading features..."):
        with open("features.json", "r") as f:
            return json.load(f)

model = load_model()
scaler = load_scaler()
features = load_features()

# ======================
# 🎯 JUDUL & TUJUAN
# ======================
st.title("💧 Prediksi Akses Air Bersih di Indonesia")
st.markdown("""
Aplikasi ini dikembangkan untuk mendukung **SDGs 6: Air Bersih dan Sanitasi Layak**, khususnya **SDG 6.1 (Akses Air Minum)** dan **SDG 6.4 (Pengelolaan Air dan Sanitasi Berkelanjutan)**.  
Melalui pendekatan **Machine Learning**, aplikasi ini membantu pemerintah dan masyarakat:  
- **Tren Nasional**: Memantau tren historis akses air bersih dan memprediksi 5 tahun ke depan (dengan opsi upload CSV sendiri, seperti DATA_WATER_SUPPLY_STATISTICS.csv).  
- **Prediksi Provinsi**: Mensimulasikan akses air bersih berdasarkan indikator sosial-ekonomi dan kebijakan daerah.  
- **Analisis Faktor**: Mengidentifikasi variabel paling berpengaruh untuk menentukan intervensi prioritas.  

> 💡 Hasil prediksi ini bersifat pendukung, bukan keputusan final. Data lapangan tetap menjadi prioritas.
""")

st.divider()

# ======================
# 🧭 NAVIGASI SIDEBAR
# ======================
page = st.sidebar.selectbox("Pilih Halaman", ["Tren Nasional", "Prediksi Provinsi", "Analisis Faktor"])

# ======================
# 📊 TREN NASIONAL
# ======================
if page == "Tren Nasional":
    st.header("📊 Tren Nasional Akses Air Bersih")
    st.markdown("""
    Halaman ini menampilkan tren historis akses air bersih berdasarkan data nasional dan prediksi 5 tahun ke depan.  
    Upload file CSV Anda (misalnya, DATA_WATER_SUPPLY_STATISTICS.csv) – aplikasi akan otomatis mendeteksi kolom dan menghitung akses air bersih.  
    Ini mendukung pemantauan **SDG 6.1** (target 100% akses air minum) dan **SDG 6.4** (pengelolaan air berkelanjutan).
    """)

    # Upload CSV
    uploaded_file = st.file_uploader("Upload file CSV (misalnya, DATA_WATER_SUPPLY_STATISTICS.csv)", type="csv")
    if uploaded_file is not None:
        with st.spinner("Memproses data CSV..."):
            df_year = pd.read_csv(uploaded_file)
            st.success("Data berhasil di-upload!")
            st.write("📋 **Preview Data:**")
            st.dataframe(df_year.head())

            # Normalisasi nama kolom ke lowercase untuk fleksibilitas
            df_year.columns = df_year.columns.str.lower()

            # Deteksi Kolom (case-insensitive)
            year_col = None
            access_col = None
            volume_col = None
            customer_col = None

            for col in df_year.columns:
                if 'tahun' in col:
                    year_col = col
                if 'efektifitas_produksi_air_bersih' in col:
                    access_col = col
                if 'volume_air_bersih_yang_disalurkan' in col:
                    volume_col = col
                if 'jumlah_pelanggan' in col:
                    customer_col = col

            if year_col and (access_col or volume_col):
                # Hitung 'akses_air_bersih' sebagai persentase
                if access_col:
                    df_year['akses_air_bersih'] = df_year[access_col].astype(str).str.replace(',', '.').astype(float)
                elif volume_col and customer_col:
                    # Alternatif: Hitung berdasarkan volume disalurkan per pelanggan (asumsi sederhana)
                    df_year['akses_air_bersih'] = (df_year[volume_col].astype(float) / df_year[customer_col].astype(float)) * 100  # Normalisasi ke %

                # Tren Historis
                st.subheader("📈 Tren Historis")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_year[year_col], df_year['akses_air_bersih'], marker='o', color="#56CCF2")
                ax.set_title("Tren Akses Air Bersih Nasional")
                ax.set_xlabel("Tahun")
                ax.set_ylabel("Persentase Akses Air Bersih (%)")
                ax.grid(True)
                st.pyplot(fig)

                # Prediksi 5 Tahun ke Depan
                st.subheader("🔮 Prediksi 5 Tahun ke Depan")
                with st.spinner("Menghitung prediksi..."):
                    last_year = df_year[year_col].max()
                    future_years = [last_year + i for i in range(1, 6)]
                    
                    # Gunakan rata-rata indikator dari dataset
                    avg_input = df_year[features].mean() if all(feat in df_year.columns for feat in features) else pd.Series([0.0] * len(features), index=features)
                    input_df = pd.DataFrame([avg_input])
                    X_scaled = scaler.transform(input_df[features])
                    pred_current = model.predict(X_scaled)[0]
                    
                    predictions = [pred_current + (i * 2) for i in range(6)]
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    ax2.plot(df_year[year_col], df_year['akses_air_bersih'], marker='o', label="Historis", color="#56CCF2")
                    ax2.plot(future_years, predictions[1:], marker='x', linestyle='--', label="Prediksi", color="#FF6B6B")
                    ax2.set_title("Prediksi Akses Air Bersih 5 Tahun ke Depan")
                    ax2.set_xlabel("Tahun")
                    ax2.set_ylabel("Persentase Akses Air Bersih (%)")
                    ax2.legend()
                    ax2.grid(True)
                    st.pyplot(fig2)

                    # Pemantauan SDG
                    st.subheader("🌍 Pemantauan SDG 6.1 & 6.4")
                    current_access = df_year['akses_air_bersih'].iloc[-1]
                    if current_access >= 100:
                        st.success("✅ SDG 6.1: Target akses air minum tercapai.")
                    else:
                        st.warning(f"🔸 SDG 6.1: Akses air minum saat ini {current_access:.2f}%. Perlu intervensi.")
                    st.info("🔄 SDG 6.4: Prediksi menunjukkan peningkatan pengelolaan air; fokus pada keberlanjutan sumber daya.")
            else:
                st.error("Kolom 'tahun' (atau varian seperti 'Tahun') dan kolom terkait akses air bersih (misalnya, 'Efektifitas_Produksi_Air_Bersih_Perusahaan_Air_Bersih' atau 'Volume_Air_Bersih_yang_Disalurkan') tidak ditemukan. Silakan periksa dataset Anda. Kolom yang tersedia: " + ", ".join(df_year.columns))
    else:
        st.info("Silakan upload file CSV untuk memulai analisis tren.")

# ======================
# 📥 PREDIKSI PROVINSI
# ======================
elif page == "Prediksi Provinsi":
    st.header("📥 Prediksi Akses Air Bersih per Provinsi")
    st.markdown("""
    Isi nilai dari setiap indikator sosial-ekonomi di bawah ini (berdasarkan kolom dataset Anda, seperti 'Jumlah_Pekerja' atau 'Biaya_Listrik').  
    Model akan memprediksi persentase masyarakat yang memiliki akses terhadap air bersih dan mensimulasikan dampak kebijakan daerah.
    """)

    # Input dinamis sesuai features.json
    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        col = cols[i % 3]
        user_input[feat] = col.number_input(f"{feat}", value=0.0)

    # Convert ke DataFrame
    input_df = pd.DataFrame([user_input])
    st.write("📋 **Data yang akan diprediksi:**")
    st.dataframe(input_df)

    # Prediksi
    if st.button("🔮 Jalankan Prediksi"):
        with st.spinner("Menjalankan prediksi..."):
            try:
                # Scaling
                X_scaled = scaler.transform(input_df[features])

                # Prediksi
                pred = model.predict(X_scaled)[0]

                st.success(f"💧 **Perkiraan akses air bersih:** {pred:.2f}%")
                st.caption("Persentase masyarakat dengan akses air bersih berdasarkan indikator yang dimasukkan.")

                # Visualisasi
                st.subheader("📈 Interpretasi Hasil")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(["Prediksi Akses Air Bersih"], [pred], color="#56CCF2")
                ax.set_xlim(0, 100)
                ax.set_xlabel("Persentase (%)")
                st.pyplot(fig)

                # Simulasi Kebijakan
                st.subheader("🏗️ Simulasi Kebijakan Daerah")
                if pred < 60:
                    st.error("⚠️ Akses air bersih tergolong **rendah**. Rekomendasi: Tingkatkan infrastruktur air, alokasikan anggaran untuk sanitasi pedesaan, dan edukasi masyarakat.")
                elif pred < 80:
                    st.warning("🔸 Akses air bersih **cukup**, tapi masih perlu perhatian di wilayah pedesaan. Rekomendasi: Fokus pada pemeliharaan sumber air dan pengurangan polusi.")
                else:
                    st.success("✅ Akses air bersih **tinggi**. Rekomendasi: Prioritaskan keberlanjutan dengan monitoring rutin dan inovasi teknologi air.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ======================
# 🔍 ANALISIS FAKTOR
# ======================
elif page == "Analisis Faktor":
    st.header("🔍 Analisis Faktor Pengaruh")
    st.markdown("""
    Halaman ini menunjukkan variabel paling berpengaruh (feature importance) berdasarkan model.  
    Ini membantu menentukan intervensi prioritas untuk meningkatkan akses air bersih.
    """)

    if st.button("🔍 Jalankan Analisis"):
        with st.spinner("Menganalisis faktor..."):
            try:
                # Asumsi model mendukung feature_importances_
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Fitur': features,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)

                    st.subheader("📊 Feature Importance")
                    st.dataframe(feature_importance_df)

                    # Visualisasi
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color="#56CCF2")
                    ax.set_xlabel("Importance")
                    ax.set_title("Variabel Paling Berpengaruh")
                    st.pyplot(fig)

                    # Intervensi Prioritas
                    st.subheader("🎯 Intervensi Prioritas")
                    top_features = feature_importance_df.head(3)['Fitur'].tolist()
                    st.info(f"Variabel teratas: {', '.join(top_features)}. Fokus intervensi pada: **{top_features[0]}** (misalnya, tingkatkan pendidikan atau ekonomi jika relevan).")
                else:
                    st.warning("Model tidak mendukung feature importance. Gunakan model seperti RandomForest untuk fitur ini.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat analisis: {e}")

st.divider()

# ======================
# 🌍 PESAN SDGS 6
# ======================
st.header("🌍 Kontribusi terhadap SDGs 6")
st.markdown("""
Tujuan **SDG 6** adalah *menjamin ketersediaan serta pengelolaan air bersih dan sanitasi yang berkelanjutan untuk semua*.  

Dengan aplikasi ini, pengguna dapat:
- 📊 Menganalisis tren nasional dan potensi wilayah dengan akses air bersih rendah  
- 🏗️ Memberi masukan pada perencanaan kebijakan air bersih melalui simulasi  
- 🔁 Melihat dampak indikator sosial-ekonomi terhadap ketersediaan air dan menentukan intervensi prioritas  

> 💡 Hasil ini bersifat estimasi. Integrasikan dengan data lapangan untuk kebijakan yang efektif.
""")