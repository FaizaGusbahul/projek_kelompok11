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
default_path = "TUGAS_DATA_KELOMPOK11/DATA_WATER_SUPPLY_STATISTICS.csv"

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
manual_prov = st.sidebar.selectbox("Pilih kolom Provinsi (optional)", options=[None] + df.columns.tolist())

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
# 6. PREDIKSI PROVINSI (dengan selector provinsi + default median per provinsi)
# ===================== BLOK "🔮 Prediksi Provinsi" (PERBAIKAN: TIDAK HILANG INDIKATOR) =====================
elif menu == "🔮 Prediksi Akses Air Bersih per Provinsi":
    st.header("🔮 Prediksi Akses Air Bersih per Provinsi (fix indikator hilang)")

    model_path = "model.pkl"
    scaler_path = "scaler.pkl"
    features_path = "features.json"

    if not os.path.exists(model_path):
        st.error("❌ File `model.pkl` tidak ditemukan. Pastikan file model sudah ada di folder proyek.")
    else:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        def norm_name(s: str):
            return str(s).strip().lower().replace(" ", "_")

        # deteksi provinsi column (manual_prov jika ada)
        prov_candidates = ["provinsi", "province", "nama_provinsi", "kabupaten_provinsi", "region"]
        prov_col = manual_prov if ('manual_prov' in locals() and manual_prov) else find_best_column(df.columns.tolist(), prov_candidates)

        if prov_col:
            df['_prov__clean'] = df[prov_col].astype(str).str.strip()
            df['_prov__clean_lower'] = df['_prov__clean'].str.lower()
            unique_prov = sorted(df['_prov__clean'].dropna().unique().tolist())
        else:
            unique_prov = []

        # 1) Dapatkan expected_features dari model/scaler/features.json (tanpa langsung memaksa ini jadi widget)
        expected_features = None
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            st.info("Menggunakan feature_names_in_ dari model.")
        elif os.path.exists(features_path):
            try:
                with open(features_path, "r") as f:
                    expected_features = json.load(f)
                    st.info("Menggunakan fitur dari features.json.")
            except Exception:
                expected_features = None
        elif scaler is not None and hasattr(scaler, "feature_names_in_"):
            expected_features = list(scaler.feature_names_in_)
            st.info("Menggunakan feature_names_in_ dari scaler.")
        # fallback ke kolom numerik DF
        if not expected_features:
            expected_features = df.select_dtypes(include='number').columns.tolist()
            st.info("Fallback ke kolom numerik dataset sebagai expected_features.")

        st.write("DEBUG: expected_features (what model expects):", expected_features[:20], " ... (total {})".format(len(expected_features)))

        # 2) Coba match expected_features ke kolom dataset (normalize vs original)
        # buat peta normalized df cols -> original df col name
        df_cols_norm_map = {col: norm_name(col) for col in df.columns.tolist()}
        norm_to_dfcol = {v: k for k, v in df_cols_norm_map.items()}

        matched_map = {}         # expected_feature -> matched_df_col (or None)
        widget_features = []     # list of expected_feature names that we will render (preserve model feature key)

        for feat in expected_features:
            norm_feat = norm_name(feat)
            matched = None
            # direct normalized match
            if norm_feat in norm_to_dfcol:
                matched = norm_to_dfcol[norm_feat]
            else:
                # fuzzy match against normalized df column names
                df_norm_list = list(df_cols_norm_map.values())
                close = difflib.get_close_matches(norm_feat, df_norm_list, n=1, cutoff=0.7)
                if close:
                    matched = norm_to_dfcol[close[0]]
            matched_map[feat] = matched
            # if matched is not None, we can use this expected feature as widget, otherwise postpone
            if matched:
                widget_features.append(feat)

        # If after matching we got NO widget_features, fallback to df numeric columns
        if not widget_features:
            st.warning("Tidak dapat mencocokkan expected_features ke kolom dataset. Menggunakan kolom numerik dataset sebagai indikator.")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            # Set widget_features to be those column names (we'll use them as both expected key and label)
            widget_features = numeric_cols
            # Rebuild matched_map for these numeric columns (key = col, matched = col)
            matched_map = {col: col for col in numeric_cols}
            expected_features = numeric_cols  # override expected_features for downstream building

        st.write("DEBUG: widget_features (akan ditampilkan):", widget_features[:20], " ... (total {})".format(len(widget_features)))

        # UI: pilih provinsi
        if prov_col and unique_prov:
            sel_prov_raw = st.selectbox("Pilih Provinsi", options=["-- Pilih --"] + unique_prov)
            sel_prov = None if sel_prov_raw == "-- Pilih --" else sel_prov_raw
        else:
            sel_prov = None
            st.info("Kolom provinsi tidak ditemukan di dataset — nilai default per provinsi tidak tersedia.")

        # Hitung median global dan per-prov untuk setiap matched feature
        global_medians = {}
        prov_medians = {}
        for feat in widget_features:
            dfcol = matched_map.get(feat) if matched_map.get(feat) else feat
            if dfcol in df.columns:
                global_medians[feat] = pd.to_numeric(df[dfcol], errors="coerce").median(skipna=True)
                if sel_prov is not None:
                    mask = df['_prov__clean_lower'] == str(sel_prov).strip().lower()
                    ser = pd.to_numeric(df.loc[mask, dfcol], errors="coerce")
                    prov_medians[feat] = ser.median(skipna=True) if not ser.empty else None
                else:
                    prov_medians[feat] = None
            else:
                global_medians[feat] = None
                prov_medians[feat] = None

        allow_manual = st.checkbox("Izinkan edit manual nilai indikator setelah autofill", value=False)
        st.write("Nilai input akan diisi otomatis dari median provinsi (jika ada).")

        # Render inputs. Use widget_features list (these keys are either expected feature names or df numeric col names)
        inputs_for_prediction = {}
        for feat in widget_features:
            default_val = 0.0
            if prov_medians.get(feat) is not None and not pd.isna(prov_medians.get(feat)):
                default_val = float(prov_medians[feat])
            elif global_medians.get(feat) is not None and not pd.isna(global_medians.get(feat)):
                default_val = float(global_medians[feat])
            else:
                default_val = 0.0

            disabled_flag = (sel_prov is not None) and (not allow_manual)
            keyname = f"pred_{feat}"
            try:
                inputs_for_prediction[feat] = st.number_input(label=f"{feat}", value=default_val, step=0.01, format="%f", key=keyname, disabled=disabled_flag)
            except TypeError:
                inputs_for_prediction[feat] = st.number_input(label=f"{feat} (default: {default_val})", value=default_val, step=0.01, format="%f", key=keyname)

        if sel_prov:
            st.caption(f"Default diisi dengan median provinsi: {sel_prov}. (Centang 'Izinkan edit manual' untuk mengubah nilai.)")
        else:
            st.caption("Tidak ada provinsi dipilih — nilai default memakai median global atau 0.0.")

        # Prediksi: butuh membangun X_input yang sesuai expected_features order & names.
        if st.button("Prediksi"):
            # Build final expected_features list to use for model input:
            # If we still have model.feature_names_in_ originally (and matched_map had matches), keep that.
            final_expected = expected_features

            # Create dict values for final_expected: if we have an input for that feat use it, else fallback 0
            final_input_dict = {}
            for feat in final_expected:
                if feat in inputs_for_prediction:
                    final_input_dict[feat] = inputs_for_prediction[feat]
                else:
                    # try to find by matched_map reverse: maybe expected feature maps to dfcol which we have
                    mapped_col = matched_map.get(feat)
                    if mapped_col and mapped_col in inputs_for_prediction:
                        final_input_dict[feat] = inputs_for_prediction[mapped_col]
                    else:
                        # fallback: try normalized name match with inputs_for_prediction keys
                        normf = norm_name(feat)
                        found = None
                        for k in inputs_for_prediction.keys():
                            if norm_name(k) == normf:
                                found = inputs_for_prediction[k]
                                break
                        final_input_dict[feat] = found if found is not None else 0.0

            X_input = pd.DataFrame([final_input_dict], columns=final_expected)

            # Ensure types numeric
            X_input = X_input.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            # Apply scaler if available
            try:
                if scaler is not None:
                    X_scaled = scaler.transform(X_input)
                else:
                    X_scaled = X_input.values
            except Exception as e:
                st.error(f"Gagal mengaplikasikan scaler: {e}")
                # fallback to values
                X_scaled = X_input.values

            # Predict
            try:
                pred = model.predict(X_scaled)[0]
                st.success(f"💧 Prediksi jumlah air bersih: **{pred:.2f}**")
                if sel_prov:
                    st.write(f"Provinsi: **{sel_prov}**")
            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

        # cleanup temp cols
        if '_prov__clean' in df.columns:
            df.drop(columns=[c for c in ['_prov__clean', '_prov__clean_lower'] if c in df.columns], inplace=True)
# ========================================================================================



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
