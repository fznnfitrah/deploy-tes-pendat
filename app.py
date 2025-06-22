import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model ===
model = joblib.load("nb_model.pkl")

# === Load contoh data ===
df = pd.read_csv("iris.csv")

st.set_page_config(page_title="Prediksi Iris - Decision Tree", layout="centered")

# === Judul dan deskripsi ===
st.title("🌸 Prediksi Jenis Iris dengan Decision Tree")
st.write("Masukkan nilai fitur di bawah ini untuk memprediksi jenis bunga Iris.")

# === Sidebar untuk input fitur ===
st.sidebar.header("Input Fitur")

# Ambil nama kolom fitur dari CSV (kecuali kolom target 'Class')
feature_columns = [col for col in df.columns if col.lower() != 'class']

# Buat input untuk masing-masing fitur
input_data = []
for feature in feature_columns:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())

    value = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=0.1
    )
    input_data.append(value)

# === Prediksi ===
if st.button("🔍 Prediksi"):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)[0]

    st.success(f"🌟 Hasil Prediksi: **{prediction}**")

# === Opsi: tampilkan data contoh ===
with st.expander("📄 Lihat Dataset Contoh"):
    st.dataframe(df.head())
