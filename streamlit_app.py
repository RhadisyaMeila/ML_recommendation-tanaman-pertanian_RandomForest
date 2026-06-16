import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Konfigurasi Halaman & Tema
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main { background-color: #f9fbf9; }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    h1 { color: #1b5e20; text-align: center; }
    .result-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #1b5e20;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Sistem Rekomendasi Tanaman Pertanian")
st.write("Masukkan kondisi nutrisi tanah dan parameter lingkungan di bawah ini untuk mengetahui jenis tanaman yang paling optimal ditanam.")
st.markdown("---")

# 2. Load Model dan Scaler
@st.cache_resource
def load_artifacts():
    # Perhatikan penambahan 'src/' di depan nama file
    with open('src/model_rf.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('src/scaler.pkl', 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Error: File 'model_rf.pkl' atau 'scaler.pkl' tidak ditemukan di repositori Space Anda.")

# 3. Form Input Data
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Kandungan Nutrisi Tanah")
    n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90, step=1)
    p = st.number_input("Fosfor (P)", min_value=0, max_value=200, value=42, step=1)
    k = st.number_input("Kalium (K)", min_value=0, max_value=200, value=43, step=1)
    ph = st.number_input("Tingkat Keasaman (pH Tanah)", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

with col2:
    st.subheader("🌤️ Kondisi Mikroklimat")
    temp = st.number_input("Suhu (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
    hum = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    rain = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=200.0, step=0.1)

st.markdown("---")

# 4. Proses Prediksi
if st.button("Analisis & Berikan Rekomendasi"):
    # Buat array data input
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    
    # Lakukan transformasi data dengan scaler bawaan training
    input_scaled = scaler.transform(input_data)
    
    # Prediksi menggunakan model Random Forest
    prediction = model.predict(input_scaled)[0]
    
    # Menampilkan Hasil
    st.markdown("### 📊 Hasil Rekomendasi Tanaman:")
    st.markdown(f"""
        <div class="result-box">
            Tanaman yang paling cocok untuk lahan Anda adalah: <br>
            <span style="font-size: 28px; text-transform: uppercase;">✨ {prediction} ✨</span>
        </div>
    """, unsafe_allow_html=True)
    st.balloons()