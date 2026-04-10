import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURATION ---
st.set_page_config(page_title="Prediksi Life Expectancy", page_icon="📈")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_assets():
    # Menggunakan nama file standar hasil sinkronisasi tadi
    model = joblib.load('best_modell.pkl')
    scaler = joblib.load('scalerr.pkl')
    features = joblib.load('feature_columnss.pkl')
    return model, scaler, features

model, scaler, features = load_model_assets()

# --- UI INTERFACE ---
st.title("🌍 Life Expectancy AI Predictor")
st.markdown("---")

if features:
    st.sidebar.header("Input Data Parameter")
    
    # Fungsi untuk membuat input dinamis berdasarkan isi feature_columns.pkl
    def get_user_input():
        input_data = {}
        for col in features:
            if col == 'Gender':
                choice = st.sidebar.selectbox("Gender", ["Pria", "Wanita"])
                input_data[col] = 0 if choice == "Pria" else 1
            elif col == 'Year':
                input_data[col] = st.sidebar.number_input("Tahun", value=2024, step=1)
            else:
                # Otomatis membuat slider/number input untuk fitur numerik lainnya
                input_data[col] = st.sidebar.number_input(f"Input {col}", value=0.0, format="%.4f")
        
        return pd.DataFrame(input_data, index=[0])

    df_input = get_user_input()

    # --- PREDICTION LOGIC ---
    if st.button("🚀 Prediksi Sekarang"):
        # 1. Sinkronisasi urutan kolom (Menjamin nama fitur cocok dengan scaler)
        X_final = df_input[features]
        
        # 2. Log Transformation (Hanya untuk fitur ekonomi agar distribusi normal)
        log_cols = ['GDP', 'GNI', 'Per Capita', 'Sucide Rate']
        for col in log_cols:
            if col in X_final.columns:
                X_final[col] = np.log1p(X_final[col])
        
        try:
            # 3. Scaling
            X_scaled = scaler.transform(X_final)
            
            # 4. Predict
            prediction = model.predict(X_scaled)
            
            # 5. Output
            st.success(f"### Estimasi Angka Harapan Hidup: {prediction[0]:.2f} Tahun")
            
            # Dekorasi hasil
            st.progress(min(max(int(prediction[0]), 0), 100))
            st.balloons()
            
        except Exception as e:
            st.error(f"Error teknis: {e}")
            st.info("Saran: Pastikan file scaler.pkl dan feature_columns.pkl Anda sudah sinkron (18 fitur).")

else:
    st.error("File model atau fitur tidak ditemukan di folder!")

st.markdown("---")
st.caption("Developed by Gemini AI Engineer - 2026")