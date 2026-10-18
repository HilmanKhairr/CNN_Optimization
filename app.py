import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Label dan icon untuk setiap kelas objek
CLASS_INFO = {
    'airplane': {'nama': 'Pesawat Terbang', 'icon': '✈️'},
    'automobile': {'nama': 'Mobil', 'icon': '🚗'},
    'bird': {'nama': 'Burung', 'icon': '🐦'},
    'cat': {'nama': 'Kucing', 'icon': '🐱'},
    'deer': {'nama': 'Rusa', 'icon': '🦌'},
    'dog': {'nama': 'Anjing', 'icon': '🐶'},
    'frog': {'nama': 'Katak', 'icon': '🐸'},
    'horse': {'nama': 'Kuda', 'icon': '🐴'},
    'ship': {'nama': 'Kapal Laut', 'icon': '🚢'},
    'truck': {'nama': 'Truk', 'icon': '🚚'}
}

@st.cache_resource
def load_cnn_model():
    model_path = 'cifar10_cnn_optimized.keras'
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan.")
        st.stop()
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_names():
    json_path = 'class_names.json'
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return list(CLASS_INFO.keys())

model = load_cnn_model()
class_names = load_class_names()

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarHeader"] {
            display: none !important;
        }
        [data-testid="stSidebarUserContent"], section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Team Assignment 2: Optimasi Model & Aplikasi")
    st.write("**Mata Kuliah:** Artificial Intelligence (COSC6023036)")
    st.write("**Kelompok 5:**")
    st.markdown("""
    1. Adinda Diva Latifah (2902827270)
    2. Hilman Wasiandi Khairurrizal (2902817490)
    3. Mochamad Arif Rahman (2902820094)
    4. Muhammad Arizal Ma'ruf (2902820182)
    """)
    st.divider()

    st.subheader("Arsitektur Model")
    st.markdown("""
    * **Model:** Optimized Dual-Conv CNN
    * **Dataset:** CIFAR-10 (10 Kelas)
    * **Input:** 32 x 32 x 3 (RGB)
    * **Fitur Optimasi:**
      - Batch Normalization
      - Dropout (0.2 – 0.5)
      - Data Augmentation
      - Learning Rate Scheduler
    """)
    st.divider()

    st.subheader("Daftar Kelas Objek")
    for name in class_names:
        info = CLASS_INFO.get(name, {'nama': name, 'icon': ''})
        st.write(f"{info['icon']} **{info['nama']}** *({name})*")

# Main Content
st.title("Klasifikasi Citra Objek CIFAR-10")
st.write("Aplikasi untuk mendeteksi objek pada gambar menggunakan model Convolutional Neural Network (CNN).")

uploaded_file = st.file_uploader(
    "Pilih file gambar",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.subheader("Gambar Input")
        st.image(image, caption=f"Resolusi asli: {image.size[0]} x {image.size[1]} px", use_container_width=True)
        
        img_resized = image.resize((32, 32))
        with st.expander("Lihat gambar hasil resize (32x32 px)"):
            st.image(img_resized, width=128)
            st.caption("Ukuran input yang diproses oleh model CNN.")

    with col2:
        st.subheader("Hasil Prediksi")
        
        with st.spinner("Memproses gambar..."):
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_batch, verbose=0)[0]
            
            top_idx = int(np.argmax(predictions))
            top_class = class_names[top_idx]
            top_confidence = float(predictions[top_idx]) * 100
            top_info = CLASS_INFO.get(top_class, {'nama': top_class, 'icon': ''})

        st.success(f"Prediksi Utama: {top_info['icon']} **{top_info['nama']}** *({top_class.upper()})*")
        
        if top_confidence >= 70:
            status_text = "Tinggi (Akurat)"
        elif top_confidence >= 40:
            status_text = "Cukup"
        else:
            status_text = "Rendah"

        st.metric("Tingkat Keyakinan", f"{top_confidence:.2f}%", delta=status_text)
        st.progress(float(predictions[top_idx]))
        
        st.divider()
        st.markdown("**Distribusi Probabilitas:**")
        
        df_prob = pd.DataFrame({
            'Kelas Objek': [f"{CLASS_INFO[c]['icon']} {CLASS_INFO[c]['nama']}" for c in class_names],
            'Probabilitas (%)': [round(float(p) * 100, 2) for p in predictions]
        }).sort_values(by='Probabilitas (%)', ascending=False)
        
        st.bar_chart(df_prob.set_index('Kelas Objek'), color="#1f77b4")
        
        with st.expander("Detail Top-5 Prediksi"):
            st.dataframe(df_prob.head(5), use_container_width=True, hide_index=True)

else:
    st.info("Silakan unggah gambar objek (misal: pesawat, mobil, kucing, anjing, kapal) untuk melihat prediksi model.")