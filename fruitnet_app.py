import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ====================
# Fungsi load model
# ====================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fruitnet_model.h5")
    return model

# ====================
# Fungsi load label
# ====================
@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# ====================
# Layout UI
# ====================
st.title("FruitNet 🍓")
st.header("Klasifikasi Gambar Buah")

uploaded_file = st.file_uploader("Upload gambar buah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", width=150)

    # Preprocessing gambar
    image = image.resize((100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 100, 100, 3)
    img_array = img_array / 255.0  # Normalisasi

    # Load model dan labels
    model = load_model()
    labels = load_labels()

    # ====================
    # Prediksi
    # ====================
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)

    # Debugging info
    st.write("Nilai prediksi:", pred)
    st.write("Index prediksi:", pred_index)
    st.write("Jumlah label:", len(labels))

    # Validasi & tampilkan hasil
    if pred_index < len(labels):
        st.success(f"✅ Prediksi: **{labels[pred_index]}**")
    else:
        st.error("❌ Gagal mengenali gambar: label tidak tersedia.")
