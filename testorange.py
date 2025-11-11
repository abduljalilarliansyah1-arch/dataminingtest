import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Prediksi Iris", page_icon="🌸", layout="centered")

st.title("🌸 Prediksi Spesies Bunga Iris")
st.write("Model ini menggunakan algoritma **k-Nearest Neighbors (kNN)** untuk memprediksi jenis bunga Iris.")

# ==============================
# 1. Load model
# ==============================
model_path = r"E:\tugas\data mining\test.pkcls"

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ==============================
# 2. Input fitur
# ==============================
st.subheader("Masukkan Data Fitur:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width  = st.number_input("Sepal Width (cm)",  min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width  = st.number_input("Petal Width (cm)",  min_value=0.0, step=0.1)

# ==============================
# 3. Prediksi
# ==============================
if st.button("Prediksi"):
    try:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Beberapa model Orange/scikit-learn bisa dipanggil langsung dengan ()
        prediction = model(input_data)

        # Deteksi tipe model (Orange atau scikit-learn)
        if hasattr(model, "domain") and hasattr(model.domain.class_var, "values"):
            label = model.domain.class_var.values[int(prediction[0])]
        else:
            label = str(prediction[0])

        st.markdown(
            f"<h3 style='text-align:center;'>🌼 Hasil Prediksi: "
            f"<span style='color:#008000;'>{label}</span></h3>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ==============================
# 4. Footer
# ==============================
st.write("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit dan model PKCLS Anda.")