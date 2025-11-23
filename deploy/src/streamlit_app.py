import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Usia Wajah - ANN",
    layout="centered",
    initial_sidebar_state="expanded"
)

def run():
    st.title("Prediksi Usia Wajah Menggunakan Artificial Neural Network (ANN)")

    # Gambar header
    st.image(
        "https://wallpapers.com/images/hd/group-of-people-silhouette-with-reflection-35gvku3mvmyb7rno.jpg",
        caption="Ilustrasi deteksi usia wajah - sumber: Google"
    )

    # Latar belakang
    with st.expander("📘 Tentang Proyek Ini", expanded=False):
        st.markdown("""
        Dalam era teknologi modern, sistem kecerdasan buatan (AI) banyak diterapkan 
        dalam pengenalan wajah (*face recognition*).  
        Salah satu penerapannya adalah **prediksi usia wajah** menggunakan model 
        **Artificial Neural Network (ANN)** untuk mengelompokkan wajah manusia ke 
        dalam kategori seperti **Anak-anak**, **Dewasa**, atau **Lanjut Usia**.
        """)

    # Load model
    model = tf.keras.models.load_model('src/improved_ann_model.keras')

    # --- Ambil otomatis ukuran input model
    input_shape = model.input_shape[1:4]  # (height, width, channel)
    st.info(f"Model membutuhkan ukuran gambar: {input_shape[0]}x{input_shape[1]} piksel (RGB)")

    # Upload image
    st.header("📷 Unggah Gambar untuk Prediksi")
    uploaded_file = st.file_uploader(
        "Seret dan lepas file gambar di sini, atau klik untuk memilih",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # Menampilkan gambar
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)

        # --- Preprocessing otomatis sesuai model
        img_resized = image.resize((input_shape[0], input_shape[1]))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --- Prediksi
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction, axis=1)[0]

        # --- Mapping label (ubah sesuai model)
        label_map = {0: "Anak-anak 👶", 1: "Dewasa 🧑", 2: "Lanjut Usia 👴"}
        result_label = label_map.get(pred_class, "Tidak diketahui")

        with col2:
            st.markdown("### 🔍 Hasil Prediksi:")
            st.success(f"**{result_label}**")
            st.markdown("### 📊 Probabilitas per Kelas:")

            fig, ax = plt.subplots()
            ax.bar(label_map.values(), prediction[0], color=["skyblue", "orange", "lightgreen"])
            ax.set_ylabel("Probabilitas")
            ax.set_ylim(0, 1)
            ax.set_title("Distribusi Probabilitas Prediksi")
            st.pyplot(fig)

    else:
        st.warning("📤 Silakan unggah gambar wajah terlebih dahulu untuk memulai prediksi.")

    # Tentang model
    with st.expander("🧩 Arsitektur Model"):
        st.markdown("""
        Model ANN terdiri dari beberapa lapisan utama:
        - **Flatten Layer** untuk mengubah citra menjadi vektor 1D  
        - **Dense Layer (512, 256, 128 neuron)** dengan aktivasi ReLU  
        - **Batch Normalization** untuk menstabilkan pelatihan  
        - **Dropout** untuk mengurangi overfitting  
        - **Output Layer (Softmax)** dengan 3 neuron untuk klasifikasi  
        """)

    with st.expander("📈 Kesimpulan"):
        st.markdown("""
        Model ANN yang telah di-*improve* menunjukkan performa yang lebih stabil dan akurasi yang meningkat.  
        Dengan arsitektur yang optimal, model ini mampu memperkirakan kategori usia wajah dengan hasil yang cukup baik.
        """)

if __name__ == '__main__':
    run()
