"""
Streamlit Web Application: Cat and Dog Recognition Using Convolutional Networks.
Unified Full-Page Interactive Dashboard featuring single-image prediction,
Grad-CAM explainability, model evaluation metrics, and architectural overview.
Fully optimized for high-contrast dark mode readability.
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Cat & Dog Recognition | Deep Learning",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Styling (Dark Mode & Light Mode Compatible)
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #60a5fa !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #cbd5e1 !important;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .cat-badge {
        background-color: rgba(37, 99, 235, 0.25) !important;
        color: #93c5fd !important;
        border: 1.5px solid #3b82f6 !important;
        padding: 8px 20px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.4rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
    }
    .dog-badge {
        background-color: rgba(234, 88, 12, 0.25) !important;
        color: #fdba74 !important;
        border: 1.5px solid #f97316 !important;
        padding: 8px 20px;
        border-radius: 24px;
        font-weight: 800;
        font-size: 1.4rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(249, 115, 22, 0.2);
    }
    .section-header {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc !important;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
        border-bottom: 2.5px solid #3b82f6 !important;
        padding-bottom: 0.5rem;
    }
    .highlight-label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    .caption-text {
        color: #94a3b8 !important;
        font-size: 0.88rem;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    from src.predict import CatDogPredictor
    model_path = Path("models/best_model.pt")
    config_path = Path("models/model_config.json")
    if not model_path.exists() or not config_path.exists():
        return None
    return CatDogPredictor(str(model_path), str(config_path))


@st.cache_data
def load_metrics():
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    st.markdown('<div class="main-title">🐾 Cat and Dog Recognition Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Klasifikasi Citra Berbasis Deep Learning & Transfer Learning (PyTorch + MobileNetV2)</div>', unsafe_allow_html=True)

    metrics = load_metrics()
    predictor = load_predictor()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Informasi Model")
        if metrics:
            best_label = metrics.get("best_model_selected", "MobileNetV2 Fine-Tuned")
            test_m = metrics.get("final_holdout_test_metrics", {})
            st.success(f"**Model Aktif:** {best_label}")
            st.metric("Test Accuracy", f"{test_m.get('accuracy', 0)*100:.2f}%")
            st.metric("Test F1-Score", f"{test_m.get('f1_score', 0)*100:.2f}%")
            st.metric("Test ROC-AUC", f"{test_m.get('roc_auc', 0):.4f}")
        else:
            st.info("Model / metrik sedang dimuat...")

        st.markdown("---")
        st.subheader("📋 Dataset & Split")
        st.markdown("""
        - **Total Dataset:** 3.000 citra JPEG RGB
        - **Training Internal:** 1.600 citra (80%)
        - **Validation Internal:** 400 citra (20%)
        - **Hold-Out Test:** 1.000 citra (arsip validation)
        - **Label:** `cats = 0`, `dogs = 1`
        """)

        st.markdown("---")
        st.subheader("💡 Contoh Cepat (Demo Samples)")
        sample_options = {
            "Contoh Kucing 1 (cat.2000.jpg)": "dataset/validation/cats/cat.2000.jpg",
            "Contoh Kucing 2 (cat.2001.jpg)": "dataset/validation/cats/cat.2001.jpg",
            "Contoh Anjing 1 (dog.2000.jpg)": "dataset/validation/dogs/dog.2000.jpg",
            "Contoh Anjing 2 (dog.2001.jpg)": "dataset/validation/dogs/dog.2001.jpg"
        }
        chosen_sample = st.selectbox("Pilih gambar dari hold-out test:", list(sample_options.keys()), index=0)

    # -------------------------------------------------------------
    # SECTION 1: INTERACTIVE IMAGE PREDICTION & EXPLAINABILITY
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">1. 🔍 Pengujian & Prediksi Citra Interaktif</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    img_to_process = None

    with col1:
        st.subheader("Pilih atau Unggah Gambar")
        uploaded_file = st.file_uploader(
            "Upload gambar hewan (format JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Pilih file gambar berformat JPG atau PNG"
        )

        if uploaded_file is not None:
            try:
                img_to_process = Image.open(uploaded_file).convert("RGB")
                st.image(img_to_process, caption="Gambar yang Diunggah", use_container_width=True)
            except Exception as e:
                st.error(f"Gagal membaca file gambar: {e}")
        elif chosen_sample and sample_options[chosen_sample]:
            sample_path = sample_options[chosen_sample]
            if os.path.exists(sample_path):
                img_to_process = Image.open(sample_path).convert("RGB")
                st.image(img_to_process, caption=f"Contoh Terpilih: {chosen_sample}", use_container_width=True)
            else:
                st.warning("Path contoh tidak ditemukan.")

    with col2:
        st.subheader("Hasil Inferensi & Interpretabilitas")
        if img_to_process is not None:
            if predictor is None:
                st.error("Model checkpoint belum tersedia di folder `models/`.")
            else:
                with st.spinner("Menganalisis citra dengan CNN..."):
                    res = predictor.predict(img_to_process, generate_cam=True)

                pred_class = res["predicted_class"]
                conf = res["confidence"]
                p_cat = res["cat_probability"]
                p_dog = res["dog_probability"]

                badge_class = "cat-badge" if pred_class == "Cat" else "dog-badge"
                icon = "🐱" if pred_class == "Cat" else "🐶"

                st.markdown(f'<div class="{badge_class}">{icon} Terdeteksi: {pred_class.upper()}</div>', unsafe_allow_html=True)
                st.write("")
                st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{conf:.2f}%")

                st.markdown("**Probabilitas Prediksi:**")
                st.write(f"🐱 **Cat (Kucing):** {p_cat:.2f}%")
                st.progress(float(p_cat / 100.0))

                st.write(f"🐶 **Dog (Anjing):** {p_dog:.2f}%")
                st.progress(float(p_dog / 100.0))

                # Grad-CAM Attention Map
                st.write("---")
                st.markdown("**Visualisasi Atensi Grad-CAM (Fitur Kunci):**")
                if res["cam_mask"] is not None:
                    cam_mask = res["cam_mask"]
                    cam_img = Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(img_to_process.size, resample=Image.Resampling.BILINEAR)
                    cam_array = np.array(cam_img) / 255.0
                    heatmap = plt.cm.jet(cam_array)[:, :, :3]
                    overlay = (0.55 * np.array(img_to_process) / 255.0) + (0.45 * heatmap)
                    overlay = np.clip(overlay, 0, 1)

                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        st.image(cam_mask, caption="Heatmap Fitur Penting", use_container_width=True, clamp=True)
                    with cam_col2:
                        st.image(overlay, caption="Overlay Atensi pada Citra", use_container_width=True)
                    st.markdown('<div class="caption-text">Grad-CAM menyoroti area yang berkontribusi terhadap keputusan model (area merah/kuning memiliki pengaruh tertinggi).</div>', unsafe_allow_html=True)

                # Warning / Disclaimer
                st.warning("⚠️ **Catatan Penting:** Prediksi merupakan keluaran probabilistik model machine learning dan dapat keliru terutama pada gambar yang kabur, memiliki objek majemuk, atau sudut pemotretan ekstrem.")

    # -------------------------------------------------------------
    # SECTION 2: EVALUATION DASHBOARD & METRICS
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">2. 📊 Hasil Evaluasi Model pada Hold-Out Test Set (1.000 Citra)</div>', unsafe_allow_html=True)
    if metrics:
        test_m = metrics.get("final_holdout_test_metrics", {})
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Akurasi (Accuracy)", f"{test_m.get('accuracy', 0)*100:.2f}%")
        m_col2.metric("Presisi (Precision)", f"{test_m.get('precision', 0)*100:.2f}%")
        m_col3.metric("Recall (Sensitivity)", f"{test_m.get('recall', 0)*100:.2f}%")
        m_col4.metric("F1-Score", f"{test_m.get('f1_score', 0)*100:.2f}%")

        st.write("")
        st.subheader("Grafik Evaluasi Kuantitatif Kunci")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            cm_path = "visualizations/confusion_matrix.png"
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix pada Hold-Out Test Set (1.000 Gambar)", use_container_width=True)
        with v_col2:
            roc_path = "visualizations/roc_curve.png"
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Kurva ROC dan Nilai AUC (0.9987)", use_container_width=True)

        v_col3, v_col4 = st.columns(2)
        with v_col3:
            comp_path = "visualizations/model_comparison.png"
            if os.path.exists(comp_path):
                st.image(comp_path, caption="Perbandingan Metrik Baseline vs Transfer Learning", use_container_width=True)
        with v_col4:
            pred_path = "visualizations/prediction_examples.png"
            if os.path.exists(pred_path):
                st.image(pred_path, caption="Contoh Prediksi Benar dan Salah pada Hold-Out Test", use_container_width=True)

    # -------------------------------------------------------------
    # SECTION 3: METHODOLOGY & PROTOCOL
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">3. ℹ️ Metodologi & Protokol Anti Data Leakage</div>', unsafe_allow_html=True)
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("""
        #### Arsitektur Model:
        - **Custom CNN Baseline:** 4 blok Convolutional dengan Batch Normalization, ReLU, Max Pooling, Dropout, dan Global Average Pooling.
        - **Transfer Learning (MobileNetV2):** Menggunakan pretrained weights ImageNet-1K, classification head baru dengan Dropout, dan fine-tuning layer 14–18.
        """)
    with info_col2:
        st.markdown("""
        #### Protokol Anti Data Leakage:
        - Folder `validation/` (1.000 citra) diarsip murni sebagai **final hold-out test set** yang hanya dievaluasi satu kali di akhir.
        - Folder `train/` (2.000 citra) dibagi stratified menjadi 80% train internal (1.600 citra) dan 20% validation internal (400 citra).
        - Pemilihan model terbaik strictly mengacu pada metrik validation internal.
        """)


if __name__ == "__main__":
    main()
