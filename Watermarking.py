# app.py
import streamlit as st
import cv2
import numpy as np
import json
from dwt_svd_watermark import encode, decode, _save_keys, _load_keys

st.set_page_config(page_title="DWT+SVD Watermarking", layout="wide")

st.title("🔐 DWT+SVD Image Watermarking")

tab1 = st.tabs(["🔏 Encode Watermark"])

# ------------------- ENCODE -------------------
with tab1:
    st.header("Embed Text Watermark")

    host_file = st.file_uploader("Upload host image", type=["jpg", "jpeg", "png"])
    wm_text = st.text_input("Watermark text", "© Your Name 2025")
    alpha = st.slider("Embedding strength (alpha)", 0.01, 0.1, 0.05, 0.01)
    wave = st.selectbox("Wavelet", ["haar", "db2", "db4"])

    if st.button("Encode Watermark", type="primary"):
        if host_file:
            # Load image
            file_bytes = np.asarray(bytearray(host_file.read()), dtype=np.uint8)
            host_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Encode
            watermarked, keys, metrics = encode(host_img, wm_text, alpha=alpha, wave=wave)

            # Convert BGR to RGB for display
            wm_rgb = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)

            st.subheader("Watermarked Image")
            st.image(wm_rgb, channels="RGB", use_column_width=True)
            st.write(f"**PSNR:** {metrics['psnr']:.2f} dB | **SSIM:** {metrics['ssim']:.4f}")

            # Download watermarked image
            _, buffer = cv2.imencode(".png", watermarked)
            st.download_button("Download Watermarked Image", buffer.tobytes(),
                               file_name="watermarked.png", mime="image/png")

            # Download keys.json
            keys_json = json.dumps(keys, indent=2).encode("utf-8")
            st.download_button("Download Keys JSON", keys_json,
                               file_name="keys.json", mime="application/json")
        else:
            st.warning("Please upload a host image.")

# ------------------- DECODE -------------------

    st.header("Extract Watermark")

    suspect_file = st.file_uploader("Upload suspect image", type=["jpg", "jpeg", "png"])
    keys_file = st.file_uploader("Upload keys.json", type=["json"])

    if st.button("Decode Watermark", type="primary"):
        if suspect_file and keys_file:
            # Load suspect image
            file_bytes = np.asarray(bytearray(suspect_file.read()), dtype=np.uint8)
            suspect_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Load keys
            keys = json.load(keys_file)

            # Decode
            extracted, metrics = decode(suspect_img, keys)
            st.subheader("Extracted Watermark")
            st.image(extracted, channels="GRAY", use_column_width=True)
            st.write(f"**Normalized Correlation (NC):** {metrics['nc']:.4f}")

            # Download extracted watermark
            _, buffer = cv2.imencode(".png", extracted)
            st.download_button("Download Extracted Watermark", buffer.tobytes(),
                               file_name="extracted.png", mime="image/png")
        else:
            st.warning("Please upload both suspect image and keys.json.")
