import streamlit as st
import numpy as np
import cv2
import pywt
from scipy.linalg import svd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# -------------------- Helpers --------------------
def text_to_watermark(text, shape):
    """Convert text to binary watermark image with given shape"""
    wm = np.zeros(shape, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(wm, text, (10, shape[0]//2), font, 1, (255,), 2, cv2.LINE_AA)
    return wm

def dwt_svd_embed(img, watermark, alpha=0.1):
    """Embed watermark using DWT + SVD"""
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Apply SVD to LH subband
    U, S, Vt = svd(LH)
    wm_resized = cv2.resize(watermark, (LH.shape[1], LH.shape[0]))
    U_wm, S_wm, Vt_wm = svd(wm_resized.astype(np.float32))

    # Embed
    S_emb = S + alpha * S_wm
    LH_emb = np.dot(U, np.dot(np.diag(S_emb), Vt))

    # Reconstruct
    coeffs_emb = LL, (LH_emb, HL, HH)
    watermarked_img = pywt.idwt2(coeffs_emb, 'haar')
    return np.uint8(np.clip(watermarked_img, 0, 255)), (U, S, Vt)

def dwt_svd_extract(img, U, S, Vt, alpha=0.1, shape=(128, 128)):
    """Extract watermark using inverse embedding"""
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    U_new, S_new, Vt_new = svd(LH)
    S_wm_extracted = (S_new - S) / alpha
    wm_extracted = np.dot(U, np.dot(np.diag(S_wm_extracted), Vt))
    wm_resized = cv2.resize(wm_extracted, (shape[1], shape[0]))
    return np.uint8(np.clip(wm_resized, 0, 255))

# -------------------- Streamlit App --------------------
st.title("🖼️ DWT + SVD Image Watermarking")

st.sidebar.header("Options")
alpha = st.sidebar.slider("Embedding Strength (alpha)", 0.01, 0.5, 0.1)

uploaded_file = st.file_uploader("Upload Artwork", type=["jpg", "png", "jpeg"])
watermark_text = st.text_input("Enter your copyright text", "© MyArt")

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize for performance
    max_size = 512
    if img.shape[0] > max_size or img.shape[1] > max_size:
        img = cv2.resize(img, (max_size, max_size))

    st.image(img, caption="Original Image", use_column_width=True)

    # Generate watermark
    watermark = text_to_watermark(watermark_text, (128, 128))
    st.image(watermark, caption="Generated Watermark", use_column_width=True)

    if st.button("Embed Watermark"):
        with st.spinner("Embedding watermark..."):
            watermarked_img, svd_keys = dwt_svd_embed(img, watermark, alpha)
            st.image(watermarked_img, caption="Watermarked Image", use_column_width=True)

            # Store in session
            st.session_state["watermarked_img"] = watermarked_img
            st.session_state["svd_keys"] = svd_keys
            st.session_state["orig_img"] = img
            st.session_state["watermark"] = watermark

    if "watermarked_img" in st.session_state and st.button("Extract Watermark"):
        with st.spinner("Extracting watermark..."):
            wm_extracted = dwt_svd_extract(
                st.session_state["watermarked_img"],
                *st.session_state["svd_keys"],
                alpha=alpha,
                shape=st.session_state["watermark"].shape
            )
            st.image(wm_extracted, caption="Extracted Watermark", use_column_width=True)

            # Metrics
            psnr_val = psnr(st.session_state["orig_img"], st.session_state["watermarked_img"])
            ssim_val = ssim(st.session_state["orig_img"], st.session_state["watermarked_img"])

            accuracy = np.sum(
                st.session_state["watermark"] == (wm_extracted > 128).astype(np.uint8)*255
            ) / st.session_state["watermark"].size * 100

            st.write(f"🔹 **PSNR**: {psnr_val:.2f} dB")
            st.write(f"🔹 **SSIM**: {ssim_val:.4f}")
            st.write(f"🔹 **Watermark Extraction Accuracy**: {accuracy:.2f}%")
