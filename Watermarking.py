import streamlit as st
import numpy as np
import cv2
import pywt
from scipy.linalg import svd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# -------------------- Helpers --------------------
def text_to_watermark(text, shape):
    wm = np.zeros(shape, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(wm, text, (10, shape[0]//2), font, 1, (255,), 2, cv2.LINE_AA)
    return wm

def dwt_svd_embed(img, watermark, alpha=0.1):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    U, S, Vt = svd(LH)
    wm_resized = cv2.resize(watermark, (LH.shape[1], LH.shape[0]))
    U_wm, S_wm, Vt_wm = svd(wm_resized.astype(np.float32))

    S_emb = S + alpha * S_wm[:len(S)]
    k = min(len(S_emb), U.shape[0], Vt.shape[0])
    S_emb = S_emb[:k]
    U = U[:, :k]
    Vt = Vt[:k, :]

    LH_emb = U @ np.diag(S_emb) @ Vt
    coeffs_emb = LL, (LH_emb, HL, HH)
    watermarked_img = pywt.idwt2(coeffs_emb, 'haar')

    return np.uint8(np.clip(watermarked_img, 0, 255)), (U, S, Vt)

def dwt_svd_extract(img, U, S, Vt, alpha=0.1, shape=(128, 128)):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    U_new, S_new, Vt_new = svd(LH)
    S_wm_extracted = (S_new - S) / alpha
    wm_extracted = np.dot(U, np.dot(np.diag(S_wm_extracted), Vt))
    wm_resized = cv2.resize(wm_extracted, (shape[1], shape[0]))

    # Normalize for visibility
    wm_resized = wm_resized - wm_resized.min()
    wm_resized = wm_resized / (wm_resized.max() + 1e-6)
    wm_resized = (wm_resized * 255).astype(np.uint8)

    return wm_resized

# -------------------- Streamlit App --------------------
st.title("🎨 Artist Watermark Protection (DWT + SVD)")

st.sidebar.header("Options")
alpha = st.sidebar.slider("Embedding Strength (alpha)", 0.01, 0.5, 0.1)

tab1, tab2 = st.tabs(["🖼️ Embed Watermark", "🔎 Verify Artwork"])

# ------------- TAB 1: EMBED -----------------
with tab1:
    uploaded_file = st.file_uploader("Upload your Artwork", type=["jpg", "png", "jpeg"], key="embed")
    watermark_text = st.text_input("Enter your copyright text", "© MyArt")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        max_size = 512
        if img.shape[0] > max_size or img.shape[1] > max_size:
            img = cv2.resize(img, (max_size, max_size))

        st.image(img, caption="Original Image", use_column_width=True)

        watermark = text_to_watermark(watermark_text, (128, 128))
        st.image(watermark, caption="Generated Watermark", use_column_width=True)

        if st.button("Embed Watermark"):
            with st.spinner("Embedding watermark..."):
                watermarked_img, svd_keys = dwt_svd_embed(img, watermark, alpha)
                st.image(watermarked_img, caption="Watermarked Image", use_column_width=True)

                # Save session
                st.session_state["svd_keys"] = svd_keys
                st.session_state["watermark"] = watermark
                st.session_state["orig_img"] = img

                # Download button
                _, buffer = cv2.imencode(".png", watermarked_img)
                st.download_button("⬇️ Download Watermarked Artwork", buffer.tobytes(), "watermarked.png", "image/png")

# ------------- TAB 2: VERIFY -----------------
with tab2:
    verify_file = st.file_uploader("Upload Artwork to Verify", type=["jpg", "png", "jpeg"], key="verify")

    if verify_file is not None and "svd_keys" in st.session_state:
        file_bytes = np.asarray(bytearray(verify_file.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if test_img.shape != st.session_state["orig_img"].shape:
            test_img = cv2.resize(test_img, (st.session_state["orig_img"].shape[1],
                                             st.session_state["orig_img"].shape[0]))

        st.image(test_img, caption="Artwork to Verify", use_column_width=True)

        if st.button("Extract Watermark from Uploaded Artwork"):
            with st.spinner("Extracting watermark..."):
                wm_extracted = dwt_svd_extract(
                    test_img,
                    *st.session_state["svd_keys"],
                    alpha=alpha,
                    shape=st.session_state["watermark"].shape
                )
                st.image(wm_extracted, caption="Extracted Watermark", use_column_width=True)

                # Metrics
                psnr_val = psnr(st.session_state["orig_img"], test_img)
                ssim_val = ssim(st.session_state["orig_img"], test_img)
                accuracy = np.sum(
                    st.session_state["watermark"] == (wm_extracted > 128).astype(np.uint8)*255
                ) / st.session_state["watermark"].size * 100

                st.write(f"🔹 **PSNR**: {psnr_val:.2f} dB")
                st.write(f"🔹 **SSIM**: {ssim_val:.4f}")
                st.write(f"🔹 **Watermark Match Accuracy**: {accuracy:.2f}%")
