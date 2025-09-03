import io, math
import numpy as np
import streamlit as st
import cv2
import pywt
from scipy.linalg import svd

# Try to import skimage SSIM; fall back to a pure-OpenCV implementation if missing
try:
    from skimage.metrics import structural_similarity as ssim_sk
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# ========================= Utility functions =========================
def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Ensure single-channel uint8 grayscale."""
    if img is None:
        return None
    if img.ndim == 2:
        return np.uint8(np.clip(img, 0, 255))
    return np.uint8(np.clip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255))


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_gray_cv(a: np.ndarray, b: np.ndarray) -> float:
    # SSIM for grayscale using OpenCV (fallback)
    if a.shape != b.shape:
        raise ValueError("SSIM inputs must have the same shape.")
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    ksize = (11, 11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(a, ksize, sigma)
    mu2 = cv2.GaussianBlur(b, ksize, sigma)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(a * a, ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, ksize, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, ksize, sigma) - mu1_mu2
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean())  # <- fixed: no extra ')'


def ssim_metric(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_SKIMAGE:
        return float(ssim_sk(a, b, data_range=255))
    return ssim_gray_cv(a, b)


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    am, bm = a.mean(), b.mean()
    num = np.sum((a - am) * (b - bm))
    den = np.sqrt(np.sum((a - am) ** 2) * np.sum((b - bm) ** 2)) + 1e-12
    return float(num / den)


def bit_error_rate(a: np.ndarray, b: np.ndarray, thresh: int = 128) -> float:
    a_bits = (a >= thresh).astype(np.uint8)
    b_bits = (b >= thresh).astype(np.uint8)
    return float(np.mean(a_bits ^ b_bits))


def make_even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - (h % 2), : w - (w % 2)]


def crop_like(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    return src[:h, :w]


def to_square_canvas(img: np.ndarray, size=(1200, 1200), pad_value: int = 0) -> np.ndarray:
    """Pad image to a square canvas while preserving aspect ratio."""
    h, w = img.shape[:2]
    th, tw = size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left
    canvas = np.full((th, tw), pad_value, dtype=resized.dtype)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def _layout(H, W, block=8, border=50):
    inner_h = (H - 2 * border) // block
    inner_w = (W - 2 * border) // block
    total = inner_h * inner_w
    return inner_h, inner_w, total


# ========================= Attacks =========================
def apply_attack(img, attack_type, ksize=3, sigma=1.0, jpeg_q=85, sp_amount=0.02, gauss_std=5.0):
    k = int(max(1, ksize))
    if k % 2 == 0:
        k += 1
    if attack_type == "None":
        return img.copy()
    if attack_type == "Gaussian blur":
        return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma))
    if attack_type == "Median filter":
        return cv2.medianBlur(img, k)
    if attack_type == "Average filter":
        return cv2.blur(img, (k, k))
    if attack_type == "JPEG compression":
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
        if not ok:
            return img.copy()
        return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if attack_type == "Salt & pepper noise":
        out = img.copy()
        p = float(sp_amount)
        mask = np.random.random(img.shape)
        out[mask < p / 2] = 0
        out[mask > 1 - p / 2] = 255
        return out
    if attack_type == "Gaussian noise":
        noise = np.random.normal(0, float(gauss_std), img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.uint8(np.clip(out, 0, 255))
    return img.copy()


# ========================= Method A: DWT + DCT (HL) =========================
def mA_embed(host_gray, wm_gray, alpha=0.05):
    LL, (LH, HL, HH) = pywt.dwt2(host_gray, "haar")
    wm_resized = cv2.resize(wm_gray, (HL.shape[1], HL.shape[0]), interpolation=cv2.INTER_AREA)
    HL_dct = cv2.dct(np.float32(HL))
    wm_dct = cv2.dct(np.float32(wm_resized))
    HL_dct_wm = HL_dct + alpha * wm_dct
    HL_wm = cv2.idct(HL_dct_wm)
    watermarked = pywt.idwt2((LL, (LH, HL_wm, HH)), "haar")
    watermarked = crop_like(np.uint8(np.clip(watermarked, 0, 255)), host_gray)
    return watermarked, HL_dct, wm_resized


def mA_extract(attacked_img, HL_dct_ref, alpha, wm_shape):
    _, (LH2, HL2, HH2) = pywt.dwt2(attacked_img, "haar")
    HL2_dct = cv2.dct(np.float32(HL2))
    wm_ex_dct = (HL2_dct - HL_dct_ref) / alpha
    wm_ex = cv2.idct(wm_ex_dct)
    wm_ex = cv2.resize(wm_ex, (wm_shape[1], wm_shape[0]), interpolation=cv2.INTER_AREA)
    return np.uint8(np.clip(wm_ex, 0, 255))


# ========================= Method B: DWT + SVD (select subband) =========================
def mB_embed(host_gray, wm_gray, alpha=0.05, subband_type=""):
    LL, (LH, HL, HH) = pywt.dwt2(host_gray, "haar")

    if subband_type == "LL":
        subband = LL
    elif subband_type == "LH":
        subband = LH
    elif subband_type == "HL":
        subband = HL
    elif subband_type == "HH":
        subband = HH
    else:
        raise ValueError("Invalid subband_type. Choose 'LL','LH','HL','HH'.")

    wm_resized = cv2.resize(wm_gray, (subband.shape[1], subband.shape[0]), interpolation=cv2.INTER_AREA)

    U, S, Vt = np.linalg.svd(subband, full_matrices=False)
    Uw, Sw, Vtw = np.linalg.svd(wm_resized, full_matrices=False)

    S_marked = S + alpha * Sw
    subband_wm = np.dot(U, np.dot(np.diag(S_marked), Vt))

    if subband_type == "LL":
        coeffs_wm = (subband_wm, (LH, HL, HH))
    elif subband_type == "LH":
        coeffs_wm = (LL, (subband_wm, HL, HH))
    elif subband_type == "HL":
        coeffs_wm = (LL, (LH, subband_wm, HH))
    else:  # HH
        coeffs_wm = (LL, (LH, HL, subband_wm))

    watermarked = pywt.idwt2(coeffs_wm, "haar")
    watermarked = crop_like(np.uint8(np.clip(watermarked, 0, 255)), host_gray)

    return watermarked, (U, Vt, S, subband_type), (Uw, Vtw), wm_resized.shape


def mB_extract(attacked_img, host_keys, wm_keys, wm_shape, alpha=0.05):
    (U, Vt, S, subband_type) = host_keys
    (Uw, Vtw) = wm_keys

    LL2, (LH2, HL2, HH2) = pywt.dwt2(attacked_img, "haar")
    if subband_type == "LL":
        subband2 = LL2
    elif subband_type == "LH":
        subband2 = LH2
    elif subband_type == "HL":
        subband2 = HL2
    else:
        subband2 = HH2

    Ua, Sa, Vta = np.linalg.svd(subband2, full_matrices=False)
    Sw_ex = (Sa - S) / alpha
    wm_ex = np.dot(Uw, np.dot(np.diag(Sw_ex), Vtw))
    wm_ex = (wm_ex > 128).astype(np.uint8) * 255
    return wm_ex


# ========================= Method C: DWT (HL additive) =========================
def mC_embed(host_gray, wm_gray, alpha=0.05, wm_size=(64, 64)):
    LL, (LH, HL, HH) = pywt.dwt2(host_gray, "haar")
    wm_small = cv2.resize(wm_gray, wm_size, interpolation=cv2.INTER_AREA).astype(np.float32)

    HLf = HL.astype(np.float32)
    Hh, Hw = HLf.shape
    wh, ww = wm_small.shape
    pad = np.zeros_like(HLf, dtype=np.float32)
    y0 = (Hh - wh) // 2
    x0 = (Hw - ww) // 2
    pad[y0:y0+wh, x0:x0+ww] = wm_small

    HL_wm = HLf + alpha * pad

    watermarked = pywt.idwt2((LL, (LH, HL_wm, HH)), "haar")
    watermarked = crop_like(np.uint8(np.clip(watermarked, 0, 255)), host_gray)
    return watermarked, (HLf, (y0, x0, wh, ww)), wm_small


def mC_extract(attacked_img, HLref_and_loc, alpha, wm_shape):
    HL_ref, (y0, x0, wh, ww) = HLref_and_loc
    _, (LH2, HL2, HH2) = pywt.dwt2(attacked_img, "haar")
    HL2f = HL2.astype(np.float32)
    pad_est = (HL2f - HL_ref) / alpha
    wm_ex = pad_est[y0:y0+wh, x0:x0+ww]
    wm_ex = cv2.resize(wm_ex, (wm_shape[1], wm_shape[0]), interpolation=cv2.INTER_AREA)
    return np.uint8(np.clip(wm_ex, 0, 255))


# ========================= Method D: DCT block (random blocks + quantization) =========================
def dct_block_embed(host_gray, wm_gray, key=50, bs=8, b_cut=50, wm_size=(128, 128),
                    indx=3, indy=2, fact=8.0, canvas=(1200, 1200)):
    rng = np.random.default_rng(seed=int(key))
    img_canvas = to_square_canvas(host_gray, canvas)
    wm_small = cv2.resize(wm_gray, wm_size, interpolation=cv2.INTER_AREA)
    H, W = img_canvas.shape
    inner_h, inner_w, total_blocks = _layout(H, W, bs, b_cut)
    bits_needed = wm_size[0] * wm_size[1]
    if total_blocks < bits_needed:
        raise ValueError("Not enough blocks for watermark (DCT).")
    wm_bits = (wm_small.reshape(-1) >= 127).astype(np.uint8)
    used = set()
    idx = 0
    imf = img_canvas.astype(np.float32)
    while idx < bits_needed:
        x = int(rng.integers(0, total_blocks))
        if x in used:
            continue
        used.add(x)
        bi = (x // inner_w) * bs + b_cut
        bj = (x % inner_w) * bs + b_cut
        block = imf[bi : bi + bs, bj : bj + bs]
        dctb = cv2.dct(block)
        coef = dctb[indx, indy] / fact
        bit = int(wm_bits[idx])
        q = math.floor(coef) + (0.25 if bit == 0 else 0.75)
        dctb[indx, indy] = q * fact
        imf[bi : bi + bs, bj : bj + bs] = cv2.idct(dctb)
        idx += 1
    wmed = np.uint8(np.clip(imf, 0, 255))
    state = dict(
        img_canvas=img_canvas,
        wm_small=wm_small,
        key=int(key),
        bs=bs,
        b_cut=b_cut,
        wm_size=wm_size,
        indx=indx,
        indy=indy,
        fact=float(fact),
        canvas=canvas,
    )
    return wmed, state, wm_small


def dct_block_extract(attacked_img, state):
    rng2 = np.random.default_rng(seed=int(state["key"]))
    img_canvas = to_square_canvas(attacked_img, state["canvas"])
    H, W = img_canvas.shape
    bs = state["bs"]
    b_cut = state["b_cut"]
    wm_size = state["wm_size"]
    indx = state["indx"]
    indy = state["indy"]
    fact = state["fact"]
    inner_h, inner_w, total_blocks = _layout(H, W, bs, b_cut)
    bits_needed = wm_size[0] * wm_size[1]
    if total_blocks < bits_needed:
        raise ValueError("Not enough blocks to extract (DCT).")
    wm_bits = np.zeros(bits_needed, dtype=np.uint8)
    used = set()
    idx = 0
    imf = img_canvas.astype(np.float32)
    while idx < bits_needed:
        x = int(rng2.integers(0, total_blocks))
        if x in used:
            continue
        used.add(x)
        bi = (x // inner_w) * bs + b_cut
        bj = (x % inner_w) * bs + b_cut
        block = imf[bi : bi + bs, bj : bj + bs]
        dctb = cv2.dct(block)
        coef = dctb[indx, indy] / fact
        frac = coef - math.floor(coef)
        bit = 1 if frac >= 0.5 else 0
        wm_bits[idx] = bit
        idx += 1
    wm_rec = (wm_bits.reshape(wm_size[0], wm_size[1]) * 255).astype(np.uint8)
    return wm_rec, img_canvas


# ========================= Streamlit UI =========================
st.set_page_config(page_title="Watermarking Comparison", layout="wide")
st.title("Watermarking Comparison")
st.caption("Methods: A) DWT+DCT(HL), B) DWT+SVD, C) DWT additive(HL 64×64), D) DCT block (random blocks + quantization)")

# ------- consistent image size & layout helpers -------
IMG_W = 320  # medium size everywhere

def vspace(px=12):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)

def show_center(img, cap):
    # three columns: center content in the middle one
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(img, caption=cap, clamp=True, width=IMG_W)

def show_pair(imgL, capL, imgR, capR):
    # fixed two equal columns for stability
    c1, c2 = st.columns(2, gap="large")
    c1.image(imgL, caption=capL, clamp=True, width=IMG_W)
    c2.image(imgR, caption=capR, clamp=True, width=IMG_W)


with st.sidebar:
    st.header("Inputs")
    up_host = st.file_uploader("Upload Host Image", type=["png", "jpg", "jpeg"])
    up_wm = st.file_uploader("Upload Watermark Image", type=["png", "jpg", "jpeg"])

    st.header("Embedding Strength (A/B/C)")
    alpha_A = st.slider("alpha A (DWT+DCT HL)", 0.01, 0.30, 0.05, 0.01)
    alpha_B = st.slider("alpha B (DWT+SVD)", 0.01, 0.50, 0.10, 0.01)
    alpha_C = st.slider("alpha C (DWT additive HL)", 0.01, 0.30, 0.05, 0.01)

    st.header("Method D (DCT block) Parameters")
    key_D  = st.number_input("D key (seed)", min_value=0, max_value=100000, value=50, step=1)
    indx_D = st.slider("D indx (row index in DCT block)", 1, 7, 3, 1)
    indy_D = st.slider("D indy (col index in DCT block)", 1, 7, 2, 1)
    fact_D = st.slider("D fact (quantization step)", 2.0, 40.0, 30.0, 0.5)
    bcut_D = st.slider("D b_cut (border offset)", 0, 200, 50, 5)
    wm_w_D = st.slider("D WM width", 32, 256, 64, 16)
    wm_h_D = st.slider("D WM height", 32, 256, 64, 16)

    st.header("Attack")
    attack = st.selectbox(
        "Attack type",
        ["None", "Gaussian blur", "Median filter", "Average filter", "JPEG compression", "Salt & pepper noise", "Gaussian noise"],
        index=0,
    )
    ksize = st.slider("kernel (for blur/median/average)", 1, 21, 3, 2)
    sigma = st.slider("sigma (Gaussian blur)", 0.1, 5.0, 1.0, 0.1)
    jpeg_q = st.slider("JPEG quality", 10, 100, 85, 1)
    gauss_std = st.slider("Noise σ (Gaussian noise)", 1.0, 30.0, 5.0, 0.5)

    run = st.button("Run all methods")

# Load images
def _read_streamlit_file(file):
    bytes_data = file.read()
    arr = np.frombuffer(bytes_data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return ensure_gray(bgr)

if run:
    if up_host is None or up_wm is None:
        st.warning("Please upload both Host and Watermark images.")
        st.stop()

    host = _read_streamlit_file(up_host)
    wm = _read_streamlit_file(up_wm)

    # ---- Originals row (stable two columns) ----
    st.markdown("#### Originals")
    show_pair(host, "original host image", wm, "original watermark image")
    st.divider()

    # A/B/C host processing: limit to 512 and ensure even dims for DWT
    max_sz = 512
    host_small = cv2.resize(host, (max_sz, max_sz), interpolation=cv2.INTER_AREA) if (host.shape[0] > max_sz or host.shape[1] > max_sz) else host.copy()
    host_small = make_even(host_small)

    # -------------------- Method A --------------------
    A = st.container()
    with A:
        A_img, A_ref, A_wm_res = mA_embed(host_small, wm, alpha=alpha_A)
        A_att = apply_attack(A_img, attack, ksize=ksize, sigma=sigma, jpeg_q=jpeg_q, gauss_std=gauss_std)
        A_att = crop_like(A_att, host_small)
        A_ex_orig = mA_extract(A_img, A_ref, alpha=alpha_A, wm_shape=A_wm_res.shape)
        A_ex_att  = mA_extract(A_att, A_ref, alpha=alpha_A, wm_shape=A_wm_res.shape)

        A_host_psnr = compute_psnr(host_small, A_img)
        A_host_ssim = ssim_metric(host_small, A_img)
        A_att_psnr  = compute_psnr(host_small, A_att)
        A_att_ssim  = ssim_metric(host_small, A_att)
        A_wm_psnr   = compute_psnr(A_wm_res, A_ex_att)
        A_wm_ssim   = ssim_metric(A_wm_res, A_ex_att)
        A_nc        = normalized_correlation(A_wm_res, A_ex_att)
        A_ber       = bit_error_rate(A_wm_res, A_ex_att)

        st.subheader("Method A")
        show_center(A_img, "watermarked image"); vspace()
        show_pair(host_small, "extract original host image", A_ex_orig, "extract original watermark image"); vspace()
        show_center(A_att, "attacked image"); vspace()
        show_pair(A_att, "extract attacked host image", A_ex_att, "extract attacked watermark image"); vspace()
       
        st.write(
            f"- Host PSNR/SSIM: **{A_host_psnr:.2f} dB**, **{A_host_ssim:.4f}** | After attack: **{A_att_psnr:.2f} dB**, **{A_att_ssim:.4f}**\n"
            f"- WM PSNR/SSIM/NC/BER: **{A_wm_psnr:.2f} dB**, **{A_wm_ssim:.4f}**, **NC {A_nc:.4f}**, **BER {A_ber:.4f}**"
        )
        st.divider()

    # -------------------- Method B (LL) --------------------
    B = st.container()
    with B:
        B_img, B_keys, B_wm_keys, B_wm_shape = mB_embed(host_small, wm, alpha=alpha_B, subband_type="LH")
        B_att = apply_attack(B_img, attack, ksize=ksize, sigma=sigma, jpeg_q=jpeg_q, gauss_std=gauss_std)
        B_att = crop_like(B_att, host_small)
        B_ex_orig = mB_extract(B_img, B_keys, B_wm_keys, B_wm_shape, alpha=alpha_B)
        B_ex_att  = mB_extract(B_att, B_keys, B_wm_keys, B_wm_shape, alpha=alpha_B)
        B_wm_res  = cv2.resize(wm, (B_wm_shape[1], B_wm_shape[0]), interpolation=cv2.INTER_AREA)

        B_host_psnr = compute_psnr(host_small, B_img)
        B_host_ssim = ssim_metric(host_small, B_img)
        B_att_psnr  = compute_psnr(host_small, B_att)
        B_att_ssim  = ssim_metric(host_small, B_att)
        B_wm_psnr   = compute_psnr(B_wm_res, B_ex_att)
        B_wm_ssim   = ssim_metric(B_wm_res, B_ex_att)
        B_nc        = normalized_correlation(B_wm_res, B_ex_att)
        B_ber       = bit_error_rate(B_wm_res, B_ex_att)

        st.subheader("Method B")
        show_center(B_img, "watermarked image"); vspace()
        show_pair(host_small, "extract original host image", B_ex_orig, "extract original watermark image"); vspace()
        show_center(B_att, "attacked image"); vspace()
        show_pair(B_att, "extract attacked host image", B_ex_att, "extract attacked watermark image"); vspace()
        
        st.write(
            f"- Host PSNR/SSIM: **{B_host_psnr:.2f} dB**, **{B_host_ssim:.4f}** | "
            f"After attack: **{B_att_psnr:.2f} dB**, **{B_att_ssim:.4f}**\n"
            f"- WM PSNR/SSIM/NC/BER: **{B_wm_psnr:.2f} dB**, **{B_wm_ssim:.4f}**, "
            f"**NC {B_nc:.4f}**, **BER {B_ber:.4f}**"
        )
        st.divider()

    # -------------------- Method C (HL additive, 64x64) --------------------
    C = st.container()
    with C:
        C_img, C_ref_and_loc, C_wm_res = mC_embed(host_small, wm, alpha=alpha_C, wm_size=(64,64))
        C_att = apply_attack(C_img, attack, ksize=ksize, sigma=sigma, jpeg_q=jpeg_q, gauss_std=gauss_std)
        C_att = crop_like(C_att, host_small)
        C_ex_orig = mC_extract(C_img, C_ref_and_loc, alpha=alpha_C, wm_shape=C_wm_res.shape)
        C_ex_att  = mC_extract(C_att, C_ref_and_loc, alpha=alpha_C, wm_shape=C_wm_res.shape)

        C_host_psnr = compute_psnr(host_small, C_img)
        C_host_ssim = ssim_metric(host_small, C_img)
        C_att_psnr  = compute_psnr(host_small, C_att)
        C_att_ssim  = ssim_metric(host_small, C_att)
        C_wm_psnr   = compute_psnr(C_wm_res, C_ex_att)
        C_wm_ssim   = ssim_metric(C_wm_res, C_ex_att)
        C_nc        = normalized_correlation(C_wm_res, C_ex_att)
        C_ber       = bit_error_rate(C_wm_res, C_ex_att)

        st.subheader("Method C")
        show_center(C_img, "watermarked image"); vspace()
        show_pair(host_small, "extract original host image", C_ex_orig, "extract original watermark image"); vspace()
        show_center(C_att, "attacked image"); vspace()
        show_pair(C_att, "extract attacked host image", C_ex_att, "extract attacked watermark image"); vspace()
        
        st.write(
            f"- Host PSNR/SSIM: **{C_host_psnr:.2f} dB**, **{C_host_ssim:.4f}** | After attack: **{C_att_psnr:.2f} dB**, **{C_att_ssim:.4f}**\n"
            f"- WM PSNR/SSIM/NC/BER: **{C_wm_psnr:.2f} dB**, **{C_wm_ssim:.4f}**, **NC {C_nc:.4f}**, **BER {C_ber:.4f}**"
        )
        st.divider()

    # -------------------- Method D (DCT block on canvas) --------------------
    D = st.container()
    with D:
        D_img, D_state, D_wm_res = dct_block_embed(
            host,
            wm,
            key=key_D,
            bs=8,
            b_cut=bcut_D,
            wm_size=(wm_h_D, wm_w_D),  # (H, W)
            indx=indx_D,
            indy=indy_D,
            fact=fact_D,
            canvas=(1200, 1200),
        )

        D_att = apply_attack(D_img, attack, ksize=ksize, sigma=sigma, jpeg_q=jpeg_q, gauss_std=gauss_std)
        D_ex_att, _ = dct_block_extract(D_att, D_state)
        D_ex_orig, _ = dct_block_extract(D_img, D_state)

        D_cover_canvas = D_state["img_canvas"]
        D_att_canvas   = to_square_canvas(D_att, D_state["canvas"])

        D_host_psnr = compute_psnr(D_cover_canvas, D_img)
        D_host_ssim = ssim_metric(D_cover_canvas, D_img)
        D_att_psnr  = compute_psnr(D_cover_canvas, D_att_canvas)
        D_att_ssim  = ssim_metric(D_cover_canvas, D_att_canvas)
        D_wm_psnr   = compute_psnr(D_wm_res, D_ex_att)
        D_wm_ssim   = ssim_metric(D_wm_res, D_ex_att)
        D_nc        = normalized_correlation(D_wm_res, D_ex_att)
        D_ber       = bit_error_rate(D_wm_res, D_ex_att)

        st.subheader("Method D")
        show_center(D_img, "watermarked image"); vspace()
        show_pair(D_cover_canvas, "extract original host image", D_ex_orig, "extract original watermark image"); vspace()
        show_center(D_att, "attacked image"); vspace()
        show_pair(D_att, "extract attacked host image", D_ex_att, "extract attacked watermark image"); vspace()
        
        st.write(
            f"- Host (canvas) PSNR/SSIM: **{D_host_psnr:.2f} dB**, **{D_host_ssim:.4f}** | After attack: **{D_att_psnr:.2f} dB**, **{D_att_ssim:.4f}**\n"
            f"- WM PSNR/SSIM/NC/BER: **{D_wm_psnr:.2f} dB**, **{D_wm_ssim:.4f}**, **NC {D_nc:.4f}**, **BER {D_ber:.4f}**"
        )


