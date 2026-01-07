import torch
import cv2
import numpy as np
import os
import mediapipe as mp
from torchvision import transforms
from PIL import Image

from app.core.config import settings
from app.schemas.prediction import PredictionResult
from models.model import DeePixBiS

# ============================================================
# CONFIGURATION & INITIALIZATION
# ============================================================

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Model Variables
model = None

# Transform untuk DeePixBiS
# CHANGE: Ditambahkan Normalize((0.5...), (0.5...))
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Initialize MediaPipe Face Detection (Load sekali saja di awal biar cepat)
mp_face_detection = mp.solutions.face_detection
# model_selection=1 (range jauh/selfie full body), 0 (jarak dekat webcam)
# min_detection_confidence=0.5
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def load_model():
    """
    Loads the DeePixBiS model weights.
    """
    global model
    try:
        model = DeePixBiS(pretrained=True).to(DEVICE)
        
        if os.path.exists(settings.MODEL_PATH):
            state_dict = torch.load(settings.MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✅ Model loaded successfully from {settings.MODEL_PATH}")
        else:
            print(f"⚠️ WARNING: Model file not found at {settings.MODEL_PATH}. Using random weights.")
            
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

# Auto-load on import
try:
    load_model()
except Exception:
    pass 

# ============================================================
# UTILS & PRE-PROCESSING (FACE CROP)
# ============================================================

def get_face_crop(image: np.ndarray) -> np.ndarray:
    """
    Mendeteksi wajah menggunakan MediaPipe dan melakukan cropping.
    Mengembalikan numpy array wajah (cropped). 
    Return None jika tidak ada wajah.
    """
    ih, iw, _ = image.shape
    
    # MediaPipe butuh RGB
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.detections:
        return None

    # Ambil wajah dengan confidence tertinggi (index 0 biasanya urut confidence)
    detection = results.detections[0] 
    bboxC = detection.location_data.relative_bounding_box
    
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                 int(bboxC.width * iw), int(bboxC.height * ih)

    # Tambahkan Padding 15% agar dagu/dahi tidak terpotong pas
    # DeePixBiS perlu sedikit konteks
    pad_h = int(h * 0.15)
    pad_w = int(w * 0.15)
    
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(iw - x, w + pad_w * 2)
    h = min(ih - y, h + pad_h * 2)

    # Validasi ukuran crop agar tidak 0
    if w <= 0 or h <= 0:
        return None

    cropped_face = image[y:y+h, x:x+w]
    return cropped_face

def read_image_from_bytes(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decoding failed.")
    return img

# ============================================================
# HELPER FUNCTIONS (OPTICS + STATISTICS)
# ============================================================

def glare_score(image: np.ndarray) -> float:
    # Menggunakan HSV Value channel untuk deteksi over-exposure (pantulan layar)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2] / 255.0
    bright = (v > 0.95).astype(np.uint8) # Threshold glare dinaikkan sedikit
    return float(bright.mean())

def patch_max_score(spoof_map: torch.Tensor) -> float:
    """
    Mengambil rata-rata patch tertinggi dari output map model.
    spoof_map shape: (1, H, W)
    """
    m = spoof_map.squeeze().detach().cpu().numpy()
    h, w = m.shape
    # Bagi image map menjadi 4 kuadran
    patches = [
        m[:h//2, :w//2], m[:h//2, w//2:],
        m[h//2:, :w//2], m[h//2:, w//2:],
    ]
    return float(max(p.mean() for p in patches))

def local_variance(image: np.ndarray) -> float:
    """
    Mendeteksi kekayaan tekstur.
    Wajah asli: Variance tinggi (pori-pori, rambut).
    Layar/Kertas: Variance rendah (flat/blur).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # Hitung variance per blok
    # Menggunakan kernel standard deviation filter bisa lebih cepat, tapi loop ini oke untuk size kecil
    vars_ = []
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            block = gray[i:i+16, j:j+16]
            vars_.append(block.var())
            
    return float(np.mean(vars_))

def color_correlation(image: np.ndarray) -> float:
    """
    Mendeteksi linearitas warna (Channel R vs G).
    Layar HP sering memiliki korelasi warna yang tidak wajar dibanding kulit alami.
    """
    b, g, r = cv2.split(image)
    
    # Downsample biar cepat hitung correlation
    r_small = cv2.resize(r, (64, 64)).flatten()
    g_small = cv2.resize(g, (64, 64)).flatten()

    if r_small.std() == 0 or g_small.std() == 0:
        return 1.0

    return float(np.corrcoef(r_small, g_small)[0, 1])

def frequency_spike_ratio(image: np.ndarray) -> float:
    """
    FFT untuk mendeteksi Moiré Pattern (Grid pixel layar).
    Sangat efektif pada crop wajah.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128)) # Resize 128 cukup untuk FFT

    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(f) + 1)

    h, w = mag.shape
    center_h, center_w = h // 2, w // 2
    
    # Masking DC Component (Frekuensi rendah di tengah)
    mag[center_h-5:center_h+5, center_w-5:center_w+5] = 0

    mean = mag.mean()
    std = mag.std()
    
    # Deteksi spike yang jauh diatas rata-rata (pola berulang)
    spikes = (mag > mean + 3.5 * std).astype(np.uint8)
    return float(spikes.mean())

# ============================================================
# MAIN PREDICT
# ============================================================

def predict(image: np.ndarray) -> PredictionResult:
    if model is None:
        raise RuntimeError("Model is not loaded.")

    # [STEP 1] CROP WAJAH
    face_crop = get_face_crop(image)

    if face_crop is None:
        # Jika tidak ada wajah, langsung return FAIL/Error
        return PredictionResult(
            decision="FAIL",
            mean_score=0.0,
            patch_max_score=0.0,
            local_variance=0.0,
            rgb_corr=0.0,
            freq_spike_ratio=0.0,
            glare_asym=0.0,
            evidence_count=1,
            evidence=["no_face_detected"],
            reason="Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah terlihat jelas."
        )

    # Gunakan face_crop untuk semua analisis selanjutnya
    target_img = face_crop

    # -----------------
    # DeepPixBiS Inference
    # -----------------
    rgb_crop = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    # Transform akan meresize crop ke 224x224 & Normalize
    tensor = transform(rgb_crop).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)
        # Handle output format (kadang tuple, kadang tensor direct)
        spoof_map = output[0] if isinstance(output, (tuple, list)) else output
        
        # Mean score: Rata-rata probabilitas pixel (0=spoof, 1=real)
        mean_score = float(torch.mean(spoof_map).item())
        max_patch = patch_max_score(spoof_map)

    # -----------------
    # Heuristic Checks (Pada Wajah Crop)
    # -----------------
    lv = local_variance(target_img)
    corr = color_correlation(target_img)
    spike_ratio = frequency_spike_ratio(target_img)

    h_f, w_f, _ = target_img.shape
    gl_left = glare_score(target_img[:, :w_f//2])
    gl_right = glare_score(target_img[:, w_f//2:])
    glare_asym = abs(gl_left - gl_right)

    # ============================================================
    # EVIDENCE COLLECTION & THRESHOLDING
    # ============================================================
    
    evidence = []
    
    # 1. DeepPixBiS Check
    # Nilai ini biasanya terbalik tergantung training label (0=Real atau 0=Spoof). 
    # Di repo Saiyam/GeorgePoly: Output mendekati 0 = Fake, 1 = Real.
    # Maka Patch Score Rendah = Spoof.
    
    # PERHATIAN: Sesuaikan logika if ini dengan output model kamu.
    # Jika model kamu outputnya: 1=Real, 0=Fake.
    # Maka "Spoof Patch" adalah patch yang nilainya KECIL.
    # Namun kode kamu sebelumnya: if max_patch >= 0.6 -> Strong Spoof. 
    # Ini berarti asumsinya Output Tinggi = Fake? 
    # SAYA IKUTI ASUMSI KODEMU SEBELUMNYA (High Score = Spoof). 
    # JIKA TERBALIK (Real > 0.5), SILAKAN BALIK LOGIKANYA.
    
    # Tuned Thresholds V3 (Aggressive for Fake Detection)
    # Analysis:
    # 53 Fakes are in range 0.45-0.60.
    # Only 3 Real images are in range 0.45-0.60.
    # Decision: Raise Strong threshold to 0.60 to catch those 53 fakes.
    
    if mean_score < 0.60: 
        evidence.append("strong_ai_spoof_detection")
    elif mean_score < 0.80: 
        evidence.append("weak_ai_spoof_detection")

    # Context-Aware Heuristics
    # If the AI score is low (< 0.80), we STRICTLY check for other anomalies.
    # If the AI score is high (> 0.80), we RELAX the checks to allow for environmental noise.
    
    is_confused = mean_score < 0.80  # AI is not confident it's real
    
    # 2. Frequency Check (Moiré)
    thr_spike = 0.005 if is_confused else 0.01
    if spike_ratio > thr_spike:
        evidence.append("screen_moire_pattern_detected")

    # 3. Color Check
    thr_corr = 0.99 if is_confused else 0.995
    if corr > thr_corr: 
        evidence.append("unnatural_color_correlation")

    # 4. Texture Check
    thr_lv = 50 if is_confused else 30 
    if lv < thr_lv: 
        evidence.append("flat_texture_detected")

    # 5. Glare Check
    thr_glare = 0.15 if is_confused else 0.25
    if glare_asym > thr_glare: 
        evidence.append("asymmetric_screen_glare")

    # ============================================================
    # FINAL DECISION LOGIC
    # ============================================================

    # Logika diperketat
    decision = "PASS"
    reason = "real_camera"

    if "no_face_detected" in evidence:
        decision = "FAIL"
        reason = "no_face"
    
    elif "strong_ai_spoof_detection" in evidence:
        decision = "FAIL"
        reason = "ai_model_rejected"
        
    elif len(evidence) >= 2:
        decision = "FAIL"
        reason = f"multiple_anomalies: {', '.join(evidence)}"
        
    elif evidence == ["weak_ai_spoof_detection"]:
        # Weak spoof signal ALONE is insufficient to fail (could be a low-quality real selfie)
        decision = "PASS"
        reason = "low_confidence_spoof_ignored"
        
    elif len(evidence) == 1:
        # Jika cuma 1 anomali (misal texture flat karena lighting gelap), 
        # tapi AI bilang aman (score > 0.65), kita loloskan.
        decision = "PASS" 
        reason = f"warning: {evidence[0]}"

    return PredictionResult(
        decision=decision,
        mean_score=round(mean_score, 4),
        patch_max_score=round(max_patch, 4),
        local_variance=round(lv, 2),
        rgb_corr=round(corr, 4),
        freq_spike_ratio=round(spike_ratio, 5),
        glare_asym=round(glare_asym, 4),
        evidence_count=len(evidence),
        evidence=evidence,
        reason=reason
    )