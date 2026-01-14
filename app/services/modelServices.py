# ============================================================
# IMPORTS
# ============================================================
import torch
import cv2
import numpy as np
import os
import mediapipe as mp
import torch.nn.functional as F
from collections import OrderedDict
import open_clip 
from PIL import Image

from app.core.config import settings
from app.schemas.prediction import PredictionResult
from models.minifasnet import MiniFASNetV2

# ============================================================
# CONFIGURATION & INITIALIZATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Models
model_mini = None      # Layer 2: MiniFASNetV2
model_clip = None      # Layer 1: OpenCLIP
transform_clip = None 
tokenizer_clip = None

# MediaPipe Face Detector
# MediaPipe Face Detector
face_detector = None

def load_model():
    global model_mini, model_clip, transform_clip, tokenizer_clip, face_detector
    
    # 1. Load OpenCLIP (Layer 1)
    try:
        print("Loading Layer 1: OpenCLIP (ViT-B-32)...")
        model_clip, _, transform_clip = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        model_clip = model_clip.to(DEVICE)
        tokenizer_clip = open_clip.get_tokenizer('ViT-B-32')
        print("✅ OpenCLIP loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load OpenCLIP: {e}")
        return False

    # 2. Load MiniFASNetV2 (Layer 2)
    try:
        print("Loading Layer 2: MiniFASNetV2...")
        model_mini = MiniFASNetV2(conv6_kernel=(5, 5)).to(DEVICE)
        state_dict = torch.load(settings.MODEL_PATH, map_location=DEVICE, weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model_mini.load_state_dict(new_state_dict)
        model_mini.eval()
        print("✅ MiniFASNetV2 loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load MiniFASNetV2: {e}")
        return False

    # 3. Load MediaPipe Face Detector
    try:
        if face_detector is None:
            print("Loading Layer 0: MediaPipe Face Detector...")
            face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
            print("✅ MediaPipe Face Detector loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Face Detector: {e}")
        return False
        
    return True

# Auto-load
try:
    load_model()
except Exception:
    pass

# ============================================================
# HELPER FUNCTIONS (Crop, Read)
# ============================================================
def read_image_from_bytes(data: bytes) -> np.ndarray:
    try:
        from PIL import Image, ImageOps
        import io
        image_pil = Image.open(io.BytesIO(data))
        image_pil = ImageOps.exif_transpose(image_pil)
        image_np = np.array(image_pil)
        if image_np.shape[2] == 4:
             # RGBA -> BGR
             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:
             # RGB -> BGR
             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_np
    except Exception as e:
        print(f"EXIF Handle Error: {e}")
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

def get_face_crop(image: np.ndarray, scale: float = 2.7):
    if image is None: return None
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)
    if not results.detections: return None
    
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    x, y, wd, ht = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
    
    # Scale logic
    box_w, box_h = wd, ht
    scale = min((h - 1) / box_h, min((w - 1) / box_w, scale))
    new_width, new_height = int(box_w * scale), int(box_h * scale)
    center_x, center_y = x + box_w / 2, y + box_h / 2
    
    x1 = int(center_x - new_width / 2)
    y1 = int(center_y - new_height / 2)
    x2 = int(center_x + new_width / 2)
    y2 = int(center_y + new_height / 2)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    face_crop = image[y1:y2, x1:x2]
    return face_crop if face_crop.size > 0 else None

# ============================================================
# INFERENCE LOGIC (Layer 1 & Layer 2)
# ============================================================

# ============================================================
# LAYER 0: FREQUENCY DOMAIN ANALYSIS (FFT)
# ============================================================
def analyze_frequency_domain(face_crop: np.ndarray) -> dict:
    """
    Analyzes the image in frequency domain to detect periodic noise (Moire patterns).
    Returns: { "is_anomaly": bool, "score": float, "reason": str }
    """
    try:
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 2. FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        
        # 3. Calculate Energy Concentration (Classic blur/moire metric)
        # Periodic patterns (moire) create distinct spikes in high frequencies.
        # Natural images have energy concentrated in center (low freq).
        
        # Mask center (Low Freq)
        cx, cy = w // 2, h // 2
        radius = 5  # Blocking DC component
        mask = np.ones((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), radius, 0, -1)
        
        # High Frequency Magnitude
        hf_magnitude = magnitude_spectrum * mask
        hf_mean = np.mean(hf_magnitude)
        hf_max = np.max(hf_magnitude)
        
        # Spike Ratio: If max spike is way higher than mean background noise -> Artificial Pattern
        spike_ratio = hf_max / (hf_mean + 1e-6)
        
        # Threshold: Tuned conservatively. 
        # Moire patterns often cause spike_ratio > 3.5 or 4.0
        # Normal images usually < 3.0
        threshold = 4.0 
        
        is_anomaly = spike_ratio > threshold
        return {"is_anomaly": is_anomaly, "score": float(spike_ratio), "reason": "moire_pattern_detected" if is_anomaly else "normal_freq"}
        
    except Exception as e:
        print(f"FFT Error: {e}")
        return {"is_anomaly": False, "score": 0.0, "reason": "fft_error"}

# ============================================================
# INFERENCE LOGIC (Layer 1, 2, 3)
# ============================================================

def infer_clip_layer1(face_crop: np.ndarray) -> dict:
    """
    Layer 1: OpenCLIP Zero-Shot
    """
    if model_clip is None: return {"is_real": False, "score": 0.0, "reason": "Model Not Loaded"}
    
    # Prepare Image (PIL RGB)
    img_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    img_tensor = transform_clip(img_pil).unsqueeze(0).to(DEVICE)
    
    # Prepare Prompts (Global Expanded List)
    # STRICTER OPTIMIZATION: Removed "selfie" to prevent screen-selfies from passing.
    # We rely on specific NEGATIVE prompts to catch spoofs.
    prompts = [
        "a real human face",              # [0] Real
        
        "a photo of a screen",            # [1] Spoof
        "a digital display",              # [2] Spoof
        "glossy reflection on screen",    # [3] Spoof
        "monitor bezel",                  # [4] Spoof
        "a printed paper face",           # [5] Spoof
        "a spoof face"                    # [6] Spoof
    ]
    text = tokenizer_clip(prompts).to(DEVICE)
    
    with torch.no_grad():
        image_features = model_clip.encode_image(img_tensor)
        text_features = model_clip.encode_text(text)
        
        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Probabilities
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = text_probs[0].cpu().numpy()
        
    # Real Class [0]
    real_prob = probs[0]
    # Sum of Spoof Classes [1:]
    spoof_prob = sum(probs[1:])  
    
    # Logic: Real must be significant
    # STRICT MODE: Raised to 0.50 (Real must be > Spoof probability)
    is_real = real_prob > settings.CLIP_THRESHOLD
    
    reason = "CLIP_Real" if is_real else f"CLIP_Spoof (Real={real_prob:.2f} vs Spoof={spoof_prob:.2f})"
    
    return {"is_real": is_real, "score": float(real_prob), "reason": reason, "spoof_score": float(spoof_prob)}

def infer_minifasnet_layer2(face_crop: np.ndarray) -> float:
    """Layer 2: MiniFASNet Texture Analysis"""
    if face_crop is None: return 0.0
    face_resized = cv2.resize(face_crop, (80, 80))
    face_transposed = np.transpose(face_resized, (2, 0, 1))
    img_tensor = torch.from_numpy(face_transposed).float().to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        prediction = model_mini(img_tensor)
        prob = F.softmax(prediction, dim=1)
        return prob[0][1].item() # Real Score

def predict(image: np.ndarray) -> PredictionResult:
    """
    3-LAYER PIPELINE (Optimized):
    Layer 1 (FFT): Physics Check (Moire Patterns)
    Layer 2 (OpenCLIP): Semantic Check (Screen/Digital/Glossy)
    Layer 3 (MiniFASNet): Texture Check (Triple Scale TTA)
    """
    if model_clip is None or model_mini is None:
        load_model()

    # Get Face Crop (Standard 2.7x Context)
    face_crop = get_face_crop(image, scale=2.7)
    if face_crop is None:
         return PredictionResult(decision="FAIL", mean_score=0.0, patch_max_score=0.0, local_variance=0.0, rgb_corr=0.0, freq_spike_ratio=0.0, glare_asym=0.0, evidence_count=1, evidence=["no_face_detected"], reason="No Face Detected")

    # === LAYER 1: FREQUENCY ANALYSIS (FFT) ===
    fft_result = analyze_frequency_domain(face_crop)
    if fft_result["is_anomaly"]:
         return PredictionResult(
            decision="FAIL",
            mean_score=0.0,
            patch_max_score=0.0,
            local_variance=0.0,
            rgb_corr=0.0,
            freq_spike_ratio=fft_result["score"],
            glare_asym=0.0,
            evidence_count=1,
            evidence=[f"Layer1_FFT_Rejected: {fft_result['reason']}", f"spike_ratio={fft_result['score']:.2f}"],
            reason="layer1_fft_moire_detected"
        )
    
    # === LAYER 2: OpenCLIP ===
    clip_result = infer_clip_layer1(face_crop)
    if not clip_result["is_real"]:
        return PredictionResult(
            decision="FAIL",
            mean_score=clip_result["score"],
            patch_max_score=0.0,
            local_variance=0.0,
            rgb_corr=0.0,
            freq_spike_ratio=0.0,
            glare_asym=0.0,
            evidence_count=2,
            evidence=[f"Layer1_FFT_Passed", f"Layer2_CLIP_Rejected: {clip_result['reason']}", f"spoof_prob={clip_result['spoof_score']:.2f}"],
            reason="layer2_clip_rejected"
        )
        
    # === LAYER 3: MiniFASNet (TRIPLE TTA) ===
    # TTA 1: Scale 2.7 (Context)
    score_context = infer_minifasnet_layer2(face_crop)
    
    # TTA 2: Scale 1.5 (Texture Standard - BEST PRACTICE)
    crop_15 = get_face_crop(image, scale=1.5)
    score_15 = infer_minifasnet_layer2(crop_15) if crop_15 is not None else 0.0
    
    # TTA 3: Scale 1.3 (Zoomed/Face)
    crop_13 = get_face_crop(image, scale=1.3)
    score_13 = infer_minifasnet_layer2(crop_13) if crop_13 is not None else 0.0
        
    # Consensus: STRICTEST wins (Minimum)
    final_score = min(score_context, score_15, score_13)
    
    # Decision
    decision = "FAIL"
    threshold = settings.MINIFASNET_THRESHOLD
    evidence = [f"Layer1_FFT_Passed", f"Layer2_CLIP_Passed ({clip_result['score']:.2f})", f"TTA_Scores=[{score_context:.2f}, {score_15:.2f}, {score_13:.2f}]"]
    
    reason = "model_confidence_low"

    if final_score > threshold:
        decision = "PASS"
        reason = "model_confidence_high"
        
        # Calculate Variance for stats
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F)
        lv = laplacian.var()
        
        if lv < 5: 
             decision = "FAIL"
             reason = "flat_texture_detected"
             evidence.append(f"low_variance={lv:.2f}")
    else:
        lv = 0.0 
        decision = "FAIL"
        reason = "layer3_minifasnet_rejected"
        evidence.append("minifasnet_score_too_low")

    return PredictionResult(
        decision=decision,
        mean_score=float(final_score),
        patch_max_score=0.0, 
        local_variance=round(lv, 2),
        rgb_corr=0.0,
        freq_spike_ratio=fft_result["score"],
        glare_asym=0.0,
        evidence_count=len(evidence),
        evidence=evidence,
        reason=reason
    )
