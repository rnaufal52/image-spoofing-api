# 🛡️ 3-Layer Hybrid Anti-Spoofing API

A robust, production-ready Anti-Spoofing API that combines **Physics (FFT)**, **Semantics (OpenCLIP)**, and **Texture (MiniFASNetV2)** analysis to detect presentation attacks (screens, paper, digital replays) with **96.74% accuracy**.

## 🚀 Architecture: The 3-Layer Guard

This system uses a "Swiss Cheese Model" approach where each layer covers the weaknesses of the others:

1.  **Layer 1 (Physics Guard): Frequency Domain Analysis (FFT)**
    *   **Goal:** Detects Moiré patterns (aliasing artifacts) common in LCD/OLED screens.
    *   **Method:** Fast Fourier Transform (FFT) -> High-Frequency Spike Ratio.
    *   **Action:** Rejects if `Spike Ratio > 4.0`.

2.  **Layer 2 (Semantic Guard): OpenCLIP (ViT-B-32)**
    *   **Goal:** Zero-Shot Semantic Classification.
    *   **Method:** CLIP checks if the image looks like *"a photo of a screen"*, *"digital display"*, etc., vs *"a real human face"*.
    *   **Action:** Rejects if `Spoof Probability > Real Probability`.

3.  **Layer 3 (Texture Guard): MiniFASNetV2 (Triple TTA)**
    *   **Goal:** Depth & Micro-Texture Analysis.
    *   **Method:** Runs the specialized MiniFASNetV2 model on **3 Test-Time Augmentations (TTA)**:
        *   **2.7x Crop:** Full Context (Head + Background).
        *   **1.5x Crop:** Standard Face.
        *   **1.3x Crop:** Zoomed Face (Detail).
    *   **Action:** Consenus Vote (Minimum score determines Pass/Fail).

## 📊 Performance (Validasi Akhir)

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Overall Accuracy** | **96.74%** | Tested on ~600 diverse samples. |
| **Real User Acceptance** | **98.83%** | Extremely friendly to real users (low False Rejection). |
| **Fake Detection** | **~95.8%** | Strong security against screens/prints. |
| **Latency** | **~0.6s** | Fast CPU inference (no GPU required). |

## 🛠️ Installation

### Local Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/rnaufal52/image-spoofing-api.git
    cd image-spoofing-api
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirement.txt
    ```
    *Note: This will install `torch`, `open_clip_torch`, `mediapipe`, `opencv-python`, etc.*

4.  **Run the server**:
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## 📚 Usage

### Endpoint: `/anti-spoof`

*   **Method**: `POST`
*   **Input**: Form-data with a file field named `file`.

**Example Request:**

```bash
curl -X POST "http://127.0.0.1:8000/anti-spoof" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/image.jpg"
```

**Example Response (PASS):**

```json
{
    "success": true,
    "status_code": 200,
    "message": "Anti-spoofing analysis completed successfully",
    "data": {
        "decision": "PASS",
        "mean_score": 0.892,
        "evidence_count": 3,
        "evidence": [
            "Layer1_FFT_Passed",
            "Layer2_CLIP_Passed (0.91)",
            "TTA_Scores=[0.95, 0.89, 0.92]"
        ],
        "reason": "model_confidence_high"
    }
}
```

**Example Response (FAIL):**

```json
{
    "success": true,
    "status_code": 200,
    "data": {
        "decision": "FAIL",
        "mean_score": 0.0,
        "evidence": [
            "Layer1_FFT_Passed",
            "Layer2_CLIP_Rejected: CLIP_Spoof (Real=0.10 vs Spoof=0.90)",
            "spoof_prob=0.90"
        ],
        "reason": "layer2_clip_rejected"
    }
}
```

## 🔧 Configuration

Adjust logic or paths in `app/core/config.py` or `.env`:

```ini
MODEL_PATH="models/2.7_80x80_MiniFASNetV2.pth"
DEVICE="auto" # or "cpu" / "cuda"
```

## 🧠 Models Used

1.  **OpenCLIP**: `ViT-B-32` (laion2b_s34b_b79k).
2.  **MiniFASNetV2**: Specialized Anti-Spoofing lightweight model.

---
*Built for High-Throughput & High-Security Production Environments.*
