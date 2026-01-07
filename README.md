# Image Spoofing Detection API

A robust FastAPI-based application for detecting face spoofing (anti-spoofing) using the **DeePixBiS** architecture. This API is designed to distinguish between real human faces and presentation attacks (screens, printed photos, etc.).

## 🚀 Features

*   **Deep Learning Model**: Uses a DenseNet161-based DeePixBiS model.
*   **High Accuracy**: Tuned thresholds achieve **98.25% accuracy on Real images** and **89.32% on Fake images**.
*   **Heuristic Analysis**: Includes additional checks for:
    *   Moiré patterns (screen frequency detection).
    *   Artificial color correlations.
    *   Asymmetric glare.
    *   Flat texture detection.
*   **Ready for Deployment**: Includes Docker support.

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

4.  **Run the server**:
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

### Docker Setup

1.  **Build and run**:
    ```bash
    docker-compose up --build
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

**Example Response:**

```json
{
    "success": true,
    "status_code": 200,
    "message": "Anti-spoofing analysis completed successfully",
    "data": {
        "decision": "FAIL",
        "mean_score": 0.4553,
        "patch_max_score": 0.6833,
        "local_variance": 242.6,
        "rgb_corr": 0.9688,
        "freq_spike_ratio": 0.00116,
        "glare_asym": 0.0,
        "evidence_count": 1,
        "evidence": [
            "strong_ai_spoof_detection"
        ],
        "reason": "ai_model_rejected"
    }
}
```

## 🧠 Model & Credits

This project uses the **DeePixBiS** (Deep Pixel-wise Binary Supervision) architecture.

*   **Original Repository**: [Saiyam26/Face-Anti-Spoofing-using-DeePixBiS](https://github.com/Saiyam26/Face-Anti-Spoofing-using-DeePixBiS)
*   The model weights used in this project are derived from the pre-trained weights provided in the repository above, fine-tuned for better performance in this specific API implementation.

## 🔧 Configuration

You can adjust settings in the `.env` file:

```ini
PROJECT_NAME="Image Spoofing API"
MODEL_PATH="models/DeePixBiS.pth"
PORT=8000
```
