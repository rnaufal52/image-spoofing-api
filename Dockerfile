FROM python:3.10-slim

WORKDIR /app

# Mencegah Python membuat file .pyc dan memastikan log muncul segera
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Penting: Menjadikan folder /app sebagai root pencarian module
ENV PYTHONPATH=/app

# Install dependencies sistem untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Copy seluruh project
COPY . .

EXPOSE 8001

# Menjalankan uvicorn dengan format module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]