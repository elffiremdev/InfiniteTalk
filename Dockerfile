# === Base image ===
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# === System dependencies ===
RUN apt-get update && apt-get install -y \
    git wget ffmpeg libgl1 libsndfile1 libavcodec-extra python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# === Python setup ===
WORKDIR /app
COPY requirements_replicate.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# === Hugging Face CLI ve hızlandırma opsiyonları ===
RUN pip install huggingface_hub hf-transfer==0.1.9.dev1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# === PyTorch (CUDA 12.1) ===
RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# === Model dosyalarını indir ===
RUN mkdir -p /app/weights && \
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir /app/weights/InfiniteTalk --ignore-patterns ".git*" && \
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /app/weights/Wan2.1-I2V-14B-480P --ignore-patterns ".git*" && \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir /app/weights/chinese-wav2vec2-base --ignore-patterns ".git*"

# === Kodu kopyala ===
COPY . /app
ENV PYTHONUNBUFFERED=1

# === Default komut ===
CMD ["python3", "predict.py"]
