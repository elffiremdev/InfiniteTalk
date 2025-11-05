# Base image: lightweight + CUDA 12.x + Python 3.10
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libsndfile1 \
    libavcodec-extra \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Python kurulumu
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip setuptools wheel

# Gereken Python bağımlılıklarını yükle
COPY requirements_replicate.txt .
RUN pip install -r requirements_replicate.txt

# Hugging Face CLI kurulumunu ekle
RUN pip install huggingface_hub hf-transfer

# CUDA 12.x destekli PyTorch kurulumu
RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio xformers

# Ağırlıkları build aşamasında indir (cache’e dahil ediliyor)
RUN mkdir -p /app/weights && \
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir /app/weights/InfiniteTalk --ignore-patterns ".git*" && \
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /app/weights/Wan2.1-I2V-14B-480P --ignore-patterns ".git*" && \
    huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir /app/weights/chinese-wav2vec2-base --ignore-patterns ".git*"

# Uygulama dosyalarını kopyala
COPY . /app

# Replicate cog CLI
RUN pip install cog

# Ortam değişkenleri
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# Çalıştırılacak entrypoint
ENTRYPOINT ["cog", "predict"]
