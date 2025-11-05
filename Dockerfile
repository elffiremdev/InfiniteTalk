# ---------------------------------------------------------
# 🚀 InfiniteTalk - Production Dockerfile
# Base image: CUDA 12.1 + Python 3.10 (Ubuntu 22.04)
# ---------------------------------------------------------

FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Sistem bağımlılıklarını yükle (minimal kurulum)
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Python ve temel araçlar
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && pip install --no-cache-dir --upgrade pip setuptools wheel

# Gereken Python bağımlılıklarını yükle
COPY requirements_replicate.txt .
RUN pip install --no-cache-dir -r requirements_replicate.txt

# Hugging Face CLI (CLI komutu için [cli] eklentisi şart)
RUN pip install --no-cache-dir "huggingface_hub[cli]" hf-transfer

# CUDA 12.x destekli PyTorch kurulumu (Resmî index URL ile)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio xformers

# Replicate Cog CLI
RUN pip install --no-cache-dir cog

# Uygulama dosyalarını kopyala
COPY . /app

# Ortam değişkenleri
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1
# Hugging Face token dışarıdan arg olarak alınabilir
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_TOKEN}

# ---------------------------------------------------------
# 🧠 Model indirimi runtime'da yapılır
# Kullanıcıya özel veya büyük modellerin build sırasında
# indirilmesi CI/CD ortamında disk dolumuna neden olur.
# Bunun yerine, ilk çalıştırmada indirilecektir.
# ---------------------------------------------------------

# Örnek olarak, ilk çalıştırmada modeli indirip cache'e alabilirsin:
# CMD ["bash", "-c", "huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir /app/weights/InfiniteTalk --ignore-patterns '.git*' && cog predict"]

# Ancak Cog zaten kendi 'predict' komutuyla entrypoint tanımlar:
ENTRYPOINT ["cog", "predict"]
