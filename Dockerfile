FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
    poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the Bina model into the image
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ARG BINA_MODEL_ID=Reza2kn/Bina-0.1-Koochik
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${BINA_MODEL_ID}', local_dir='/workspace/models/bina', local_dir_use_symlinks=False)" \
 && rm -rf /tmp/hf_cache

ENV BINA_MODEL_PATH=/workspace/models/bina
ENV MODEL_DEVICE=auto
ENV MODEL_DTYPE=bfloat16

COPY handler.py .

CMD ["python", "-u", "handler.py"]