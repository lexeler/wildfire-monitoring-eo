# Образ инференса и обучения решения LitWin (Python 3.12; GPU через CUDA 12.4
# из колёс torch, нужен только драйвер NVIDIA на хосте; работает и на CPU).
#   docker build -t litwin-fire .
#   docker run --rm --gpus all -v /path/to/test:/data:ro -v "$PWD/out":/out litwin-fire
#   -> out/submission.csv
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# libgomp1 — OpenMP для LightGBM, libexpat1 — нужна колёсам rasterio (GDAL)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY firemon/ firemon/
COPY configs/ configs/
COPY scripts/ scripts/
COPY weights/ weights/
COPY tests/ tests/
COPY inference.py train.py ./

CMD ["python", "inference.py", "--data-dir", "/data", "--output", "/out/submission.csv"]
