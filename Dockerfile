FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgomp1 \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Keep pip's networking configuration compatible with the
# pip version shipped in the Python base image.
RUN python -m pip install --upgrade \
    --retries 10 \
    --timeout 300 \
    pip

# Install the exact CPU PyTorch version used by the project.
RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0

# Install the remaining runtime dependencies.
RUN python -m pip install \
    --retries 10 \
    --timeout 300 \
    --prefer-binary \
    fastapi==0.110.0 \
    "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 \
    pydantic==2.7.1 \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4 \
    torch-geometric==2.5.0 \
    pm4py==2.7.11

COPY app ./app
COPY src ./src
COPY artifacts/final_experiment ./artifacts/final_experiment

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
