FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -U pip && pip install -e .

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8003/health || exit 1
CMD ["uvicorn", "pharmagpt_vn.api.main:app", "--host", "0.0.0.0", "--port", "8003"]
