# API AdaptLab em modo demo (sem GPU, sem torch). Não validado neste PC: não há Docker local.
# Construir: docker build -t adaptlab-api .    Correr: docker run -p 8000:8000 adaptlab-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ADAPTLAB_MODE=demo

WORKDIR /app

# Só as dependências da API (sem torch/transformers).
RUN pip install --no-cache-dir \
    "fastapi==0.165.0" \
    "uvicorn==0.54.0" \
    "pydantic==2.14.4" \
    "pyyaml==6.0.3" \
    "scikit-learn==1.9.1" \
    "numpy==2.5.5"

COPY adaptlab ./adaptlab
COPY api ./api
COPY configs ./configs
COPY data/demo ./data/demo
COPY reports ./reports

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
