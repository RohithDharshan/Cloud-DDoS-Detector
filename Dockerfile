FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; scikit-learn/pandas wheels are pure enough on
# python:3.11-slim that we don't need build-essential.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the running service needs: source + trained model artifacts + dashboard.
COPY src/ src/
COPY model/artifacts/ model/artifacts/
COPY frontend/ frontend/

ENV ARTIFACTS_DIR=model/artifacts
ENV FRONTEND_DIR=frontend
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/health').read()" || exit 1

# Shell form so $PORT expands — cloud platforms like Render/Cloud Run inject
# their own port via this env var at runtime; defaults to 8000 locally/Docker Compose.
CMD uvicorn src.service:app --host 0.0.0.0 --port ${PORT:-8000}
