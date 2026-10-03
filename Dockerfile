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
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8000"]
