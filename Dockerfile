FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Bake the default model into the image so the first request is fast (runtime download still works if this fails).
RUN mkdir -p models data && (python -c "from ultralytics import YOLO; YOLO('models/yoloe-11s-seg-pf.pt')" || echo "model pre-download skipped")
EXPOSE 8501
# Render provides $PORT; locally it falls back to 8501.
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
