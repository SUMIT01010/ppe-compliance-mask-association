# CPU-only image. MPS does not exist inside Docker and a CUDA base would quadruple the image for
# no benefit on the target (a small always-on inference box). Flip the torch index-url to a cu121
# wheel and set DEVICE=cuda if this is ever deployed to a GPU node.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEVICE=cpu \
    WEIGHTS=/app/weights/best.pt

# opencv (via ultralytics) needs these; without libgl1 the import fails at runtime, not build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

COPY config.py .
COPY src/ src/
COPY app/ app/

# Mount the trained detector at run time rather than baking a 22MB+ checkpoint into the layer:
#   docker run -p 8000:8000 -v $(pwd)/runs/detector_demo/weights:/app/weights ppe-compliance
# MobileSAM (38MB) is fetched by ultralytics on first ARM B request; pre-bake it for an
# air-gapped deploy by adding: RUN python -c "from ultralytics import SAM; SAM('mobile_sam.pt')"
VOLUME /app/weights

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
