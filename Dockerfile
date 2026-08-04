FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# opencv-python needs libGL/glib at the OS level even in a "headless" container
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY upscale.py .

ENTRYPOINT ["python", "upscale.py"]
CMD ["--help"]
