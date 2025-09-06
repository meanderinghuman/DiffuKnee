# Dockerfile for DiffuKnee - diffusion-based knee MRI segmentation
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Copy repo
COPY . /workspace

# Install python deps
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Default command (train as default)
CMD ["python3", "train.py"]
