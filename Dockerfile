FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV FORCE_CUDA="0"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip     \
    libgl1-mesa-glx \
    libsm6          \
    libxext6        \
    libxrender-dev  \
    libglib2.0-0    \
    git             \
    python3-dev     \
    python3-wheel

COPY . /yolo
WORKDIR /yolo

RUN pip3 install --upgrade pip \
    && pip3 install   \
        opencv-python \
        supervision   \
        mmengine==0.10.3 \
        setuptools    \
        openmim       \
        numpy==1.26.3 \
    && pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 \
    && mim install mmcv==2.0.0 \
    && pip3 install mmyolo==0.6.0 \
    && pip3 install transformers==4.36.2 \
    && pip3 install timm==0.6.13 \
    && pip3 install lvis==0.5.3

RUN cd /yolo/third_party/mmdetection-3.0.0 && pip3 install -e .

RUN pip3 install -e .

ENTRYPOINT [ "python3", "demo/predict.py" ]