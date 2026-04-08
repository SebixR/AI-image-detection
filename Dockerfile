FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_CACHE_DIR=/root/.cache/pip

# systemowy Python + narzędzia
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git wget curl vim htop \
    build-essential cmake \
    libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*
	

COPY requirements.txt /tmp/requirements.txt

# venv
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

CMD ["/bin/bash"]