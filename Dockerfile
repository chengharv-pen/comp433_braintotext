# ============================
# Stage 0 — Base CUDA
# ============================
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/conda/bin:$PATH

# System deps
RUN apt-get update && apt-get install -y \
    wget bzip2 git cmake build-essential \
    zlib1g-dev \
    libz-dev \
    libssl-dev \
    libffi-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh


# ============================
# Stage 1 — Build BOTH Conda envs
# ============================
FROM base AS builder
WORKDIR /workspace

# Copy full project (needed for -e . installs)
COPY . /workspace/

# Accept TOS for conda (only needed here)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ------------------------------
# Build b2txt25
# ------------------------------
RUN conda create -y -n b2txt25 python=3.10 && \
    bash -lc "\
        conda activate b2txt25 && \
        pip install --upgrade pip && \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
        pip install \
            redis==5.2.1 \
            jupyter==1.1.1 \
            numpy==2.1.2 \
            pandas==2.3.0 \
            matplotlib==3.10.1 \
            scipy==1.15.2 \
            scikit-learn==1.6.1 \
            tqdm==4.67.1 \
            g2p_en==2.1.0 \
            h5py==3.13.0 \
            omegaconf==2.3.0 \
            editdistance==0.8.1 \
            huggingface-hub==0.33.1 \
            transformers==4.53.0 \
            tokenizers==0.21.2 \
            accelerate==1.8.1 \
            bitsandbytes==0.46.0 \
            -e . \
        && conda clean -afy \
    "

# ------------------------------
# Build b2txt25_lm
# ------------------------------
RUN conda create -y -n b2txt25_lm python=3.9 && \
    bash -lc "\
        conda activate b2txt25_lm && \
        pip install --upgrade pip && \
        pip install \
            torch==1.13.1 \
            redis==5.0.6 \
            jupyter==1.1.1 \
            numpy==1.24.4 \
            matplotlib==3.9.0 \
            scipy==1.11.1 \
            scikit-learn==1.6.1 \
            tqdm==4.66.4 \
            g2p_en==2.1.0 \
            omegaconf==2.3.0 \
            huggingface-hub==0.23.4 \
            transformers==4.40.0 \
            tokenizers==0.19.1 \
            accelerate==0.33.0 \
            bitsandbytes==0.41.1 \
            -e . \
        && cd language_model/runtime/server/x86 && python setup.py install \
        && conda clean -afy \
    "


# ============================
# Stage 2 — Final Runtime Image
# ============================
FROM base
WORKDIR /workspace

# Copy only the conda environments (not the pip caches, temp files, builds)
COPY --from=builder /opt/conda/envs/b2txt25 /opt/conda/envs/b2txt25
COPY --from=builder /opt/conda/envs/b2txt25_lm /opt/conda/envs/b2txt25_lm


# Default environment
ENV DEFAULT_ENV=b2txt25
SHELL ["bash", "-lc"]
RUN echo "conda activate ${DEFAULT_ENV}" >> ~/.bashrc

# Install lsb-release and redis
RUN apt-get update && apt-get install -y curl gnupg lsb-release \
    && curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/redis.list \
    && apt-get update && apt-get install -y redis \
    && apt-get update && apt-get install -y nano \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash", "-il"]

# for runpod
#RUN apt-get update && apt-get install -y ttyd
#EXPOSE 7681
#CMD ["ttyd", "-p", "7681", "bash"]