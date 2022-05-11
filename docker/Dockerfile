ARG PYTORCH=1.6.0
ARG CUDA=10.1
ARG CUDNN=7
ARG MMCV_VERSION=1.4.2

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

# due to the issue https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV_VERSION
RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]

# Install MMTune
COPY . /mmtune
WORKDIR /mmtune
RUN pip install --no-cache-dir -e .
