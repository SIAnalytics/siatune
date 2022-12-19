ARG PYTORCH=1.9.0
ARG CUDA=11.1
ARG CUDNN=8
ARG MMCV_VERSION=1.4.7

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

# Install MMCV with MIM
ARG MMCV_VERSION
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir openmim \
 && mim install mmcv-full==$MMCV_VERSION

# Install SIATune
COPY . /siatune
WORKDIR /siatune
RUN pip install --no-cache-dir -e .
