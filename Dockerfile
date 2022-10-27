FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

# Install needed apt packages
COPY apt.txt apt.txt
RUN apt -qq update && xargs -a apt.txt apt -qq install -y --no-install-recommends \
    && rm -rf /var/cache/*

# Create user home directory
ENV USER aicrowd
ENV HOME_DIR /home/$USER

# Replace HOST_UID/HOST_GUID with your user / group id
ENV HOST_UID 1001
ENV HOST_GID 1001

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# Set up user
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    ${USER}
USER ${USER}
WORKDIR ${HOME_DIR}
ENV CONDA_DIR ${HOME_DIR}/.conda
ENV PATH ${CONDA_DIR}/bin:${PATH}

# Download miniconda for python
RUN wget -nv -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
    && bash miniconda.sh -b -p ${CONDA_DIR} \
    && . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && rm -rf miniconda.sh \
    && conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -c pytorch \
    && conda clean -a -y
ENV FORCE_CUDA="1"

# Install needed pypi packages
USER ${USER}
COPY --chown=1001:1001 environment.yml environment.yml
RUN conda env create -f environment.yml

# Activate minerl environment
SHELL ["conda", "run", "-n", "minerl", "/bin/bash", "-c"]

ENV CONDA_DEFAULT_ENV="minerl"
ENV PATH /home/aicrowd/.conda/envs/minerl/bin:$PATH

# DONOT copy user files in the base image
# As participants have access to it.
# Copy user files
# COPY --chown=1001:1001 . ${HOME_DIR}
COPY --chown=1001:1001 Dockerfile-full ${HOME_DIR}/Dockerfile-full

# TODO: Add a README for users

