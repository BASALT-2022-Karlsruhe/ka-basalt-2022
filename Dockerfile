# In case you don't want to use Dockerfile and build from
# scratch, you can delete simply "Dockerfile" from your repository.
# Additional details on specifying dependencies are available in the README.

# Pre-installed conda and apt based MineRL runtime.
# This is done to save time during the submissions and faster debugging for you.
FROM aicrowd/base-images:minerl-22-base

# ---------------------------------------------
# Project-specific System Dependencies
# ---------------------------------------------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends xvfb
USER aicrowd

# Install needed apt packages
ARG DEBIAN_FRONTEND=noninteractive
USER root
COPY apt.txt apt.txt
RUN apt -qq update && xargs -a apt.txt apt -qq install -y --no-install-recommends \
 && rm -rf /var/cache/*

# Set the user and conda environment paths
USER aicrowd
ENV HOME_DIR /home/$USER
ENV CONDA_DEFAULT_ENV="minerl"
ENV PATH /home/aicrowd/.conda/envs/minerl/bin:$PATH
ENV FORCE_CUDA="1"

# Use MineRL environment
SHELL ["conda", "run", "-n", "minerl", "/bin/bash", "-c"]

# Conda environment update
COPY environment.yml environment.yml
RUN conda env update --name minerl -f environment.yml --prune

# Copy the files
COPY --chown=1001:1001 . /home/aicrowd

# ---------------------------------------------
# Build Python depencies and utilize caching
# ---------------------------------------------

RUN python3 --version
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install -r /home/aicrowd/requirements.txt

# ---------------------------------------------
# Download data
# ---------------------------------------------
## Demonstrations # TODO: Fix downloading demonstrations... Are they in the wrong place?
RUN python3 utils/download_dataset.py --json-file /home/aicrowd/utils/data_index_files/find-cave-Jul-28.json --output-dir /home/aicrowd/data/MineRLBasaltFindCave-v0 --num-demos 8
RUN python3 utils/download_dataset.py --json-file /home/aicrowd/utils/data_index_files/build-house-Jul-28.json --output-dir /home/aicrowd/data/MineRLBasaltBuildVillageHouse-v0 --num-demos 8
RUN python3 utils/download_dataset.py --json-file /home/aicrowd/utils/data_index_files/pen-animals-Jul-28.json --output-dir /home/aicrowd/data/MineRLBasaltCreateVillageAnimalPen-v0 --num-demos 8
RUN python3 utils/download_dataset.py --json-file /home/aicrowd/utils/data_index_files/waterfall-Jul-28.json --output-dir /home/aicrowd/data/MineRLBasaltMakeWaterfall-v0 --num-demos 8
## Model & Weights
RUN wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model -P /home/aicrowd/data/VPT-models
RUN wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights -P /home/aicrowd/data/VPT-models

RUN chown -R 1001:1001 /home/aicrowd/data/

CMD ["xvfb-run", "python3", "/home/aicrowd/train.py"]