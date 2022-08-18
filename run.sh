#!/bin/sh

# get variables from .env
export $(cat .env | xargs)

# create folders for shared volumes
mkdir -p data

# generate requirements.txt
pip-compile setup.py

# build and run container
docker-compose up --build # -d

# start bash on container
docker exec -it --user root ${CONTAINER_NAME} /bin/bash