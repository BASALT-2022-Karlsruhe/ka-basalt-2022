#!/bin/sh

# create folders for shared volumes
mkdir -p data
mkdir -p train

[ ! -f .env ] || export $(sed 's/#.*//g' .env | xargs)

echo "Please insert password!"
sudo mkdir -p ${MODELS_ROOT}${NAME}_${VERSION}

# build and run container
docker-compose up --build -d