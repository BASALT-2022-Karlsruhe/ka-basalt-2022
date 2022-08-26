#!/bin/sh

# create folders for shared volumes
mkdir -p data
mkdir -p train

[ ! -f .env ] || export $(sed 's/#.*//g' .env | xargs)

echo "Please insert password!"
sudo mkdir ${MODELS_ROOT}${NAME}_${VERSION}

# build and run container
docker-compose up --build -d