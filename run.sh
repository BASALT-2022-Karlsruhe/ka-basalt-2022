#!/bin/sh

# create folders for shared volumes
mkdir -p data
mkdir -p train

[ ! -f .env ] || export $(sed 's/#.*//g' .env | xargs)
mkdir -p ${MODELS_ROOT}/${NAME}_${VERSION} # please make sure to create in a folder with access rights

# build and run container
docker-compose up -d