#!/bin/sh

# create folders for shared volumes
mkdir -p data
mkdir -p train

# build and run container
docker-compose up --build -d