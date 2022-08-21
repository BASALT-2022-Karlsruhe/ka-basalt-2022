#!/bin/sh

# create folders for shared volumes
mkdir -p data

# generate requirements.txt
# pip-compile setup.py

# build and run container
docker-compose up --build -d