# Getting started

1. Create an .env file in project's root folder and add:

```shell
CONTAINER_NAME=kabasalt
```

2. Build container and start docker container with ka_basalt (More info about ka_basalt here: https://github.com/BASALT-2022-Karlsruhe/ka_basalt)
```shell
sh run.sh
```
This starts the file `train.py`

# Docker 
## Logs
You can view the outputs of the container by doing:
```shell
docker logs -f kabasalt_experiment
```
