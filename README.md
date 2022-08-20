# Getting started

0. Clone https://github.com/BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader somewhere *outside* of this project
   1. git clone git@github.com:BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader.git
   2. `cd ka-basalt-2022-datadownloader
   3. `run.sh` --> You should end up with a volume containing the downloaded demonstration data 
1. Move back to this project: ka-basalt-2022
2. Create an .env file in project's root folder and add:

```shell
CONTAINER_NAME=kabasalt_container # add your lastname to make it unique
IMAGE_NAME=kabasalt # add your lastname to make it unique
PORT=9998 
```

2. Build container and start docker container 
```shell
sh run.sh
```
3. This starts `bin/bash` on the container . From here you can now start e.g. `train.py` to train your agent

4. To be able to specify different GPUs for e.g. train.py, create a `docker-compose.override.yaml` file in the project's root folder and add the following:
```
services:
  kabasalt:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1'] # specify the device ids of the GPUs on your system, which will be used for training
              capabilities: [ gpu ]
```
Make sure you specify the correct system's device_ids of the GPUs you want to use for training.

# Docker 
## Logs
You can view the outputs of the container by doing:
```shell
docker logs -f kabasalt
