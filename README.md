# Getting started

0. Clone https://github.com/BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader somewhere *outside* of this project
   1. `git clone git@github.com:BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader.git`
   2. `cd ka-basalt-2022-datadownloader`
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

# Docker 
## Logs
You can view the outputs of the container by doing:
```shell
docker logs -f kabasalt
