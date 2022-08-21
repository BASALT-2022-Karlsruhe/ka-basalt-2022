# Getting started

0. Clone https://github.com/BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader somewhere *outside* of this project. 
For example to a shared folder, where all on your server have access to:
   1. `git clone git@github.com:BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader.git`
   2. `cd ka-basalt-2022-datadownloader`
   3. Adjust the number of demonstrations you want to download for each environment directly in the Dockerfile.
   4. `run.sh` --> You should end up with a volume containing the downloaded demonstration data 
2. Move back to this project: ka-basalt-2022
3. create a `docker-compose.override.yaml` file in the project's root folder and add the following:
```yaml
services:
  kabasalt:
    image: kabasalt_container_ADD_UNIQUE_APPENDIX # make unique by adding e.g. lastname: kabasalt_container_laurito
    container_name: kabasalt_ADD_UNIQUE_APPENDIX  # make unique by adding e.g. lastname: kabasalt_laurito 
    ports:
      - 9998:9999 #  if necessary, change Port before the :
```

2. Build container and start docker container 
```shell
sh run.sh
```
3. This starts `bin/bash` on the container . From here you can now start e.g. `train.py` to train your agent

4. To be able to specify different GPUs for e.g. train.py, add the following deploy part under kabasalt: to your docker-compose.override.yaml
```yaml
services:
  kabasalt:
    ...
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
docker logs -f ka-basalt-2022
```

# start bash on container
```shell
docker exec -it --user root ${CONTAINER_NAME} /bin/bash
``
