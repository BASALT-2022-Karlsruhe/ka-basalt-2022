# Getting started

1. Create an .env file in project's root folder and add:

```shell
CONTAINER_NAME=kabasalt # You can change the name if you have multiple containers...
```

2. Build container and start docker container 
```shell
sh run.sh
```
This starts the file `train.py`

# Docker 
## Logs
You can view the outputs of the container by doing:
```shell
docker logs -f kabasalt
