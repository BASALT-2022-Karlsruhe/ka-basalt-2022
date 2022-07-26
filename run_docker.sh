docker build -t kabasalt2022 .
# and run container
docker run -it --init  kabasalt2022
# start bash on container
# docker exec -it --user root <CONTAINER_NAME> /bin/bash