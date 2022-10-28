# Getting started

0. Clone https://github.com/BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader somewhere *outside* of this project. 
For example to a shared folder, where all on your server have access to:
1. Download the basalt data
   1. Clone `git clone git@github.com:BASALT-2022-Karlsruhe/ka-basalt-2022-datadownloader.git`
   2. Move into dir`cd ka-basalt-2022-datadownloader`
   3. Create an `.env` file and adjust the number of samples you'd like to download (see ReadMe.md)
   3. `run.sh` --> You should end up with a volume containing the downloaded demonstration data 
2. Move back to this project: ka-basalt-2022
   1. `cd ..`
   2. `cd ka-basalt-2022`
3. Create an `.env` file and adjust the parameters
   1. `NAME=<NAME>_<Goal>` # e.g. kulbach_baseline 
   2. `VERSION='0_0_1'` # Version of your experiments
   3. `MODELS_ROOT='/home/shared/BASALT/models'` # Folder where you expect and save your models.
   4. `PORT=9898`
   5. `PYTHONUNBUFFERED=1`
   6. `DATA_ROOT=data_wombat` 
   7. `GIT_ACCESS_TOKEN=YOUR_TOKEN_HERE_123`

where DATA_ROOT=data_wombat or DATA_ROOT=data 
- data_wombat: Loads data from volume on mounted shared wombat-server folder 
- data: Loads from volume on host server (Bison) 

4. Build container and start docker container
   1. `sh run.sh`
5. This starts `bin/bash` on the container . From here you can now start e.g. `train.py` to train your agent

6. To be able to specify different GPUs for e.g. train.py, change the gpu paramter within the `docker_compose.yaml` (DO NOT COMMIT CHANGES WITHIN THIS FILE!) for the graphics card you'd like to use.

# start bash on container
```shell
docker exec -it --user root basalt_container_${NAME}_${GOAL} /bin/bash
```

# Start Training process directly + inpect logs of container
In your docker-compose .yaml (or docker-compose.override.yaml), if you change entrypoint to:

`entrypoint: "python train.py"` 

and just start run.sh, then it will start the training process directly and you should be able to see the output via `docker logs -f CONAINER_NAME`

# Submitting Results 
[Official Tutorial https://github.com/minerllabs/basalt_2022_competition_submission_template/blob/main/README.md]

1. Go to https://gitlab.aicrowd.com/, navigate to "Preferences" -> "SSH Keys" and add an ssh-key to your profile. 
2. Create a private repo. 
3. Add it as a remote via `git remote add aicrowd git@gitlab.aicrowd.com:<user>/<repo>.git`. 
4. Modify the `aicrowd.json` file. Use `"debug": true` when testing the submission process.
5. Open bash in the docker container. 
6. Run `git lfs track train/*.weights`.
7. Check if the model-weights (and other large files you want to push) are marked with `Git LFS objects to be committed` when calling `git lfs status`.  
8. If the model weights are not tracked correctly, run `git lfs migrate info --everything --include="train/*.weights"` followed by `git add --renormalize .` and check again.
9. Commit. 
10. Push the branch you want to submit via `git push aicrowd <branch>` 
11. Create a git tag with `git tag -am "submission-<version>" submission-<version>`
12. Push the tag with `git push aicrowd submission-<version>`
13. Check the status of your submission in the issues section of the repository.

# Testing code on Debian without Docker

```shell
# Install dependency packages
sudo apt -qq update && xargs -a apt.txt apt -qq install -y --no-install-recommends \
 && rm -rf /var/cache/*

# Create conda environment
conda env create -n basalt -f environment.yml --prune

# Activate environment
conda activate basalt

# Test whether your code works
python <your-script>.py 
```
